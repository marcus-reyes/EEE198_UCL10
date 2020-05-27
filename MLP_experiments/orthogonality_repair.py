# Import libraries
import os
import torch
import argparse
from networks_MLP import *
from utilities_MLP import *
from torchvision import datasets, transforms
from orthogonality_score import orthogonality
from plot_mask_trainability import plot_trainability

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument(
    "--ratio_prune", type=int, default=80, help="sparsity of masks to test"
)
parser.add_argument(
    "--trial", type=int, default=42, help="trial/experiment number"
)

def ortho_repair(model, device, mask_type=None, iterations=10000):
    """ Uses SGD to repair orthogonality/trainability of weights
        See page 7 of Signal Prop Perspective, Lee et al. 2020
    """
    model.train()
    optimizer = optim.SGD(model.parameters(),lr = 0.1)

    if mask_type is not None:
        print('Repairing:', mask_type.upper())

    for name, child_layer in model.named_children():
        if type(child_layer) == MaskedLinear:
            print('\tOrthogonalizing:',name)
            for i in range(iterations):
                optimizer.zero_grad()
                loss = orthogonality(child_layer,device,False)
                loss.backward()
                optimizer.step()
        
                if i%1000 == 0:
                    print('\tLoss:', loss.item())

def mass_ortho_repair(files, path, device, iterations=20000):
    """Applies ortho_repair to many SA pth/tar files"""

    for mask_type, filename in files.items():
        model = DeconsNet()

        chkpt = torch.load(filename, map_location=device)
        model.load_state_dict(chkpt['untrained_state_dict'])
        apply_mask_from_vector(model, chkpt['best_SA_mask'], device)

        ortho_repair(model, device, mask_type, iterations)

        torch.save(
            {   
                'repaired_init_weights': model.state_dict(),
                'SA_mask': chkpt['best_SA_mask'],
                'ave_acc': chkpt['ave_acc'],
                'iter': chkpt['iter'],
            },
            path + '/'+mask_type + '_repaired_weights.tar'
        )


if __name__ == '__main__':
    # load args
    args = parser.parse_args()

    # Load masks and init weight
    EXP_PATH = (
        os.getcwd()
        + "/sparsity_"
        + str(args.ratio_prune)
        + "/trial_"
        + str(args.trial)
    )
    # all SA filenames
    SA_files = {
        "SA3" : EXP_PATH + '/seed_44_best_SA_mask.pth',
        "SA-2x" : EXP_PATH + '/seed_46_best_SA_mask.pth',
    }
    # repair weights
    mass_ortho_repair(SA_files, EXP_PATH, DEVICE)
    
    # all criterion filenames
    files = {
        "snip": EXP_PATH + "/trained_snip_mlp.tar",
        "rand": EXP_PATH + "/trained_rand_mlp.tar",
        "lth": EXP_PATH + "/trained_lth_mlp.tar",
        "dcn": EXP_PATH + "/trained_dcn_mlp.tar",
        "SA": EXP_PATH + '/seed_46_best_SA_mask.pth',
        "full": None,
    }
    for mask_type in SA_files:
        files.update(
            {mask_type: EXP_PATH + '/'+mask_type + '_repaired_weights.tar'}
        )

    class HyperParams:
        lr = 1
        log_interval = 2
        epochs = 50
        batch_size = 64
        test_batch_size = 1000

    hp = HyperParams()

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=hp.batch_size,
        shuffle=True,
        # **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=hp.test_batch_size,
        shuffle=True,
        # **kwargs
    )

    # fine tune and plot
    plot_trainability(hp, files, EXP_PATH, DEVICE, train_loader, test_loader,
            same_init=False)


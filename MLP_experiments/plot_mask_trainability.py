import os
import torch
import argparse
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from networks_MLP import *
from utilities_MLP import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# parse script arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--ratio_prune", type=int, default=80, help="sparsity of masks to test"
)
parser.add_argument(
    "--trial", type=int, default=42, help="trial/experiment number"
)

def plot_trainability(
        hp, files, path, device, train_loader, test_loader, same_init = True
        ):
    """Plots loss curves of (masked) models

    Note: if same_init = False, no file can have None as filename 
          (since no init_weight will be set for that file otherwise)
    """

    # get any initialization (will be updated if same_init=False) 
    init_weights = torch.load(
                       files["lth"], map_location=device
                    )["init_state_dict"]
        
    for mask_type, filename in files.items():
        writer = SummaryWriter(path + "/logs/" + mask_type)

        model = DeconsNet()

        if filename is not None:
            chkpt = torch.load(filename, map_location=device)
            key = [k for k in chkpt.keys() if "mask" in k][0]
            if "SA" in key:  # if SA mask
                apply_mask_from_vector(model, chkpt[key], device)
            else:
                apply_mask_from_list(model, chkpt[key])

            if not same_init:
                init_weight = list(chkpt.values())[0] #init or repaired state_dict

        model.load_state_dict(init_weights)

        # fine tune
        print("\n===== Fine Tuning " + mask_type.upper() + " Model =====")
        optimizer = optim.Adadelta(model.parameters(), lr=hp.lr)
        for epoch in range(hp.epochs):
            train(hp, model, device, train_loader, optimizer, epoch, writer,
                    False)
            test(None, model, device, test_loader)
        writer.close()

if __name__ == "__main__":

    args = parser.parse_args()

    class HyperParams:
        lr = 1
        log_interval = 2
        epochs = 1
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

    EXP_PATH = (
        os.getcwd()
        + "/sparsity_"
        + str(args.ratio_prune)
        + "/trial_"
        + str(args.trial)
    )

    # filenames
    files = {
        "snip": EXP_PATH + "/trained_snip_mlp.tar",
        "rand": EXP_PATH + "/trained_rand_mlp.tar",
        "lth": EXP_PATH + "/trained_lth_mlp.tar",
        "dcn": EXP_PATH + "/trained_dcn_mlp.tar",
        "SA": EXP_PATH + "/trained_heur_mlp.tar",
        "full": None,
    }

    plot_trainability(hp, files, EXP_PATH, DEVICE, train_loader, test_loader)



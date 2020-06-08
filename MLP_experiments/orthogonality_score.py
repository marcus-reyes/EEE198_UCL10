# Import libraries
import os
import torch
import argparse
from networks_MLP import *
from utilities_MLP import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument(
    "--ratio_prune", type=int, default=80, help="sparsity of masks to test"
)
parser.add_argument(
    "--trial", type=int, default=42, help="trial/experiment number"
)

def orthogonality(layer,device,no_grad=True): # single layer 
    if no_grad:
        torch.set_grad_enabled(False)

    square = (torch.t(layer.weight*layer.mask) @ (layer.weight*layer.mask))
    identity = torch.eye(layer.weight.size(1)) # 784x784
    loss = torch.mean((square.to(device) - identity.to(device))**2)

    torch.set_grad_enabled(True) # revert back just in case

    return loss 

def compute_many_OS(file_names,device):
    """ Computes (per layer) Orthogonality Scores of (masked) models.
        See Signal Prop Perspective, Lee et al. 2020
    """

    # get any initialization (all are same)
    init_weights = torch.load(
                        files["lth"], map_location=device)["init_state_dict"]

    for mask_type, filename in file_names.items():
        model = DeconsNet()
        model.load_state_dict(init_weights)
        print(mask_type)

        sparsity = 0
        if filename is not None:
            chkpt = torch.load(filename, map_location=device)

            key = [k for k in chkpt.keys() if "mask" in k][0]

            on = sum([x.sum() for x in chkpt[key]])
            elems = sum([x.numel() for x in chkpt[key]])
            sparsity = 1. - (on/elems).item()

            if 'heur' in key:
                apply_mask_from_vector(model,chkpt[key],device)
            else:
                apply_mask_from_list(model,chkpt[key])

        print('\tSparsity:',sparsity)
        os = []
        for layer in model.children():
            os.append(orthogonality(layer, device).item())
        print('\tOrthogonality:',os,':',sum(os))

if __name__ == '__main__':
    args = parser.parse_args()

    # Load masks and init weight
    EXP_PATH = (
        os.getcwd()
        + "/sparsity_"
        + str(args.ratio_prune)
        + "/trial_"
        + str(args.trial)
    )

    # filenames
    files = {
        "SA": EXP_PATH + "/trained_heur_mlp.tar",
        "snip": EXP_PATH + "/trained_snip_mlp.tar",
        "dcn": EXP_PATH + "/trained_dcn_mlp.tar",
        "lth": EXP_PATH + "/trained_lth_mlp.tar",
        "rand": EXP_PATH + "/trained_rand_mlp.tar",
        "full": None,
    }

    compute_many_OS(files, DEVICE)

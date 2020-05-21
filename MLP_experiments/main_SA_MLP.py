# -*- coding: utf-8 -*-
""" Initial Simulated Annealing on MLP Experiments.

Previously a colab notebook. Tests the effectivity of using Simulated 
Annealing in finding supermasks, as defined in Deconstructing LTH paper. 

Original file is located at
    https://colab.research.google.com/drive/1SgCf6lvdVsKXXo9w2WZ4yoqudqJupHsj
"""

import os
import time
import copy
import math
import torch
import argparse

from collections import deque  # for memory
from pathlib import Path
from torchvision import datasets, transforms
import torch.optim as optim

from networks_MLP import *
from utilities_MLP import *

parser = argparse.ArgumentParser(description="Arguments for pruning exps")
parser.add_argument(
    "--ratio_prune", type = int, default = 80, help="amount to prune"
)
parser.add_argument(
    "--trial", type = int, default = 3, help="trial/experiment number"
)
parser.add_argument(
    "--k", type = int, default = 0, help="partial-train epochs"
)
parser.add_argument(
    "--reinit", help="change weights to signed constants", action='store_true'
)
parse_args = parser.parse_args()

SPARSITY = parse_args.ratio_prune
SPARSITY_PATH = os.getcwd() + "/sparsity_" + str(SPARSITY)
TRIAL = parse_args.trial
EXP_PATH = SPARSITY_PATH + "/trial_" + str(TRIAL)
if parse_args.k > 0:
    EXP_PATH = EXP_PATH + '_k_' + str(parse_args.k)
if parse_args.reinit:
    EXP_PATH = EXP_PATH + '_reinit'
Path(EXP_PATH).mkdir(parents=True, exist_ok=True)

EXP_PATH_TAR = SPARSITY_PATH + "/trial_2_may19"  # to get already existing tars

TRAINED_SNIP_PATH = EXP_PATH_TAR + "/trained_snip_mlp.tar"
TRAINED_RAND_PATH = EXP_PATH_TAR + "/trained_rand_mlp.tar"
TRAINED_LTH_PATH = EXP_PATH_TAR + "/trained_lth_mlp.tar"
TRAINED_DCN_PATH = EXP_PATH_TAR + "/trained_dcn_mlp.tar"
TRAINED_HEUR_PATH = EXP_PATH + "/trained_heur_mlp.tar"
BEST_MASK_PATH = EXP_PATH + "/best_mask_mlp.pth"  # achieved highest ave acc
PLOT_PATH = EXP_PATH + "/SA_plot.pdf"  # plot of SA optimization

FINE_TUNE = False  # boolean, set to False when saved models are present
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import Dataset & Args
class arguments:
    def __init__(
        self,
        batch_size=64,
        test_batch_size=1000,
        epochs=0,
        lr=1,
        gamma=0.7,
        seed=42,
        log_interval=6000,
    ):

        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma
        self.seed = seed
        self.log_interval = log_interval


args = arguments()
#kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
torch.manual_seed(args.seed)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=args.batch_size,
    shuffle=True,
    # **kwargs
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=args.test_batch_size,
    shuffle=True,
    # **kwargs
)


# Simulated Annealing for Early Accuracy

# =========================== HYPERPARAMETERS ==================================

## General Pruning
model = DeconsNet().to(DEVICE)  # network to work on
sparsity = SPARSITY / 100.0  # pruning sparsity
args.epochs = 50 # used as epochs to fine tune pruned models

## Simulated Annealing Search
# ham_dist value derived from mask sparsity later on
ham_dist_decay = 0.99  # decays search neighborhood size
acc_temp = 0.006  # uphill accept probability
acc_temp_decay = 0.999
iter_per_temp = 1  # allows multiple decisions per given temp value
iter_multiplier = 1.005  # increase iters for every temp decrease
max_iter_per_temp = 50  # ceiling on iterations per temp value

mem_size = 50  # number of masks to remember in set
acc_check_batch_size = 2000  # samples to check untrained accuracy
acc_check_shuffle_data = False  # will shuffle samples for every acc check
use_snip_init = False  # will use SNIP mask as init search mask if True

# =============================================================================

# Log information on experiment model

print("===== Test Model is", type(model), "=====")
print("With Layers:")
for child in model.children():
    print(child)
print()
print("To be pruned with:", sparsity, "sparsity")
untrained_state_dict = copy.deepcopy(model.state_dict())

# Get a fully trained model (for LT rewind, and Deconst pruning criterions)
trained_model = type(model)().to(DEVICE)
trained_model.load_state_dict(model.state_dict())
full_untrained_acc = test(args, trained_model, DEVICE, test_loader) #unmasked
full_acc = 0
if FINE_TUNE:
    print("\n===== Fine Tuning Full Model =====")
    optimizer = optim.Adadelta(trained_model.parameters(), lr=args.lr)
    accs = []
    for epoch in range(1, args.epochs + 1):
        train(args, trained_model, DEVICE, train_loader, optimizer, epoch)
        accs.append(test(args, trained_model, DEVICE, test_loader))
    full_acc = max(accs)
print("Best Untrained Accuracy:", full_untrained_acc)
print("Best Trained Accuracy:", full_acc)
print("Trained for:", args.epochs, "epochs")

# Get UPPER BOUND solution: SNIP
print("\n=== SNIP solution ===\n")
if FINE_TUNE:
    fine_tune_masked_copy(
        args,
        model,
        untrained_state_dict,
        "snip",
        sparsity,
        TRAINED_SNIP_PATH,
        train_loader,
        test_loader,
    )
print("=== Loads SNIP Model ===")
# snip_model = type(model)().to(DEVICE)
chkpt = torch.load(TRAINED_SNIP_PATH, map_location=DEVICE)
snip_init_weights = chkpt["init_state_dict"]
snip_masks = chkpt["snip_mask"]
# apply_mask_from_list(snip_model, snip_masks)
snip_acc = chkpt["trained_acc"]
snip_untrained_acc = chkpt["untrained_acc"]
print("Best Untrained Accuracy:", snip_untrained_acc)
print("Best Trained Accuracy:", snip_acc)
print("Trained for:", chkpt["epochs"], "epochs")

# Get LTH solution
print("\n=== LTH Solution ===\n")
if FINE_TUNE:
    fine_tune_masked_copy(
        args,
        model,
        untrained_state_dict,
        "post_mag",
        sparsity,
        TRAINED_LTH_PATH,
        train_loader,
        test_loader,
        trained_model
    )
print("=== Loads LTH Model ===")
# lth_model = type(model)().to(DEVICE)
chkpt = torch.load(TRAINED_LTH_PATH, map_location=DEVICE)
lth_init_weights = chkpt["init_state_dict"]
lth_masks = chkpt["post_mag_mask"]
# apply_mask_from_list(lth_model, lth_masks)
lth_acc = chkpt["trained_acc"]
lth_untrained_acc = chkpt["untrained_acc"]
print("Best Untrained Accuracy:", lth_untrained_acc)
print("Best Trained Accuracy:", lth_acc)
print("Trained for:", chkpt["epochs"], "epochs")

# Get Deconst solution
print("\n=== Deconst Solution ===\n")
if FINE_TUNE:
    fine_tune_masked_copy(
        args,
        model,
        untrained_state_dict,
        "mag_sign",
        sparsity,
        TRAINED_DCN_PATH,
        train_loader,
        test_loader,
        trained_model
    )
print("=== Loads Deconst Model ===")
# dcn_model = type(model)().to(DEVICE)
chkpt = torch.load(TRAINED_DCN_PATH, map_location=DEVICE)
dcn_init_weights = chkpt["init_state_dict"]
dcn_masks = chkpt["mag_sign_mask"]
# apply_mask_from_list(dcn_model, dcn_masks)
dcn_acc = chkpt["trained_acc"]
dcn_untrained_acc = chkpt["untrained_acc"]
print("Best Untrained Accuracy:", dcn_untrained_acc)
print("Best Trained Accuracy:", dcn_acc)
print("Trained for:", chkpt["epochs"], "epochs")

# Get LOWER BOUND solution: Rand
print("\n=== Initial Rand Solution ===\n")
if FINE_TUNE:
    fine_tune_masked_copy(
        args,
        model,
        untrained_state_dict,
        "rand",
        sparsity,
        TRAINED_RAND_PATH,
        train_loader,
        test_loader,
    )
print("=== Loads Rand Model ===")
# rand_model = type(model)().to(DEVICE)
chkpt = torch.load(TRAINED_RAND_PATH, map_location=DEVICE)
rand_init_weights = chkpt["init_state_dict"]
rand_masks = chkpt["rand_mask"]
# apply_mask_from_list(rand_model, rand_masks)
rand_acc = chkpt["trained_acc"]
rand_untrained_acc = chkpt["untrained_acc"]
print("Best Untrained Accuracy:", rand_untrained_acc)
print("Best Trained Accuracy:", rand_acc)
print("Trained for:", chkpt["epochs"], "epochs")


print("\n======= ! Begin Annealing ! =======\n")
#DEVICE = "cpu"  # use CPU for annealing
#model.to(DEVICE)

# For logging and other variable initialization:
ave_acc = 5
best_ave_acc = ave_acc
prev_masks = get_mask_mag(
    torch.rand((1, 784*300 + 300*100 + 100*10)), sparsity
)  # note this is a different randmask
ham_dist = int(prev_masks.sum())
prev_ham_dist = ham_dist  # for triangular decay

init_ham_dist = ham_dist  # for printing of hyperparams
init_acc_temp = acc_temp  # for printing of hyperparams
init_iter_per_temp = iter_per_temp  # for printing of hyperparams
prev_iter_per_temp = iter_per_temp

log_mod = 300  # will log every log_mod
prev_log = 0
total_iters = 0  # sum of the ff 3
total_up_steps = 0  # to get rate of up step acceptance
up_steps = 0  # three possible decisions per iteration w/in a temp val
down_steps = 0
no_steps = 0

ave_acc_record = []  # for plotting
acc_temp_record = []
ham_dist_record = []
upstep_rate_record = []

accs = [ave_acc]  # used to record accs within temp iters
closed_q = deque(maxlen=mem_size)  # memory

print("Hyperparams:")
print("\tHamming distance per step:", ham_dist * 2)
print("\tHamming distance decay:", ham_dist_decay)
print("\tInit Acc temp:", acc_temp)
print("\tAcc decay:", acc_temp_decay)
print("\tInit iter/temp:", iter_per_temp)
print("\n----------------------------")
print()

# Additional Initial Conditions
if use_snip_init:
    print("Loaded initial conditions from SNIP")
    untrained_state_dict = snip_init_weights
    # prev_masks = copy.deepcopy(snip_masks) # initial soln
else:
    print("Loaded initial conditions from RAND")
    untrained_state_dict = rand_init_weights
    # prev_masks = copy.deepcopy(rand_masks) # initial soln

# INITIAL MASK
new_masks = prev_masks  # initialize

model.load_state_dict(untrained_state_dict)  # copy common init weights
if parse_args.reinit:
    print('Change weights to signed constants')
    for child in model.children():
        with torch.no_grad():
            child.weight = torch.nn.Parameter(
                                child.weight.std() * child.weight.sign()
                            )

apply_mask_from_vector(model, new_masks, DEVICE)

if parse_args.k > 0:
    print("\n===== Partial ({}) Training SA Model =====".format(parse_args.k))
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    accs = []
    for epoch in range(1, parse_args.k + 1):
        train(args, model, DEVICE, train_loader, optimizer, epoch)
        accs.append(test(args, model, DEVICE, test_loader))
    k_acc = max(accs)
    print("k-th Accuracy:", k_acc)
    print("Pre-trained for:", args.epochs, "epochs")

mask_wrapper = MaskWrapper(new_masks, ham_dist)
closed_q.append(copy.deepcopy(mask_wrapper))

SA_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=acc_check_batch_size,
    shuffle=acc_check_shuffle_data,
    # **kwargs
)

# SA loop
# torch.manual_seed(42)
start_time = time.time()
while total_iters < 80000:  
    for _ in range(int(iter_per_temp)):
        # new_masks = step_from([prev_masks],ham_dist)[0]
        for _ in range(mem_size):
            mask_wrapper.mask = step_from([prev_masks], ham_dist)[0]
            if mask_wrapper in closed_q:
                pass
            else:
                break

        closed_q.append(copy.deepcopy(mask_wrapper))
        new_masks = mask_wrapper.mask
        apply_mask_from_vector(model, new_masks, DEVICE)
        new_acc = forward_pass(model, DEVICE, SA_loader, 1)
        ave_acc = sum(accs) / len(accs)

        acc_delta = (ave_acc - new_acc) / 100
        if acc_delta < 0:
            # take action
            prev_masks = new_masks
            accs.append(new_acc)
            down_steps += 1
        elif acc_delta >= 0:
            # probably take action based on acc_prob
            total_up_steps += 1
            allow_uphill = True
            if allow_uphill:
                q = torch.rand(1).item()
                acc_prob = math.exp(
                    -1 * acc_delta / acc_temp
                )  # larger delta smaller accept prob
                if q < acc_prob:
                    prev_masks = new_masks
                    accs.append(new_acc)
                    up_steps += 1
                else:
                    # don't take action
                    no_steps += 1
    accs = [ave_acc]  # reset to the last average

    # SCHEDULER of NEIGHBOR SIZE
    ham_dist = max(1, int(ham_dist * ham_dist_decay))
    if ham_dist < prev_ham_dist * 0.5:
        ham_dist = int(prev_ham_dist * 0.75)  # increase by 25%
        prev_ham_dist = ham_dist
    mask_wrapper.ham_dist = ham_dist  # adjust __eq__() ham_dist basis

    # SCHEDULER of TEMP
    acc_temp *= acc_temp_decay  # decrease temp

    # SCHEDULER of NUM of TRIALS within a temp value
    iter_per_temp = min(max_iter_per_temp, iter_per_temp * iter_multiplier)

    # ============================= LOGGING ============================
    sum_iters = down_steps + total_up_steps
    total_iters += sum_iters
    if total_iters - prev_log >= log_mod:
        elapsed_time = time.time() - start_time
        print("Time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        print("Iter", total_iters, ": ")
        print("\tAve Accuracy:", ave_acc, "%")
        print(
            "\tDown-Up-No steps: {:.2f}% - {:.2f}% - {:.2f}%".format(
                (down_steps / sum_iters) * 100,
                (up_steps / sum_iters) * 100,
                (no_steps / sum_iters) * 100,
            )
        )
        print("\tham_dist:", ham_dist * 2)
        print("\tAcc Temp:", acc_temp)
        total_up_steps = max(1, total_up_steps)
        print(
            "\tUpstep Accept Rate: {:.2f}%".format(
                (up_steps / total_up_steps) * 100
            )
        )
        # of the upsteps, how much does acc_prob accepts?
        print("\titer_per_temp", int(iter_per_temp))
        print("\tmemory_size", len(closed_q))

        ave_acc_record.append(ave_acc)
        ham_dist_record.append(ham_dist * 2)

        down_steps = 0
        total_up_steps = 0
        up_steps = 0
        no_steps = 0
        prev_log = total_iters

    # ================== SAVE MODEL on condition =======================
    if ave_acc > best_ave_acc:
        torch.save(
            {
                "untrained_state_dict": model.state_dict(),
                "heur_mask": prev_masks,  # may make this a list compre
                "k": parse_args.k,
                "ave_acc": ave_acc,
                "iter": total_iters,
            },
            BEST_MASK_PATH,
        )
        best_ave_acc = ave_acc

elapsed_time = time.time() - start_time
print("Elapsed time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


# ================== Fine Tune Found Model and See Results ====================

import matplotlib.pyplot as plt


# switch back to cuda
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load best found model
chkpt = torch.load(BEST_MASK_PATH, map_location=DEVICE)
print("\nLoaded heuristic model with ave acc:", chkpt["ave_acc"])
partial_k = chkpt["k"]
model.load_state_dict(chkpt["untrained_state_dict"])
apply_mask_from_vector(model, chkpt["heur_mask"], DEVICE)
total_iterations = chkpt["iter"]
model.to(DEVICE)

# test rewind accuracy
untrained_acc = test(args, model, DEVICE, test_loader)

print("\n===== Fine Tuning Heuristic Model =====")
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
accs = [0]
for epoch in range(1, args.epochs + 1 - partial_k):
    train(args, model, DEVICE, train_loader, optimizer, epoch)
    accs.append(test(args, model, DEVICE, test_loader))

torch.save(
    {
        "trained_state_dict": model.state_dict(),
        "heur_mask": prev_masks,  # may make this a list compre
        "k": parse_args.k,
        "trained_acc": max(accs),
        "epochs": epoch,
    },
    TRAINED_HEUR_PATH,
)

# Print results to textfile
import sys

sys.stdout = open(EXP_PATH + "/results.txt", "w")

print("All fine-tuned for", args.epochs, "epochs")
print("Heur:")
print(
    "\tOptimization time:",
    time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),
)
print("\tTotal Iterations:", total_iterations)
print("\tSA ave_acc:", chkpt["ave_acc"])
print("\tUntrained Accuracy:", untrained_acc)
print("\tBest Trained Accuracy:", max(accs))
print("Snip:")
print("\tUntrained Accuracy:", snip_untrained_acc)
print("\tBest Trained Accuracy:", snip_acc)
print("Decons:")
print("\tUntrained Accuracy:", dcn_untrained_acc)
print("\tBest Trained Accuracy:", dcn_acc)
print("LTH:")
print("\tUntrained Accuracy:", lth_untrained_acc)
print("\tBest Trained Accuracy:", lth_acc)
print("Rand:")
print("\tUntrained Accuracy:", rand_untrained_acc)
print("\tBest Trained Accuracy:", rand_acc)
print("Full:")
print("\tUntrained Accuracy:", full_untrained_acc)
print("\tBest Trained Accuracy:", full_acc)

print()

print("SA Hyperparams:")
print("\tsparsity:", sparsity)
print("\tMemorySize:", mem_size)
print("\tSA_batch_size:", acc_check_batch_size)
print("\tinit_ham_dist:", init_ham_dist)
print("\tham_dist_decay:", ham_dist_decay)
print("\tinit acc_temp:", init_acc_temp)
print("\tacc_temp_decay:", acc_temp_decay)
print("\tinit iter_per_temp:", init_iter_per_temp)
print("\titer_temp_multiplier:", iter_multiplier)
print("\tk:",parse_args.k)
print("\tsigned_constant:",str(parse_args.reinit))
print()


# plot accuracy and temperature
fig, ax1 = plt.subplots()

color = "tab:red"
ax1.set_xlabel("Iteration per {}".format(log_mod))
ax1.set_ylabel("Average Accuracy", color=color)
ax1.plot(ave_acc_record, color=color)
ax1.tick_params(axis="y", labelcolor=color)
ax1.grid(True)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = "tab:blue"
ax2.set_ylabel("Ham Dist", color=color)
ax2.plot(ham_dist_record, color=color)
ax2.tick_params(axis="y", labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(PLOT_PATH)

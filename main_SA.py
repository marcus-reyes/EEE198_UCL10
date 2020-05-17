# main_SA

import torch
import time
import os

# import copy
import math
from environment import PruningEnv
import logging

# import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utilities import *

import argparse

from collections import deque

xp_num_ = 700
date = 15
# Word description for marking the run labels in tensorboard
# Do not use anything other than letters, numbers, and _. "."
# messes with tensorboard

description = "90_sparse_large_ham_init_1"
writer = SummaryWriter(
    ("runs_SA_may_15/experiment_0_7_rand" + str(xp_num_) + str(description))
)


# Argument parsing
parser = argparse.ArgumentParser(description="Arguments for masker")
parser.add_argument(
    "--foldername",
    type=str,
    default="trash",
    help="folder to store masked networks in",
)
parser.add_argument(
    "--ratio_prune", type=float, default=0.7, help="amount to prune"
)
parser.add_argument(
    "--num_batches",
    type=int,
    default=1,
    help="number of batches for the search evaluation forward pass",
)
parser.add_argument(
    "--max_temp_changes", type=int, default=2, help="maximum temp levels"
)
args = parser.parse_args()

# ratio_prune = args.ratio_prune
# test_set_batches = args.num_batches

# Initialize model to be pruned and corresponding methods
env = PruningEnv()
env.reset_to_init_1()

# Obtain conv layers of the model and it's size
total_filters_count = 0
size_of_layer = []
for name, param in env.model.named_parameters():
    if "conv" in name and "weight" in name:
        total_filters_count += param.shape[0]
        size_of_layer.append(param.shape[0])

# Setting the initial mask and its sparsity
rand_values = torch.rand((total_filters_count))
mask_rank = torch.topk(
    rand_values, int(rand_values.shape[0] * args.ratio_prune), largest=False
)
mask = torch.ones((total_filters_count))
mask[mask_rank[1]] = 0
current_mask = mask

# Iteration variables
temp_changes = 0  # counter accept temp changes 
total_iter_count = 0  # iters including multiple loops within a temp change

# Model accuracy variables
ave_acc = 5
accs = [ave_acc]

# Search step variables, for logging
up_steps = 0  # away from local optimum
down_steps = 0  # towards local optimum
no_steps = 0  # stay put

# Search memory variables
mem_size = 60
mem_attempts = 30  # num of attempts to find mask that's not in mem
closed_q = deque(maxlen=mem_size)
closed_q.append(current_mask)

# HYPERPARAMS: hamming distance / search neighborhood
ham_dist = int(mask.sum())
ham_dist_decay = 0.99
prev_ham_dist = ham_dist

# HYPERPARAMS: up-step accept-probablity temperature
temp = 0.005
temp_decay = 0.995

# HYPERPARAMS: iterations within a temperature value
iter_per_temp = 1  # allows multiple decisions per given temp value
iter_multiplier = 1.005  # increase iters for every temp decrease
max_iter_per_temp = 10  # ceiling on iterations per temp value

###LogFile
###Create directory if it doesn't exist
if not os.path.exists("textlogs"):
    os.makedirs("textlogs")

###Pre-run data
log_file = open(
    "textlogs/test_may_" + str(date) + "_exp_" + str(xp_num_) + ".txt", "w"
)
log_file.write(
    os.getcwd()
    + "/masked_may_12/SA"
    + str(args.ratio_prune)
    + "_"
    + str(xp_num_)
    + "_"
    + str(description)
    + ".pth\n"
)
log_file.write("Hyperparameters\n")
log_file.write(str("temp: " + str(temp) + "\n"))
log_file.write(str("temp_decay: " + str(temp_decay) + "\n"))
log_file.write(str("iter_per_temp: " + str(iter_per_temp) + "\n"))
log_file.write(str("iter_multiplier: " + str(iter_multiplier) + "\n"))
log_file.write(str("max_iter_per_temp: " + str(max_iter_per_temp) + "\n"))
log_file.write(str("ham_dist_init: " + str(ham_dist) + "\n"))
log_file.write(str("ham_dist_decay: " + str(ham_dist_decay) + "\n"))
log_file.write(str("mem_size: " + str(mem_size) + "\n"))
log_file.close()

print("\n-------------- Simulated Annealing --------------\n")
start_time = time.time()
while temp_changes != args.max_temp_changes:
    # Find a new TEST mask
    for _ in range(int(iter_per_temp)):
        total_iter_count = total_iter_count + 1
        for _ in range(mem_attempts):
            new_mask = step_from(current_mask, ham_dist)
            if is_in_memory(new_mask, closed_q):
                pass
            else:
                closed_q.append(new_mask)
                break

        # Tentatively implement the mask
        idx = 0
        total_pruned = 0
        env.reset_to_init_1()
        for i in range(len(size_of_layer)):
            env.layer = env.layers_to_prune[i]
            layer_mask = new_mask[idx : idx + size_of_layer[i]].clone()
            layer_mask = torch.unsqueeze(layer_mask, 0)
            total_pruned += size_of_layer[i] - layer_mask.sum()

            filters_counted, pruned_counted = env.prune_layer(layer_mask)
            idx += size_of_layer[i]
        amount_pruned = total_pruned
        idx = 0

        # Check if keep or discard
        new_acc = env.forward_pass(args.num_batches)
        ave_acc = sum(accs) / len(accs)
        acc_delta = (ave_acc - new_acc) / 100
        if acc_delta < 0:
            down_steps += 1
            current_mask = new_mask
            accs.append(new_acc)
            ave_acc = sum(accs) / len(accs)
        else:
            q = torch.rand(1).item()
            acc_prob = math.exp(-1 * acc_delta / temp)
            print(acc_prob, "Acc_prob")
            if q < acc_prob:
                up_steps += 1
                current_mask = new_mask
                accs.append(new_acc)
                ave_acc = sum(accs) / len(accs)
            else:
                no_steps += 1
    accs = [ave_acc]

    # Logging progress on stdout
    elapsed_time = time.time() - start_time
    print("-------------------------------------------------")
    print("TIME", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    print("ITER:", total_iter_count)
    print("TEMP CHANGES:", temp_changes)
    print("\tAVE ACC:", accs[0])
    print(
        "\tSTEPS down-up-no: {:.2f}% - {:.2f}% - {:.2f}%".format(
            (down_steps / int(iter_per_temp)) * 100,
            (up_steps / int(iter_per_temp)) * 100,
            (no_steps / int(iter_per_temp)) * 100,
        )
    )
    print("\tHAM_DIST:", ham_dist)
    print("\tTEMP:", temp)
    print("\tITER_p_TEMP:", iter_per_temp)
    down_steps, up_steps, no_steps = 0, 0, 0

    # Logging data with tensorboard
    writer.add_scalar("ave_acc", ave_acc, temp_changes)
    writer.add_scalar("new_acc", new_acc, temp_changes)
    writer.add_scalar("iter_per_temp", int(iter_per_temp), temp_changes)
    writer.add_scalar("ham_dist", ham_dist, temp_changes)
    writer.add_scalar("amount_pruned", amount_pruned, temp_changes)

    # Schedulers of Neighbor Size, temperature, and iters per temp
    ham_dist = max(1, int(ham_dist * ham_dist_decay))
    if ham_dist < prev_ham_dist * 0.5:
        ham_dist = int(prev_ham_dist * 0.75)
        prev_ham_dist = ham_dist
    temp = temp * temp_decay
    iter_per_temp = min(max_iter_per_temp, iter_per_temp * iter_multiplier)

    temp_changes += 1

print("\n---------------- End of SA Search ---------------\n")


final_acc = env._evaluate_model()
print("Last tried (not accepted) mask has ", final_acc)

###Prune with the last ACCEPTED mask
###Apply the last mask accepted
idx = 0
total_pruned = 0
env.reset_to_init_1()

for i in range(len(size_of_layer)):
    env.layer = env.layers_to_prune[i]
    layer_mask = current_mask[idx : idx + size_of_layer[i]].clone()
    layer_mask = torch.unsqueeze(layer_mask, 0)
    total_pruned += size_of_layer[i] - layer_mask.sum()

    ###prune current layer
    filters_counted, pruned_counted = env.prune_layer(layer_mask)
    idx += size_of_layer[i]

amount_pruned = total_pruned
idx = 0

###Check the amount per layer
###Record the per layer_mask
layer_mask = []  # list
num_per_layer = []
for module in env.model.modules():
    # for conv2d obtain the filters to be kept.
    if isinstance(module, nn.BatchNorm2d):
        weight_copy = module.weight.data.clone()
        filter_mask = weight_copy.gt(0.0).float()
        layer_mask.append(filter_mask)

for i, item in enumerate(layer_mask):
    ###Have to use.item for singular element tensors to extract the element
    ###Have to use int()
    num_per_layer.append(int(item.sum().item()))

print(num_per_layer)
total = 0
for item in num_per_layer:
    total += item

print(total)


###Create folder if it does not exist
if not os.path.exists("masked_may_exp"):
    os.makedirs("masked_may_exp")

(
    layer_weights_dict,
    num_weights,
    layer_flops_dict,
    num_flops,
) = compute_weights_and_flops(
    get_network_def_from_model(env.model, [3, 32, 32])
)


###Save into .pth
PATH = (
    os.getcwd()
    + "/masked_may_exp/SA"
    + str(args.ratio_prune)
    + "_"
    + str(xp_num_)
    + "_.pth"
)
model_dicts = {
    "state_dict": env.model.state_dict(),
    "optim": env.optimizer.state_dict(),
    "filters_per_layer": num_per_layer,
}
torch.save(model_dicts, PATH)


###Sanity checks at the end
final_acc = env._evaluate_model()
final_forpass = env.forward_pass(args.num_batches)
print("Final accuracy is ", final_acc)
print("Final forward pass is ", final_forpass)
elapsed_time = time.time() - start_time
print("Elapsed time is", elapsed_time)
writer.close()


###Post-run data
log_file = open(
    "textlogs/test_may_" + str(date) + "_exp_" + str(xp_num_) + ".txt", "a"
)
log_file.write(str("up_steps: " + str(up_steps) + "\n"))
log_file.write(str("down_steps: " + str(down_steps) + "\n"))
log_file.write(str("no_steps: " + str(no_steps) + "\n"))
log_file.write(str("num_iterations: " + str(total_iter_count) + "\n"))
log_file.write(str("temps_tried: " + str(temp_changes) + "\n"))
log_file.write(str("num_batches: " + str(10) + "\n"))
log_file.write(str("final_structure: " + str(num_per_layer) + "\n"))


log_file.write(
    str("last_ave_acc: (NOT NECESSARILY THE ACTUAL ACC): " + str(accs) + "\n")
)
log_file.write(str("evaluated accuracy: " + str(final_acc) + "\n"))
log_file.write(str("Final forwardpass accuracy: " + str(final_forpass) + "\n"))

log_file.write(str("Time taken in seconds: " + str(elapsed_time) + "\n"))
log_file.close()

print("weights and flops", num_weights, num_flops)

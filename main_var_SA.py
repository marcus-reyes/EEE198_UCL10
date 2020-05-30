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

xp_num_ = 1



# Argument parsing
parser = argparse.ArgumentParser(description="Arguments for masker")
parser.add_argument(
    "--foldername",
    type=str,
    default="trash",
    help="folder to store masked networks in",
)
parser.add_argument(
    "--ratio_prune", type = float, default = 0.7, help="amount to prune"
)
parser.add_argument(
    "--num_batches",
    type=int,
    default=1,
    help="number of batches for the search evaluation forward pass",
)
parser.add_argument(
    "--max_temp_changes", type = int, default = 1000, help="maximum temp levels"
)
parser.add_argument(
    "--xp_num_", type = int, default = 100, help="experiment number"
)
parser.add_argument(
    "--method", type = str, default = "SA", help="method to use"
)
parser.add_argument(
    "--k_epoch", type = int, default = 5, help = "which k to reset to"
)
parser.add_argument(
    "--entire_eval_flag", type = int, default = 0, help = "1 if entire"
)
parser.add_argument(
    "--var_seed", type = int, default = 0, help = "for seeding of variance exp"
)
args = parser.parse_args()
xp_num_ = args.xp_num_
torch.manual_seed(args.var_seed)

# Word description for marking the run labels in tensorboard
# Do not use anything other than letters, numbers, and _. "."
# messes with tensorboard

description = "90_sparse_large_ham_init_1"
writer = SummaryWriter(
    ("runs_SA_may_exp/SA_exp_"
    + str(args.xp_num_)
    + "_"
    + str(int(args.ratio_prune*100)))
)
# Initialize model to be pruned and corresponding methods
env = PruningEnv()
if args.k_epoch == 0:
    env.reset_to_k_0()
    
elif args.k_epoch == 2:
    env.reset_to_k_2()
    
elif args.k_epoch == 5:
    env.reset_to_k_5()
    
elif args.k_epoch == 90:
    env.reset_to_k_90()
    
else:
    env.reset_to_init_1()

# Setting the initial mask and its sparsity
rand_values = torch.rand((env.total_filters))
mask_rank = torch.topk(
    rand_values, int(rand_values.shape[0] * args.ratio_prune), largest=False
)
mask = torch.ones((env.total_filters))
mask[mask_rank[1]] = 0
current_mask = mask
best_mask = current_mask
# Iteration variables
temp_changes = 0  # counter accept temp changes
total_iter_count = 0  # iters including multiple loops within a temp change

# Model accuracy variables
ave_acc = 5
accs = [ave_acc]
best_ave_acc = ave_acc
if not os.path.exists("masked_may_exp"):
    os.makedirs("masked_may_exp")
BEST_AVE_PATH = (
    os.getcwd()
    + "/masked_may_exp/SA"
    + str(args.ratio_prune)
    + "_"
    + str(xp_num_)
    + "_best_ave_acc.pth"
)

# Search step variables, for logging
down_steps = 0  # towards local optimum
up_steps = 0  # away from local optimum
no_steps = 0  # stay put
total_down_steps = 0
total_up_steps = 0
total_no_steps = 0

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
iter_per_temp = 2  # allows multiple decisions per given temp value
iter_multiplier = 1.005  # increase iters for every temp decrease
max_iter_per_temp = 10  # ceiling on iterations per temp value

###LogFile
###Create directory if it doesn't exist
if not os.path.exists("textlogs"):
    os.makedirs("textlogs")

###Pre-run data
log_file = open(
    "textlogs/exp_"
    + str(xp_num_)
    + "_sparsity_"
    + str(int(args.ratio_prune*100))
    + ".txt", "w"
)
log_file.write(
    os.getcwd()
    + "/masked_may_exp/SA_exp"
    + "_"
    + str(xp_num_)
    + "_"
    + str(int(args.ratio_prune*100))
    + ".pth\n"
)
log_file.write("Hyperparameters\n")
log_file.write(str("temp: " + str(temp) + "\n"))
log_file.write(str("temp_decay: " + str(temp_decay) + "\n"))
log_file.write(str("iter_per_temp: " + str(iter_per_temp) + "\n"))
log_file.write(str("iter_multiplier: " + str(iter_multiplier) + "\n"))
log_file.write(str("max_iter_per_temp: " + str(max_iter_per_temp) + "\n"))
log_file.write(str("ham_dist:" + str(ham_dist) + "\n"))
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

        # Tentatively apply the mask
        if args.k_epoch == 0:
            env.reset_to_k_0()
            
        elif args.k_epoch == 2:
            env.reset_to_k_2()
            
        elif args.k_epoch == 5:
            env.reset_to_k_5()
            
        elif args.k_epoch == 90:
            env.reset_to_k_90()
            
        else:
            env.reset_to_init_1()
        env.apply_mask(new_mask)

        # Check if keep or discard
        if args.entire_eval_flag == 1:
            new_acc = env.forward_pass_entire()
            new_acc = new_acc * 100
        else:
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
            if q < acc_prob:
                up_steps += 1
                current_mask = new_mask
                accs.append(new_acc)
                ave_acc = sum(accs) / len(accs)
            else:
                no_steps += 1
    accs = [ave_acc]

    # update post_run log_file variables
    total_down_steps += down_steps
    total_up_steps += up_steps
    total_no_steps += no_steps

    # Logging progress on stdout
    elapsed_time = time.time() - start_time
    if temp_changes in ([0,250,500,999]):
        print("-------------------------------------------------")
        print("TIME", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        print("ITER:", total_iter_count)
        print("TEMP CHANGES:", temp_changes)
        print("\tAVE ACC:", ave_acc)
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
    #writer.add_scalar("amount_pruned", amount_pruned, temp_changes)

    # Schedulers of Neighbor Size, temperature, and iters per temp
    ham_dist = max(1, int(ham_dist * ham_dist_decay))
    if ham_dist < prev_ham_dist * 0.5:
        ham_dist = int(prev_ham_dist * 0.75)
        prev_ham_dist = ham_dist
    temp = temp * temp_decay
    iter_per_temp = min(max_iter_per_temp, iter_per_temp * iter_multiplier)

    # Checkpoint
    if ave_acc > best_ave_acc:
        torch.save(
            {
                "untrained_state_dict": env.model.state_dict(),
                "heur_mask": current_mask,  
                "ave_acc": ave_acc,
                "iter": total_iter_count,
            },
            BEST_AVE_PATH,
        )
        best_iter_count = total_iter_count
        best_mask = current_mask
        best_ave_acc = ave_acc

    temp_changes += 1


print("\n---------------- End of SA Search ---------------\n")




# Tentatively apply the mask on the chosen k and store the accuracy
if args.k_epoch == 0:
    env.reset_to_k_0()
    
elif args.k_epoch == 2:
    env.reset_to_k_2()
    
elif args.k_epoch == 5:
    env.reset_to_k_5()
    
elif args.k_epoch == 90:
    env.reset_to_k_90()
    
else:
    env.reset_to_init_1()
env.apply_mask(best_mask)
k_epoch_accuracy = env._evaluate_model()

if args.entire_eval_flag == 1:
    k_epoch_forpass = env.forward_pass_entire()
    k_epoch_forpass = k_epoch_forpass * 100
else:
    k_epoch_forpass = env.forward_pass(args.num_batches) 


# Prune with the best mask
# Apply the best mask
env.reset_to_init_1()
env.apply_mask(best_mask)

###Check the amount per layer
layer_mask = [] #list
num_per_layer = []


idx = 0 
for i, item in enumerate(env.layer_filters):
    print("i item", i, item)
    layer_mask.append(best_mask[idx:idx + item].clone())
    num_per_layer.append(int(best_mask[idx:idx+item].sum()))
    idx = idx + item
    
print("Filters per layer:", num_per_layer)
print("Total", sum(num_per_layer))



###Save into .pth
PATH = (
    os.getcwd()
    + "/masked_may_exp/SA_exp"
    + "_"
    + str(xp_num_)
    + "_"
    + str(int(args.ratio_prune*100))
    + ".pth"
)
model_dicts = {
    "state_dict": env.model.state_dict(),
    "optim": env.optimizer.state_dict(),
    "filters_per_layer": num_per_layer,
    "iter_found": best_iter_count,
    "mask_applied": best_mask,
}
torch.save(model_dicts, PATH)

###End of run details
final_acc = env._evaluate_model()
# Check if keep or discard
if args.entire_eval_flag == 1:
    final_forpass = env.forward_pass_entire()
    final_forpass = final_forpass * 100
else:
    final_forpass = env.forward_pass(args.num_batches)
elapsed_time = time.time() - start_time
formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
writer.close()

###Post-run data
log_file = open(
    "textlogs/exp_"
    + str(xp_num_)
    + "_sparsity_"
    + str(int(args.ratio_prune*100))
    + ".txt", "a"
)
total_down_steps = total_down_steps / total_iter_count
total_up_steps = total_up_steps / total_iter_count
total_no_steps = total_no_steps / total_iter_count
log_file.write(str("entire_eval_flag: " + str(args.entire_eval_flag) + "\n"))
log_file.write(str("down_steps: " + str(total_down_steps) + "\n"))
log_file.write(str("up_steps: " + str(total_up_steps) + "\n"))
log_file.write(str("no_steps: " + str(total_no_steps) + "\n"))
log_file.write(str("num_iterations: " + str(total_iter_count) + "\n"))
log_file.write(str("temps_tried: " + str(temp_changes) + "\n"))
log_file.write(str("num_batches: " + str(args.num_batches) + "\n"))
log_file.write(str("final_structure: " + str(num_per_layer) + "\n"))


log_file.write(
    str("last_ave_acc: " + str(ave_acc) + "\n")
)
log_file.write(
    str("best_ave_acc: " + str(best_ave_acc) + "\n")
)

log_file.write(str("evaluated_accuracy: " + str(final_acc) + "\n"))
log_file.write(str("forwardpass_accuracy: " + str(final_forpass) + "\n"))
log_file.write(
    str("k_epoch_acc: " + str(k_epoch_accuracy) + "\n")
)
log_file.write(
    str("k_epoch_forpass: " + str(k_epoch_forpass) + "\n")
)
log_file.write(str("time: " + str(formatted_time) + "\n"))
log_file.write(str("var_seed: " + str(args.var_seed) + "\n"))
log_file.write(str("var_num: " + str(torch.rand((1))) + "\n"))

log_file.close()


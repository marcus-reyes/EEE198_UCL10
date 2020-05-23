#main_retrain_masked

import torch
import time
import os
import copy
import math
from environment import PruningEnv
import os
import logging
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import argparse


###Argument parsing
parser = argparse.ArgumentParser(description='Arguments for masker')

parser.add_argument('--foldername', type=str, default = 'trash',
                    help='folder to store masked networks in')
parser.add_argument(
    "--ratio_prune", type = float, default = 0.7, help="amount to prune"
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
args = parser.parse_args()


# Basemark is 5381

env = PruningEnv()

#### Obtain layers of the neural network
total_filters_count = 0
size_of_layer = []
for name, param in env.model.named_parameters():
    if 'conv' in name and 'weight' in name:
        total_filters_count += param.shape[0]
        # print(name)
        # print(param.shape[0])
        size_of_layer.append(param.shape[0])

#Obtain the initial mag_signs        
env.reset_to_init_1()
initial_mag_values = env.get_pooled_mag(abs_val = False)
initial_mag_values[initial_mag_values > 0] = 1
initial_mag_values[initial_mag_values < 0] = -1


env.reset_to_k_90()
#Obtain the final mag_values
final_mag_values = env.get_pooled_mag(abs_val = False)


#Obtain the largest final values that retained their signs
mag_sign_values = initial_mag_values*final_mag_values

mag_sign_rank = torch.topk(mag_sign_values,\
                            int(total_filters_count*args.ratio_prune),\
                            largest = False)
mag_sign_mask = torch.ones(total_filters_count)
mag_sign_mask[mag_sign_rank[1]] = 0

#Reset to initialization before applying
env.reset_to_k_90()
env.apply_mask(mag_sign_mask)
print(env._evaluate_model(),"First")


###Save into .pth
PATH = (
    os.getcwd()
    + "/masked_may_exp/SA_exp"
    + "_"
    + str(args.xp_num_)
    + "_"
    + str(int(args.ratio_prune*100))
    + "_mag_sign_rewind.pth"
)

###Log the pre training evaluation accuracy
log_file = open(
    "textlogs/exp_"
    + str(xp_num_)
    + "_sparsity_"
    + str(int(args.ratio_prune*100))
    + ".txt", "a"
)
final_acc = env._evaluate_model()
log_file.write(
    str(str(args.method) + "_evaluated_accuracy: " + str(final_acc) + "\n")
)
log_file.close()




model_dicts = {'state_dict': env.model.state_dict(),
        'optim': env.optimizer.state_dict(),
        'kept_indices' : torch.where(mag_sign_mask == 1)}
torch.save(model_dicts, PATH)



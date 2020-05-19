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
args = parser.parse_args()


env = PruningEnv()
env.reset_to_init_1()

#### Obtain layers of the neural network
total_filters_count = 0
size_of_layer = []
for name, param in env.model.named_parameters():
    if 'conv' in name and 'weight' in name:
        total_filters_count += param.shape[0]
        # print(name)
        # print(param.shape[0])
        size_of_layer.append(param.shape[0])
        



#Obtain the mask
random_values = torch.rand((960))
random_rank = torch.topk(random_values,\
                            int(total_filters_count*args.ratio_prune),\
                            largest = False)
random_mask = torch.ones(total_filters_count)
random_mask[random_rank[1]] = 0

env.reset_to_init_1()
env.apply_mask(random_mask)



###Save into .pth
PATH = (
    os.getcwd()
    + "/masked_may_exp/SA_exp"
    + "_"
    + str(args.xp_num_)
    + "_"
    + str(int(args.ratio_prune*100))
    + "_rand.pth"
)model_dicts = {'state_dict': env.model.state_dict(),
        'optim': env.optimizer.state_dict(),
        'kept_indices' : torch.where(layer_action == 1)}
torch.save(model_dicts, PATH)


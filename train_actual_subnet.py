import torch

import time
import os
import copy
from environment import PruningEnv
import os
import logging
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utilities import PrunedSubnet
import torch

import argparse
###Argument parsing
parser = argparse.ArgumentParser(description='Arguments for masker')
parser.add_argument('--criterion', type=str, default='mag',
                    help='criterion to use')
parser.add_argument('--foldername', type=str, default = 'pruned_may_exp',
                    help='folder to store masked networks in')
parser.add_argument('--ratio_prune', type=float, default = 0.5,
                    help='amount to prune')
parser.add_argument('--inv_flag', action = 'store_true', default = False,
                    help='invert criterion if True')

args = parser.parse_args()
#1 is 35,68,128,261
#2 is 37,62,140,253
#3 is 32,63,120,257

env = PruningEnv()

criterion = args.criterion
inv_flag = args.inv_flag
folder = '/'+ args.foldername + '/'


PATH = os.getcwd() + folder + 'SA0.9_6_pruned.pth'
model_dicts = torch.load(PATH)

filters_per_layer = model_dicts['filters_per_layer']
#filters_per_layer = [64,128,256,512]
pruned_subnet = PrunedSubnet(filter_counts = filters_per_layer)
pruned_subnet.build()
for name, param in pruned_subnet.model.named_modules():
    print("N,P", name, param)

pruned_subnet.model.load_state_dict(model_dicts['state_dict'])

val_acc = pruned_subnet.evaluate(env.test_dl)
print(val_acc)

    
writer = SummaryWriter('runs_may_12_70_exp_20_trained')
start = time.time()
for n_iter in range(0):
    if n_iter in ([25,55]):
        for param_group in pruned_subnet.optimizer.param_groups:
            param_group['lr'] *= 0.1
    print("EPOCH",n_iter)
    pruned_subnet.train_model(env.train_dl, num_epochs = 1)
    val_acc = pruned_subnet.evaluate(env.test_dl)
    print(val_acc)
    writer.add_scalar('Test/train', val_acc, n_iter)

end = time.time()
writer.close()
val_acc = pruned_subnet.evaluate(env.test_dl)
print(val_acc)
total_time = end - start
print(total_time)


if not os.path.exists('trained_may_exp'):
    os.makedirs('trained_may_exp')

PATH = os.getcwd() + '/trained_may_exp/SA0.9_6_pruned_trained_90ep.pth'
model_dicts = {'state_dict': pruned_subnet.model.state_dict(),
        'optim': pruned_subnet.optimizer.state_dict(),
        'filters_per_layer': filters_per_layer}
torch.save(model_dicts, PATH)


#unpruned 3728.4113001823425 named as pruned but no number
#pruned 75 2271.4468524456024 (error? CPU overclocked?)
#pruned 50 3788.0326392650604
#pruned 75 3844.9061708450317

# Import libraries
import torch
from networks_MLP import *
from utilities_MLP import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def orthogonality(layer): # single layer for now
    with torch.no_grad():
        square = (torch.t(layer.weight*layer.mask) @ (layer.weight*layer.mask))
        identity = torch.eye(layer.weight.size(1)) # 784x784
        loss = torch.mean((square.to(device) - identity.to(device))**2)

    return loss 

# Load masks and init weight
file_names = {'heur' : 'best_mask_mlp.pth',
              'snip' : 'trained_snip_mlp.tar',
              'dcn' : 'trained_dcn_mlp.tar',
              'lth' : 'trained_lth_mlp.tar',
              'rand' : 'trained_rand_mlp.tar'}

for name, file_name in file_names.items():
    chkpt = torch.load('./sparsity_90/trial_5/'+file_name, map_location=device)
    for key in chkpt.keys():
        if 'mask' in key:
            model = DeconsNet()
            model.load_state_dict(list(chkpt.values())[0])
            on = sum([x.sum() for x in chkpt[key]])
            elems = sum([x.numel() for x in chkpt[key]])
            print(key)
            print('ratio:',on/elems)

            if 'heur' in key:
                apply_mask_from_vector(model,chkpt[key],DEVICE)
            else:
                apply_mask_from_list(model,chkpt[key])

            os = []
            for layer in model.children():
                os.append(orthogonality(layer).item())
            print('\tOrthogonality:',os,':',sum(os))

print('=========')
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

import copy
chkpt = torch.load('./sparsity_80/trial_5/trained_dcn_mlp.tar',
                    map_location=device)
model = DeconsNet()
trained_model = copy.deepcopy(model)
model.load_state_dict(chkpt['init_state_dict'])
trained_model.load_state_dict(chkpt['trained_state_dict'])
model = apply_mask(model,'mag_sign',0.8, None, trained_model)
on = 0
elems = 0
for child in model.children():
    on += child.mask.sum()
    elems += child.mask.numel()
print(on)
print(elems)
print(on/elems)
mask = chkpt['mag_sign_mask']
ons = [x.sum() for x in mask]
on = sum(ons)
elems = sum([x.numel() for x in mask])
print(key)
print(ons)
print(on)
print(elems)
print(on/elems)


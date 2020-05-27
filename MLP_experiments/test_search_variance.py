import os
import sys
import torch
import pandas as pd
from torch.nn import CosineSimilarity

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

cosine_similarity = CosineSimilarity(dim=-1)

def hamming_dist(mask_list):
    dists = torch.zeros((len(mask_list),)*2)
    for x, mask_x in enumerate(mask_list):
        for y, mask_y in enumerate(mask_list):
            dists[x,y] = torch.squeeze(mask_x).dot(torch.squeeze(mask_y)) 

    return dists

# in main_SA
    # load same init weights and init SA mask
    # FINE_TUNE for first, not for the latter 2
    # run SA optim, save init and final SA mask (best_mask_mlp, append seed)
    # save plot with seed appended

# directory of mask pth files
# sparsity
    # trial
        # SA
            # init
                # seed 1    
                # seed 2
                # seed 3
        # snip
        # dcn
        # lth
        # rand

EXT_PATH = os.getcwd() + sys.argv[1] #sparsity_80/trial_6
files = {'SA1': EXT_PATH + '/seed_42_best_SA_mask.pth',
         'SA2': EXT_PATH + '/seed_43_best_SA_mask.pth',
         'SA3': EXT_PATH + '/seed_44_best_SA_mask.pth',
         'SA4': EXT_PATH + '/seed_45_best_SA_mask.pth',
         'snip': EXT_PATH + '/trained_snip_mlp.tar',
         'mag_sign': EXT_PATH + '/trained_dcn_mlp.tar',
         'post_mag': EXT_PATH + '/trained_lth_mlp.tar',
         'rand': EXT_PATH + '/trained_rand_mlp.tar'
         }

# 2d similarity matrices
masks = []
for i, (criterion, filename) in enumerate(files.items()):
    chkpt = torch.load(filename,map_location=DEVICE)
    if 'seed' in filename:
        if i == 0:
            masks.append(chkpt['init_SA_mask'])
        masks.append(chkpt['best_SA_mask'])
    else:
        mask  = torch.cat(
                    [torch.flatten(m) for m in chkpt[criterion + '_mask']],-1)
        masks.append(torch.unsqueeze(mask,0))

hamming = hamming_dist(masks)

masks = torch.stack(masks,dim=0)
masks_t = torch.transpose(masks,0,1)
similarity = cosine_similarity(masks, masks_t)*100

labels = ['init_SA'] + list(files.keys())
with pd.option_context(
        'display.max_rows', None, 
        'display.max_columns', None, 
        'display.expand_frame_repr', False
     ):
    df = pd.DataFrame(similarity.cpu().numpy(), columns=labels, index=labels)
    print(df)
    df.to_csv(EXT_PATH + '/similarity_matrix.csv')


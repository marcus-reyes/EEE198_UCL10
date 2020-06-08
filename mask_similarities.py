import os
import sys
import torch
import pandas as pd
from torch.nn import CosineSimilarity

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# similarity measures
cosine_similarity = CosineSimilarity(dim=-1)
def hamming_dist(mask_list):
    # WRONG
    dists = torch.zeros((len(mask_list),)*2)
    for x, mask_x in enumerate(mask_list):
        for y, mask_y in enumerate(mask_list):
            dists[x,y] = torch.squeeze(mask_x).dot(torch.squeeze(mask_y)) 

    return dists

# paths
EXT_PATH = os.getcwd() + sys.argv[1] #sparsity_80/trial_6
# if CNN
files = {'SA_k0_1': EXT_PATH + '/SA_exp_200_50.pth',
         'SA_k0_2': EXT_PATH + '/SA_exp_201_50.pth',
         'SA_k0_3': EXT_PATH + '/SA_exp_202_50.pth',
         'SA_k0_4': EXT_PATH + '/SA_exp_203_50.pth',
         'SA_k0_5': EXT_PATH + '/SA_exp_204_50.pth',
         'SA_k90_1': EXT_PATH + '/SA_exp_205_50.pth',
         'SA_k90_2': EXT_PATH + '/SA_exp_206_50.pth',
         'SA_k90_3': EXT_PATH + '/SA_exp_207_50.pth',
         'SA_k90_4': EXT_PATH + '/SA_exp_208_50.pth',
         'SA_k90_5': EXT_PATH + '/SA_exp_209_50.pth',
         'SA_k0_magsign': EXT_PATH + '/SA_exp_210_50_mag_sign_rewind.pth',
         'SA_k0_mag': EXT_PATH + '/SA_exp_210_50_mag_rewind.pth',
         'SA_k0_rand': EXT_PATH + '/SA_exp_210_50_rand.pth',
        }

# if MLP
#files = {'SA_k0_1': EXT_PATH + '/seed_42_best_SA_mask.pth',
#         'SA_k0_2': EXT_PATH + '/seed_43_best_SA_mask.pth',
#         'SA_k0_3': EXT_PATH + '/seed_44_best_SA_mask.pth',
#         'SA_k0_4': EXT_PATH + '/seed_45_best_SA_mask.pth',
#         'SA_k75_1': EXT_PATH + '/seed_46_best_SA_mask.pth',
#         'SA_k75_2': EXT_PATH + '/seed_47_best_SA_mask.pth',
#         'SA_k75_3': EXT_PATH + '/seed_48_best_SA_mask.pth',
#         'SA_k75_4': EXT_PATH + '/seed_49_best_SA_mask.pth',
#         'snip': EXT_PATH + '/trained_snip_mlp.tar',
#         'mag_sign': EXT_PATH + '/trained_dcn_mlp.tar',
#         'post_mag': EXT_PATH + '/trained_lth_mlp.tar',
#         'rand': EXT_PATH + '/trained_rand_mlp.tar'
#         }

# 2d similarity matrices
masks = []
for i, (criterion, filename) in enumerate(files.items()):
    chkpt = torch.load(filename,map_location=DEVICE)
    # if cnn experimets
    masks.append(chkpt['mask_applied'].unsqueeze(0))
    # if mlp experiments
    #if 'SA' in filename:
    #    masks.append(chkpt['best_SA_mask']) #shape: [1,length]
    #else:
    #    mask  = torch.cat(
    #                [torch.flatten(m) for m in chkpt[criterion + '_mask']],-1)
    #    masks.append(torch.unsqueeze(mask,0))

# add random masks
num_ones = int(masks[-1].sum().item())
num_rands = 5
for i in range(num_rands):
    rands = torch.zeros_like(masks[-1])
    print(rands.shape)
    rands[0,torch.topk(torch.rand_like(masks[-1]), num_ones).indices] = 1
    masks.append(rands)
    print(masks[-1].sum())
    print(masks[-1].shape)

#hamming = hamming_dist(masks)

masks = torch.stack(masks,dim=0)
print('After stack:', masks.shape)
masks_t = torch.transpose(masks,0,1)
similarity = cosine_similarity(masks, masks_t)*100

labels = list(files.keys()) + ['rand']*num_rands
with pd.option_context(
        'display.max_rows', None, 
        'display.max_columns', None, 
        'display.expand_frame_repr', False
     ):
    df = pd.DataFrame(similarity.cpu().numpy(), columns=labels, index=labels)
    print(df)
    df.to_csv(EXT_PATH + '/similarity_matrix.csv')


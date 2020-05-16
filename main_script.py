import subprocess as sp 


def mask_prune_train():
    

    mask_command = ['python','main_SA.py']
            # '--criterion',str(item),
            # '--foldername', 'comp_exp_loc_80_cifar',
            # '--ratio_prune', '0.8']

    sp.run(mask_command)
    
    

    prune_command = ['python', 'actual_prune.py']
                # '--foldername', 'comp_exp_loc_80_cifar']
   
    sp.run(prune_command)
    

    train_command = ['python', 'train_actual_subnet.py']
                # '--criterion', str(item),
                # '--foldername', 'comp_exp_loc_80_cifar']
    sp.run(train_command)

if __name__ == '__main__':
    mask_prune_train()
import subprocess as sp 


def mask_prune_train(xp_num_, ratio_prune):

    
    
    mask_command = ['python','main_SA.py',
            '--xp_num_' , str(xp_num_),
            '--ratio_prune', str(ratio_prune)]

    sp.run(mask_command)
    
    

    prune_command = ['python', 'actual_prune.py',
            '--xp_num_' , str(xp_num_),
            '--ratio_prune', str(ratio_prune)]
   
    sp.run(prune_command)
    

    train_command = ['python', 'train_actual_subnet.py',
            '--xp_num_' , str(xp_num_),
            '--ratio_prune', str(ratio_prune)]
    sp.run(train_command)

if __name__ == '__main__':
    xp_num_ = 6
    ratio_prune = 0.5
    mask_prune_train(xp_num_, ratio_prune)

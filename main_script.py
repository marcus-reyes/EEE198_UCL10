import subprocess as sp 


def mask_prune_train(xp_num_, ratio_prune, method_list):


    
    for method in method_list:
    
        if method == "SA":
            mask_py = "main_SA.py"
        elif method == "rand":
            mask_py = "main_pruner_random.py"
        elif method == "mag_rewind":
            mask_py = "main_pruner_mag_rewind.py"
        elif method == "mag_sign_rewind":
            mask_py = "main_pruner_mag_sign_rewind.py"
            
        mask_command = ['python', str(mask_py),
                '--xp_num_' , str(xp_num_),
                '--method' , str(method),
                '--ratio_prune', str(ratio_prune)]
        print(mask_command, "MASKCOMMAND")
        sp.run(mask_command)
        
        

        prune_command = ['python', 'actual_prune.py',
                '--xp_num_' , str(xp_num_),
                '--method' , str(method),
                '--ratio_prune', str(ratio_prune)]
       
        sp.run(prune_command)
        

        train_command = ['python', 'train_actual_subnet.py',
                '--xp_num_' , str(xp_num_),
                '--method' , str(method),
                '--ratio_prune', str(ratio_prune)]
        sp.run(train_command)

if __name__ == '__main__':
    xp_num_ = 6
    ratio_prune = 0.5
    method_list = ["mag_sign_rewind"]
    mask_prune_train(xp_num_, ratio_prune, method_list)

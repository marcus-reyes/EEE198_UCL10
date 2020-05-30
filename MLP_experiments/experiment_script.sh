# seed also ensures different ouput names for same trial
echo "Running Experiment on 90% Sparsity"
echo
python main_SA_MLP.py --ratio_prune 90 --trial 12 --fine_tune
echo "Running Experiment on 90% Sparsity seed 43"
echo
python main_SA_MLP.py --ratio_prune 90 --trial 12 --seed 43 --same_init
echo "Running Experiment on 90% Sparsity seed 44"
echo
python main_SA_MLP.py --ratio_prune 90 --trial 12 --seed 44 --same_init
#echo "Running Experiment on 90% Sparsity seed 45"
#echo
#python main_SA_MLP.py --ratio_prune 90 --trial 12 --seed 45 --same_init
echo "Running Experiment on 90% Sparsity seed 46"
echo
python main_SA_MLP.py --ratio_prune 90 --trial 12 --seed 46 --k 75 --same_init
echo "Running Experiment on 90% Sparsity seed 47"
echo
python main_SA_MLP.py --ratio_prune 90 --trial 12 --seed 47 --k 75 --same_init
echo "Running Experiment on 90% Sparsity seed 48"
echo
python main_SA_MLP.py --ratio_prune 90 --trial 12 --seed 48 --k 75 --same_init
#echo "Running Experiment on 90% Sparsity seed 49"
#echo
#python main_SA_MLP.py --ratio_prune 90 --trial 12 --seed 49 --k 75 --same_init

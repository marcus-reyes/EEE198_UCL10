# seed also ensures different ouput names for same trial
echo "Running Experiment on 70% Sparsity"
python main_SA_MLP.py --ratio_prune 70 --trial 1 --fine_tune
echo
echo "Running Experiment on 80% Sparsity"
python main_SA_MLP.py --ratio_prune 80 --trial 1 --fine_tune
echo
echo "Running Experiment on 90% Sparsity"
python main_SA_MLP.py --ratio_prune 90 --trial 1 --fine_tune

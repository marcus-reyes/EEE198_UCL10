echo "Running Experiment on 70% Sparsity"
python main_SA_MLP.py --ratio_prune 70
echo
echo "Running Experiment on 80% Sparsity"
echo
python main_SA_MLP.py --ratio_prune 80
echo
echo "Running Experiment on 90% Sparsity"
echo
python main_SA_MLP.py --ratio_prune 90
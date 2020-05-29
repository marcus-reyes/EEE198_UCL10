echo "Running Experiment on 70% Sparsity"
echo
python main_SA_MLP.py --ratio_prune 70 --trial 11 --fine_tune
echo
echo "Running Experiment on 80% Sparsity"
echo
python main_SA_MLP.py --ratio_prune 80 --trial 11 --fine_tune
echo
echo "Running Experiment on 90% Sparsity"
echo
python main_SA_MLP.py --ratio_prune 90 --trial 11 --fine_tune

echo "Running Experiment on 80% Sparsity"
echo
python main_SA_MLP.py --ratio_prune 80 --trial 6 --fine_tune
echo
echo "Running Experiment on 80% Sparsity with seed 43"
echo
python main_SA_MLP.py --ratio_prune 80 --seed 43 --trial 6
echo
echo "Running Experiment on 80% Sparsity with seed 44"
echo
python main_SA_MLP.py --ratio_prune 80 --seed 44 --trial 6
echo
echo "Running Experiment on 80% Sparsity with seed 45"
echo
python main_SA_MLP.py --ratio_prune 80 --seed 45 --trial 6

echo "Running Experiment on 70% Sparsity w Signed Const"
echo
python main_SA_MLP.py --ratio_prune 70 --reinit
echo
echo "Running Experiment on 70% Sparsity w k=1"
echo
python main_SA_MLP.py --ratio_prune 70 --k 1
echo
echo "Running Experiment on 80% Sparsity w Signed Const"
echo
python main_SA_MLP.py --ratio_prune 80 --reinit
echo
echo "Running Experiment on 80% Sparsity w k=1"
echo
python main_SA_MLP.py --ratio_prune 80 --k 1
echo
#echo "Running Experiment on 90% Sparsity"
#echo
#python main_SA_MLP.py --ratio_prune 90

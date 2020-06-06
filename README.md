# EEE 198: Simulated Annealing for Model Compression

## Authors

This codebase was developed by Jeff Sanchez and Marcus Reyes for during 
their senior-year project; UCL, UP-Diliman 2020.

## Background

This library is the implementation of our thesis project, titled
"thesis-title-here." Taking inspiration from recent works on lottery tickets
[1,2], Model Compression [3], and Early Pruning [4], we wanted to automate 
the process of finding high performing and high sparsity subnetworks (winning 
tickets), without relying on already trained networks. This form of compression
allos for memory and energy savings in training neural networks. After our 
initial attempt of using an [RL agent](github.com/prokorpio/everything_190) to 
search for these subnetworks, we resolved to use a simpler approach based on 
the heuristic search algorithm, Simulated Annealing (SA). 

Briefly explain SA setup.

To evaluate the performance of SA, we run the following experiments:

1. Mask Search on CNN
    1. Randomly initialize...

2. Actual Pruning CNN
    1. ...

3. Training Pruned CNN
    1. ...   

Additional analysis were performed on the final SA masks through the following 
experiments:

1. Orthogonality scores and training loss plots on MLP
2. Mask similarities matrices on MLP and CNN

## Summary of Results

### MLP Experiments
1. Sample plot of SA search
2. Criterion Comparison
3. Trainability
4. Mask similarities

### CNN Experiments
1. Sample plot of SA search
2. Criterion Comparison
3. Mask Similarities

## Getting Started

1. Run `setup.py` to install library dependencies. (I'll [add](https://python-packaging.readthedocs.io/en/latest/dependencies.html)) 

2. Modify paths... (if necessary)

3. Run ...

## Code Files

### Main components
1. main_SA.py
2. environment.py
3. utilities.py
4. trainer.py and train_actual_subnet.py
5. MLP_experiments/...

### Analysis Scripts
1. mask_similarities.py

### Paths and Log Folders

## Datasets
1. [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) for CNN experiments.
2. [MNIST](http://yann.lecun.com/exdb/mnist/) for MLP experiments.

## References
[1] Lottery Ticket Hypothesis
[2] Deconstructing Lottery Tickets
[3] AutoML for Model Compression
[4] SNIP


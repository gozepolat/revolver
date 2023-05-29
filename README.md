# Revolver

A PyTorch based library to simultaneously train a population of diverse models and analyze reusable modules.

Revolver relies on global parameter sharing and estimate scores for all modules. 

Revolver allows model components to be:

* described hierarchically using a _blueprinting approach_
* shared within a model (e.g. cross layer parameter sharing), between models, and across generations in a population by relying on _scopes_,
* and modified by search algorithms. 

## Getting Started
Revolver can train a population in two search modes: _random_ or _evolve_.

### Random Search
Random search has diverse building blocks from different hierarchies to construct models of varying complexity. On a GeForce RTX 3090, the following population training scenario on MNIST reaches ~99.5 test accuracy with several models in the population.
For smaller GPUs, please use a smaller batch size and a smaller learning rate.

``
python main.py --gpu_id 0 --batch_size 512 --mode population_train --search_mode random_warmup --sample_size 8 --epochs 100 --lr 0.25 --lr_drop_at_stagnate 5 --genotype_cost 1 --skeleton '[32,64,128]' --warmup_x 5 --population_size 100 --dataset MNIST --finetune_after_evolving "n" 2>&1 | tee -a ../mnist_random_warmup.log
``

### Genetic Algorithm
Evolve mode uses a genetic algorithm to search and train viable models. It is inspired from the tournament selection strategy, and the evolution of the nervous system.

``
python main.py --gpu_id 0 --batch_size 512 --mode population_train --search_mode random_warmup --sample_size 8 --epochs 100 --lr 0.25 --lr_drop_at_stagnate 5 --genotype_cost 1 --skeleton '[32,64,128]' --warmup_x 5 --population_size 100 --dataset MNIST --finetune_after_evolving "n" 2>&1 | tee -a ../mnist_random_warmup.log
``

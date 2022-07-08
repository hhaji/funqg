# FunQG: Molecular Representation Learning Via Quotient Graphs

**FunQG** is a novel graph coarsening framework specific to molecular data, utilizing functional groups based on a graph-theoretic concept called quotient graph. FunQG can accurately complete various molecular property prediction tasks with a significant parameters reduction. By experiments, this method significantly outperforms previous baselines on various datasets, besides its low computational complexity.

<p align="center">
   <img  src=https://github.com/zahta/funqg/blob/main/data/funqg.png?raw=true width="1000"/>
</p>

 <br>


## Requirements 
The resulting graphs of the FunQG are much smaller than the molecular graphs. Therefore, a GNN model architecture requires much less depth in working with resulting graphs compared to working with molecular graphs. Thus, using FunQG reduces the computational complexity compared to working with molecular graphs. We utilize one *Intel (R) Xeon (R) E5-2699 v4 @ 2.20GHz* CPU for training, testing, and hyperparameter tuning of a GNN model on each dataset in a relatively short time. Therefore, training the models is very fast and is possible on a standard laptop with only one CPU.

```
PyTorch >= 1.9.0
DGL >= 0.6.0
Ray Tune >= 1.9.0 (for hyperparameters optimization)
```

## How to use

```sh
git clone git clone https://github.com/hhaji/funqm.git
cd ./funqg
```

### Training And Evaluation Example
```sh
python train_eval_run.py --name_data <dataset> --current_dir <path> --config <config>

usage: train_eval_run.py [-h] 

  --name_data     tox21, toxcast, clintox, sider, bbbp, bace, freesolv, esol, lipo
  --current_dir     Current directory containing codes and data folder
  --global_feature     Whether to use global features
  --max_norm_status     Whether to use max-norm regularization
  --scaler_regression     Whether to use Standard scaler for regression tasks
  --division     scaffold, random
  --batch_size     Batch size
  --n_splits     Number of splits for CV
  --num_seeds     Number of random seeds to generate graphs
  --num_epochs     Number of epochs
  --device     cpu, cuda
  --patience     Number of patience of early stopping
  --config     A configuration of hyperparameters
```

### Hyperparameters Optimization Example
```sh
python hyper_tuning_run.py --name_data <dataset> --current_dir <path>

usage: hyper_tuning_run.py [-h] 

  --name_data     tox21, toxcast, clintox, sider, bbbp, bace, freesolv, esol, lipo
  --current_dir      Current directory containing codes and data folder
  --global_feature      Whether to use global features
  --max_norm_status     Whether to use max-norm regularization
  --scaler_regression     Whether to use Standard scaler for regression tasks
  --division     scaffold, random
  --batch_size     Batch size
  --name_scheduler     asha, bohb, median
  --name_search_alg     optuna, bohb, hyperopt
  --num_samples     Number of times to sample from the hyperparameter space
  --training_iteration     Number of iteration of training for hyperparameter tuning
  --max_concurrent     Maximum number of trials to run concurrently
  --num_cpus     Number of CPUs (CPU_core*Thread_per_core) for hyperparameter tuning
  --num_gpus     Number of GPUs for hyperparameter tuning
  --n_splits     Number of splits for CV
  --num_seeds     Number of random seeds to generate graphs
```

## Authors
- **Zahra Taheri** - [zahta](https://github.com/zahta)
- **Hossein Hajiabolhassan** - [hhaji](https://github.com/hhaji)

## Co-Authors
- **Ali Hojatnia** - [alihojatnia](https://github.com/alihojatnia)
- **Yavar Taheri Yeganeh** - [YavarYeganeh](https://github.com/YavarYeganeh)

## Citation



# FunQG: Molecular Representation Learning Via Quotient Graphs

**FunQG** is a novel graph coarsening framework specific to molecular data, utilizing functional groups based on a graph-theoretic concept called quotient graph. FunQG can accurately complete various molecular property prediction tasks with a significant parameters reduction. By experiments, this method significantly outperforms previous baselines on various datasets, besides its low computational complexity.

<p align="center">
   <img  src=https://github.com/zahta/funqg/blob/main/data/funqg.png?raw=true width="1000"/>
</p>

 <br>


## Requirements 
The resulting graphs of the FunQG are much smaller than the molecular graphs. Therefore, a GNN model architecture requires much less depth in working with resulting graphs compared to working with molecular graphs. Thus, using FunQG reduces the computational complexity compared to working with molecular graphs. We utilize 1 *Intel (R) Xeon (R) E5-2699 v4 @ 2.20GHz* CPU for training, testing, and hyperparameter tuning of the FunQG-DMPNN on each dataset in a relatively short time. So, training the models is very fast and is possible on a standard laptop with only one CPU.

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

### Hyperparameters Optimization Example
```sh
python hyper_tuning_run.py --name_data <dataset> --current_dir <path>

usage: hyper_tuning_run.py [-h] 

  --name_data NAME_DATA    tox21, toxcast, clintox, sider, bbbp, bace, freesolv, esol, lipo
  --current_dir CURRENT_DIR   Current directory containing codes and data folder
  --global_feature GLOBAL_FEATURE   Whether to use global features
  --max_norm_status MAX_NORM_STATUS    Whether to use max-norm regularization
  --scaler_regression SCALER_REGRESSION   Whether to use Standard scaler for regression tasks
  --division DIVISION   scaffold, random
  --batch_size BATCH_SIZE   Batch size
  --name_scheduler NAME_SCHEDULER   asha, bohb, median
  --name_search_alg NAME_SEARCH_ALG    optuna, bohb, hyperopt
  --num_samples NUM_SAMPLES   Number of times to sample from the hyperparameter space
  --training_iteration TRAINING_ITERATION    Number of iteration of training for hyperparameter tuning
  --max_concurrent MAX_CONCURRENT   Maximum number of trials to run concurrently
  --num_cpus NUM_CPUS   Number of CPUs (CPU_core*Thread_per_core) for hyperparameter tuning
  --num_gpus NUM_GPUS   Number of GPUs for hyperparameter tuning
  --n_splits N_SPLITS   Number of splits for CV
  --num_seeds NUM_SEEDS    Number of random seeds to generate graphs
```

### Training And Evaluation Example
```sh
python train_eval_run.py --name_data <dataset> --current_dir <path> --config <config>

usage: train_eval_run.py [-h] 

  --name_data NAME_DATA    tox21, toxcast, clintox, sider, bbbp, bace, freesolv, esol, lipo
  --current_dir CURRENT_DIR   Current directory containing codes and data folder
  --global_feature GLOBAL_FEATURE   Whether to use global features
  --max_norm_status MAX_NORM_STATUS    Whether to use max-norm regularization
  --scaler_regression SCALER_REGRESSION   Whether to use Standard scaler for regression tasks
  --division DIVISION   scaffold, random
  --batch_size BATCH_SIZE   Batch size
  --n_splits N_SPLITS   Number of splits for CV
  --num_seeds NUM_SEEDS    Number of random seeds to generate graphs
  --num_epochs NUM_EPOCHS   Number of epochs
  --device DEVICE   cpu, cuda
  --patience PATIENCE   Number of patience of early stopping
  --config CONFIG   A configuration of hyperparameters
```

## Authors

- **Hossein Hajiabolhassan** - [hhaji](https://github.com/hhaji)
- **Zahra Taheri** - [zahta](https://github.com/zahta)
- **Ali Hojatnia** - [alihojatnia](https://github.com/alihojatnia)
- **Yavar Taheri Yeganeh** - [YavarYeganeh](https://github.com/YavarYeganeh)

## Citation



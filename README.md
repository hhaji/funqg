# FunQG: Molecular Representation Learning Via Quotient Graphs

<div align="justify">
   
**FunQG** is a novel graph coarsening framework specific to molecular data, utilizing **Fun**ctional groups based on a graph-theoretic concept called **Q**uotient **G**raph. FunQG can accurately complete various molecular property prediction tasks with a significant parameters reduction. By experiments, this method significantly outperforms previous baselines on various datasets, besides its low computational complexity.

<p align="center">
   <img  src=https://github.com/zahta/funqg/blob/main/data/funqg.png?raw=true width="1000"/>  
</p>
<b>The overview of FunQG framework.</b> The left figure (A) illustrates the application of the FunQG framework to a molecule to find its corresponding coarsened graph, named molecular quotient graph. The right figure (B) shows the application of a GNN architecture to the graph obtained from the FunQG to predict the property of the molecule. 

## Requirements 
The resulting graphs of the FunQG are much smaller than the molecular graphs. Therefore, a GNN model architecture requires much less depth in working with resulting graphs compared to working with molecular graphs. Thus, using FunQG reduces the computational complexity compared to working with molecular graphs. We utilize one *Intel (R) Xeon (R) E5-2699 v4 @ 2.20GHz* CPU for training, testing, and hyperparameter tuning of a GNN model on each dataset in a relatively short time. Therefore, training the models is very fast and is possible on a standard laptop with only one CPU.
</div>

```
PyTorch >= 1.9.0
DGL >= 0.6.0
Ray Tune >= 1.9.0 (for hyperparameters optimization)
```

## How to Run

```sh
git clone https://github.com/hhaji/funqg.git
cd ./funqg
```

### Training And Evaluation Example
```sh
python train_eval_run.py --name_data <dataset> --current_dir <path> --config <config>

usage: train_eval_run.py [-h] 
  --name_data           tox21, toxcast, clintox, sider, bbbp, bace, freesolv, esol, lipo
  --current_dir         Current directory containing codes and data folder
  --global_feature      Whether to use global features
  --max_norm_status     Whether to use max-norm regularization
  --scaler_regression   Whether to use Standard scaler for regression tasks
  --division            scaffold, random
  --batch_size          Batch size
  --n_splits            Number of splits for CV
  --num_epochs          Number of epochs
  --device              cpu, cuda
  --patience            Number of patience of early stopping
  --config              A configuration of hyperparameters as an string, e.g.,
                        "{"GNN_Layers": 5.0, "dropout": 0.15, "lr": 0.0005}"
```

### Hyperparameters Optimization Example
```sh
python hyper_tuning_run.py --name_data <dataset> --current_dir <path>

usage: hyper_tuning_run.py [-h] 
  --name_data           tox21, toxcast, clintox, sider, bbbp, bace, freesolv, esol, lipo
  --current_dir         Current directory containing codes and data folder
  --global_feature      Whether to use global features
  --max_norm_status     Whether to use max-norm regularization
  --scaler_regression   Whether to use Standard scaler for regression tasks
  --division            scaffold, random
  --batch_size          Batch size
  --name_scheduler      None, asha, bohb, median
  --name_search_alg     None, optuna, bohb, hyperopt
  --num_samples         Number of times to sample from the hyperparameter space
  --training_iteration  Number of iteration of training for hyperparameter tuning
  --max_concurrent      Maximum number of trials to run concurrently
  --num_cpus NUM_CPUS   Number of CPUs (CPU_core*Thread_per_core) for hyperparameter tuning
  --num_gpus NUM_GPUS   Number of GPUs for hyperparameter tuning
  --n_splits N_SPLITS   Number of splits for CV
```

## Authors
- **Zahra Taheri** - [zahta](https://github.com/zahta)
- **Hossein Hajiabolhassan** - [hhaji](https://github.com/hhaji)

## Co-Authors
- **Ali Hojatnia** - [alihojatnia](https://github.com/alihojatnia)
- **Yavar Taheri Yeganeh** - [YavarYeganeh](https://github.com/YavarYeganeh)

## Citation


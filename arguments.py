
""" Global Arguments"""

import pandas as pd
import os
import sys
import numpy as np
from numpy.random import randint
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--name_data", type=str, default='lipo', help="tox21, toxcast, clintox, sider, bbbp, bace, freesolv, esol, lipo")
parser.add_argument("--current_dir", type=str, default=os.path.dirname(__file__)+"/", help="Current directory containing codes and data folder") 
parser.add_argument("--global_feature", type=bool, default=True, help="Whether to use global features")
parser.add_argument("--max_norm_status", type=bool, default=True, help="Whether to use max-norm regularization")
parser.add_argument("--scaler_regression", type=bool, default=True, help="Whether to use Standard scaler for regression tasks")
parser.add_argument("--division", type=str, default='scaffold', help='scaffold, random')
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
if sys.argv[0]!=os.path.dirname(sys.argv[0])+"/train_eval_run.py":
    ray_tune = True
    parser.add_argument("--name_scheduler", type=str, default="asha", help="asha, bohb, median")
    parser.add_argument("--name_search_alg", type=str, default="optuna", help="optuna, bohb, hyperopt")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of times to sample from the hyperparameter space")
    parser.add_argument("--training_iteration", type=int, default=64, help="Number of iteration of training for hyperparameter tuning")
    parser.add_argument("--max_concurrent", type=int, default=40, help="Maximum number of trials to run concurrently")
    parser.add_argument("--num_cpus", type=int, default=40, help="Number of CPUs (CPU_core*Thread_per_core) for hyperparameter tuning")
    parser.add_argument("--num_gpus", type=int, default=0, help="Number of GPUs for hyperparameter tuning")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of splits for CV")
    parser.add_argument("--num_seeds", type=int, default=1, help="Number of random seeds to generate graphs")
else:
    ray_tune = False
    parser.add_argument("--n_splits", type=int, default=3, help="Number of splits for CV")
    parser.add_argument("--num_seeds", type=int, default=3, help="Number of random seeds to generate graphs")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--device", type=str, default='cpu', help='cpu, cuda')
    parser.add_argument("--patience", type=int, default=20, help="Number of patience of early stopping")
    parser.add_argument('--config', default={}, type=json.loads, help="A configuration of hyperparameters")

# Print help 
# parser.print_help(sys.stderr) 

# Print help if no argument is passed
if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
    
class C:
    pass
c=C()
args, unknown = parser.parse_known_args(namespace=c)

if c.name_data in ["tox21", "toxcast", "clintox", "sider", "bbbp", "bace"]:
    task_type = "Classification"
    mode_ray = 'max'    
elif c.name_data in ["freesolv", "esol", "lipo"]:
    mode_ray = 'min'
    task_type = "Regression"
else:  # Unknown dataset error
    raise Exception('Unknown dataset, please enter a correct --name_data')

if c.name_data=="tox21":
    num_tasks = 12
elif c.name_data=="bbbp":
    num_tasks = 1   
elif c.name_data=="bace":
    num_tasks = 1   
elif c.name_data=="clintox":
    num_tasks = 2   
elif c.name_data=="toxcast":
    num_tasks = 617   
elif c.name_data=="sider":
    num_tasks = 27   
elif c.name_data=="lipo":
    num_tasks = 1   
elif c.name_data=="esol":
    num_tasks = 1   
elif c.name_data=="freesolv":
    num_tasks = 1   
else:  # Unknown dataset error
    raise Exception('Unknown dataset, please enter a correct --name_data')                           

if c.global_feature == False:
    global_size=0
else:
    global_size=200

node_feature_size = 127
edge_feature_size = 12
name_node_feature="_"+str(node_feature_size)+"_one_hot"        
name_final_zip = "Hierarchical_Quotient_type_False_Both_False_Uni_Vert_False_#quotient_2_#layers_1_127_one_hot.zip"

'''
Random seed to use when splitting data into train/val/test sets. When `n_splits > 1`,
the first fold uses the seed 0 and all subsequent folds add 1 to the seed (similar to DMPNN:
https://chemprop.readthedocs.io/en/latest/args.html?highlight=seed#chemprop.args.TrainArgs.seed)
''' 
list_seeds = list(np.arange(c.num_seeds))

"""Hyperparameters Optimization with Ray Tune"""

import pandas as pd 
from pandas import MultiIndex, Int16Dtype
import os
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import shutil
import ray
from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.utils.placement_groups import PlacementGroupFactory

import arguments
from arguments import args
from dataset import DGLDatasetClass, TuneDatasetReg
from utils_tune import TrainableCV, scheduler_fn, search_alg_fn

""" Set Path"""
current_dir = args.current_dir
results_ray = current_dir + "save_models/ray"
os.makedirs(current_dir + "save_models/result_ray/" + args.name_data+"/"+args.division+\
        "/Hierarchical_Quotient_type_False_Idx_Row_11/Both_False_Uni_Vert_False_#quotient_2_#layers_1", exist_ok=True)
result_csv = current_dir + "save_models/result_ray/" + args.name_data+"/"+args.division+ \
        "/Hierarchical_Quotient_type_False_Idx_Row_11/Both_False_Uni_Vert_False_#quotient_2_#layers_1/"+str(args.name_scheduler)+"_"+str(args.name_search_alg)+"_"+datetime.now().strftime("%Y%m%d")+".csv"
folder_data_temp = current_dir +"data_temp/"
shutil.rmtree(folder_data_temp, ignore_errors=True) 

for seed in arguments.list_seeds: 
    path_save = current_dir + "data/graph/"+args.name_data+"/"+args.division+"_"+str(seed)+"/"+arguments.name_final_zip
    shutil.unpack_archive(path_save, folder_data_temp)
   
train_set={}
val_set={}
test_set={}
scaler = {}
if arguments.task_type=="Classification":
    for name_seed in arguments.list_seeds:
        seed=name_seed
        path_data_temp = folder_data_temp + args.division+"_"+str(seed)
        print(path_data_temp)
        train_set[seed] = DGLDatasetClass(address=path_data_temp+"_train")
        val_set[seed] = DGLDatasetClass(address=path_data_temp+"_val")
        test_set[seed] = DGLDatasetClass(address=path_data_temp+"_test")
    data_cross_validation={}
    seed = arguments.list_seeds[0]
    dataset=torch.utils.data.ConcatDataset([train_set[seed], val_set[seed], test_set[seed]])
    idx_all = list(np.arange(len(dataset)))
    idx_remained = idx_all
    fraction = 0.1
    for fold_idx in range(args.n_splits):
        idx_valid_fold = random.sample(idx_remained, int(fraction*len(dataset)))
        idx_remained = list(set(idx_remained)-set(idx_valid_fold))
        idx_test_fold = random.sample(idx_remained, int(fraction*len(dataset)))
        idx_remained = list(set(idx_remained)-set(idx_test_fold))                
        idx_train_fold = list(set(idx_all)-set(idx_valid_fold)-set(idx_test_fold))
        data_cross_validation[(seed, fold_idx, 1)] = torch.utils.data.Subset(dataset, idx_train_fold)
        data_cross_validation[(seed, fold_idx, 2)] = torch.utils.data.Subset(dataset, idx_valid_fold)
        data_cross_validation[(seed, fold_idx, 3)] = torch.utils.data.Subset(dataset, idx_test_fold)
        print("train size:", len(data_cross_validation[(seed, fold_idx, 1)]),
            "valid size:", len(data_cross_validation[(seed, fold_idx, 2)]),
            "test_fold:", len(data_cross_validation[(seed, fold_idx, 3)]))
        val_size = len(data_cross_validation[(seed, fold_idx, 2)])
        test_size = len(data_cross_validation[(seed, fold_idx, 3)])
        
else:
    seed = arguments.list_seeds[0]
    path_data_temp = folder_data_temp + args.division+"_"+str(seed)

    train_set, train_labels_masks_globals = dgl.load_graphs(path_data_temp+"_train"+".bin")
    train_labels = train_labels_masks_globals["labels"].view(len(train_set),-1)
    train_masks = train_labels_masks_globals["masks"].view(len(train_set),-1)
    train_globals = train_labels_masks_globals["globals"].view(len(train_set),-1)

    val_set, val_labels_masks_globals = dgl.load_graphs(path_data_temp+"_val"+".bin")
    val_labels = val_labels_masks_globals["labels"].view(len(val_set),-1)
    val_masks = val_labels_masks_globals["masks"].view(len(val_set),-1)
    val_globals = val_labels_masks_globals["globals"].view(len(val_set),-1)

    test_set, test_labels_masks_globals = dgl.load_graphs(path_data_temp+"_test"+".bin")
    test_labels = test_labels_masks_globals["labels"].view(len(test_set),-1)
    test_masks = test_labels_masks_globals["masks"].view(len(test_set),-1)
    test_globals = test_labels_masks_globals["globals"].view(len(test_set),-1)
    
    '''Full dataset'''
    dataset = train_set + val_set + test_set

    data_labels = torch.cat((train_labels, val_labels), dim=0)
    data_labels = torch.cat((data_labels, test_labels), dim=0)

    data_masks = torch.cat((train_masks, val_masks), dim=0)
    data_masks = torch.cat((data_masks, test_masks), dim=0)

    data_globals = torch.cat((train_globals, val_globals), dim=0)
    data_globals = torch.cat((data_globals, test_globals), dim=0)    

    data_cross_validation={}
    seed = arguments.list_seeds[0]

    idx_all = list(np.arange(len(dataset)))
    idx_remained = idx_all
    fraction = 0.1
    for fold_idx in range(args.n_splits):
        idx_valid_fold = random.sample(idx_remained, int(fraction*len(dataset)))
        idx_remained = list(set(idx_remained)-set(idx_valid_fold))
        idx_test_fold = random.sample(idx_remained, int(fraction*len(dataset)))
        idx_remained = list(set(idx_remained)-set(idx_test_fold))                
        idx_train_fold = list(set(idx_all)-set(idx_valid_fold)-set(idx_test_fold))

        train_set = [dataset[i] for i in idx_train_fold]
        val_set = [dataset[i] for i in idx_valid_fold]
        test_set = [dataset[i] for i in idx_test_fold]

        train_labels = torch.index_select(data_labels, 0, torch.tensor(idx_train_fold))
        val_labels = torch.index_select(data_labels, 0, torch.tensor(idx_valid_fold))
        test_labels = torch.index_select(data_labels, 0, torch.tensor(idx_test_fold))

        train_masks = torch.index_select(data_masks, 0, torch.tensor(idx_train_fold))
        val_masks = torch.index_select(data_masks, 0, torch.tensor(idx_valid_fold))
        test_masks = torch.index_select(data_masks, 0, torch.tensor(idx_test_fold))

        train_globals = torch.index_select(data_globals, 0, torch.tensor(idx_train_fold))
        val_globals = torch.index_select(data_globals, 0, torch.tensor(idx_valid_fold))
        test_globals = torch.index_select(data_globals, 0, torch.tensor(idx_test_fold))                

        if args.scaler_regression:
            train = True
        else:
            train = False
        data_cross_validation[(seed, fold_idx, 1)] = TuneDatasetReg(train_set, train_labels, train_masks, train_globals, train=train, scaler_regression=args.scaler_regression)
        scaler[fold_idx] = data_cross_validation[(seed, fold_idx, 1)].scaler_method()
        data_cross_validation[(seed, fold_idx, 2)] = TuneDatasetReg(val_set, val_labels, val_masks, val_globals, scaler=scaler[fold_idx], scaler_regression=args.scaler_regression)
        data_cross_validation[(seed, fold_idx, 3)] = TuneDatasetReg(test_set, test_labels, test_masks, test_globals, scaler=scaler[fold_idx], scaler_regression=args.scaler_regression)

        print("train size:", len(data_cross_validation[(seed, fold_idx, 1)]),
            "valid size:", len(data_cross_validation[(seed, fold_idx, 2)]),
            "test_fold:", len(data_cross_validation[(seed, fold_idx, 3)]))
        val_size = len(data_cross_validation[(seed, fold_idx, 2)])
        test_size = len(data_cross_validation[(seed, fold_idx, 3)])

data={}
seed = arguments.list_seeds[0]
for fold_idx in range(args.n_splits):
    data[(seed, fold_idx, 1)], data[(seed, fold_idx, 2)], data[(seed, fold_idx, 3)]= data_cross_validation[(seed, fold_idx, 1)], data_cross_validation[(seed, fold_idx, 2)], data_cross_validation[(seed, fold_idx, 3)]

"""Hyperparameters tuning with Ray Tune"""

ray.init()
print(ray.available_resources())
ray.shutdown()  # Restart Ray defensively in case the ray connection is lost. 
ray.init(num_cpus=args.num_cpus, num_gpus=args.num_gpus) 

def main():
    os.makedirs(results_ray+"/"+args.name_data+"/"+args.division+"/Hierarchical_Quotient_type_False_Idx_Row_11/Both_False_Uni_Vert_False_#quotient_2_#layers_1/", exist_ok=True)
    storage_name =results_ray+"/"+args.name_data+"/"+args.division+"/Hierarchical_Quotient_type_False_Idx_Row_11/Both_False_Uni_Vert_False_#quotient_2_#layers_1/"+ \
        str(args.name_scheduler)+"_"+str(args.name_search_alg)

    config = {
        "GNN_Layers": tune.quniform(0, 5, 1), 
        "dropout": tune.quniform(0.05, 0.40, 0.05),
        "dropout1": tune.quniform(0.05, 0.40, 0.05),
        "dropout2": tune.quniform(0.05, 0.40, 0.05), 
        "lr": tune.quniform(0.0005, 0.001, 0.0001),  
        'max_norm_val': tune.quniform(2, 2.5, 0.5),     
        'hidden_size': tune.quniform(100, 200, 10),
        'readout1_out': tune.quniform(100, 300, 10),
        'readout2_out': tune.quniform(100, 300, 10),
    }  

    scheduler = scheduler_fn(args.name_scheduler, args.training_iteration, arguments.mode_ray)
    search_alg = search_alg_fn(args.name_search_alg, args.max_concurrent, arguments.mode_ray)
    if args.name_search_alg!=None:
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=args.max_concurrent)
   
    analysis=tune.run(tune.with_parameters(TrainableCV, data=data, scaler=scaler, val_size=val_size, test_size=test_size,
                global_size=arguments.global_size, num_tasks=arguments.num_tasks, global_feature=args.global_feature,
                n_splits=args.n_splits, batch_size=args.batch_size, list_seeds=arguments.list_seeds,
                task_type=arguments.task_type, training_iteration=args.training_iteration, ray_tune=arguments.ray_tune,
                scaler_regression=args.scaler_regression, max_norm_status=args.max_norm_status,
                atom_messages=args.atom_messages),
                    local_dir= storage_name,
                    scheduler=scheduler,
                    search_alg=search_alg,
                    num_samples=args.num_samples,
                    config=config,
                    verbose=2,
                    checkpoint_score_attr="metric_ray",
                    checkpoint_freq=0,
                    keep_checkpoints_num=1,
                    checkpoint_at_end=True,
                    resources_per_trial=PlacementGroupFactory([{"CPU": 1, "GPU": 0}]),
                    stop={"training_iteration": args.training_iteration},
                    )    
    '''
    # Get a dataframe for the last reported results of all of the trials
    df = analysis.results_df

    # Get a dataframe for the max accuracy seen for each trial
    df = analysis.dataframe(metric="mean_accuracy", mode="max")

    # Get a dict mapping {trial logdir -> dataframes} for all trials in the experiment.
    all_dataframes = analysis.trial_dataframes

    # Get a list of trials
    trials = analysis.trials
    '''
    best_config = analysis.get_best_config(metric="metric_ray", mode=arguments.mode_ray, scope="last")
    best_logdir = analysis.get_best_logdir(metric="metric_ray", mode=arguments.mode_ray, scope="last")
    
    result_df = analysis.results_df
    result_df.to_csv(result_csv)
    print("Ray best trial config: {}".format(best_config))  
    
    if arguments.task_type=="Classification":    
        ray_best_result = result_df["metric_ray"].max()
    else:
        ray_best_result = result_df["metric_ray"].min()
    print("Ray best trial final validation score: {}".format(ray_best_result))

    result_df = pd.read_csv(result_csv, index_col=0)
    training_iteration = args.training_iteration
    result_df = result_df[result_df.step == training_iteration]
    if arguments.task_type=="Classification":
        result_df = result_df.sort_values(by=["metric_ray"],ascending=False)
    else:
        result_df = result_df.sort_values(by=["metric_ray"],ascending=True)
    best_result = result_df.iloc[0]["metric_ray"]
    result_df_best = result_df[result_df.metric_ray==best_result]

    print("\n","Best trial final:")
    row = result_df.iloc[0]
    config = {
            "GNN_Layers": row["config.GNN_Layers"], 
            "dropout": row["config.dropout"], 
            "dropout1": row["config.dropout1"], 
            "dropout2": row["config.dropout2"], 
            "lr": row["config.lr"],   
            'hidden_size': row["config.hidden_size"], 
            'readout1_out': row["config.readout1_out"], 
            'readout2_out': row["config.readout2_out"]
            }
    if args.max_norm_status:
        config.update({'max_norm_val': row["config.max_norm_val"]})              
    print("metric score = {} \n config = {}\n".format(row["metric_ray"],config))

    return analysis

if __name__ == "__main__":
    analysis = main()


"""Training And Evaluation A GNN Model with Some Given Hyperparameters """

import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import sys
import shutil
import math 
import cloudpickle
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import random 
import pickle
from sklearn import preprocessing 

import arguments
from arguments import args

from dataset import DGLDatasetClass, DGLDatasetReg
from utils import loss_func, compute_score
from gnn import GNN

""" Set Path"""
current_dir = args.current_dir
checkpoint_path = current_dir + "save_models/model_checkpoints/" + args.name_data+"/"+args.division+"/"+ \
    "Hierarchical_Quotient_type_False_Idx_Row_11/Both_False_Uni_Vert_False_#quotient_2_#layers_1"
os.makedirs(checkpoint_path, exist_ok=True)
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
        train_set[seed] = DGLDatasetClass(address=path_data_temp+"_train")
        val_set[seed] = DGLDatasetClass(address=path_data_temp+"_val")
        test_set[seed] = DGLDatasetClass(address=path_data_temp+"_test")

elif arguments.task_type=="Regression":
    for name_seed in arguments.list_seeds:
        seed = name_seed
        path_data_temp = folder_data_temp + args.division+"_"+str(seed)
        if args.scaler_regression:
            train = True
        else:
            train = False
        train_set[seed] = DGLDatasetReg(address=path_data_temp+"_train", train=train, scaler_regression=args.scaler_regression)
        scaler.update({seed : train_set[seed].scaler_method()})
        val_set[seed] = DGLDatasetReg(address=path_data_temp+"_val", scaler=scaler[seed], scaler_regression=args.scaler_regression)
        test_set[seed] = DGLDatasetReg(address=path_data_temp+"_test", scaler=scaler[seed], scaler_regression=args.scaler_regression)
print(len(train_set[arguments.list_seeds[0]]), len(val_set[arguments.list_seeds[0]]), len(test_set[arguments.list_seeds[0]]))

""" Data Loader"""

def collate(batch):
    # batch is a list of tuples (graphs, labels, masks, globals)
    # Concatenate a sequence of graphs
    graphs = [e[0] for e in batch]
    g = dgl.batch(graphs)

    # Concatenate a sequence of tensors (labels) along a new dimension
    labels = [e[1] for e in batch]
    labels = torch.stack(labels, 0)

    # Concatenate a sequence of tensors (masks) along a new dimension
    masks = [e[2] for e in batch]
    masks = torch.stack(masks, 0)

    # Concatenate a sequence of tensors (globals) along a new dimension
    globals = [e[3] for e in batch]
    globals = torch.stack(globals, 0)

    return g, labels, masks, globals

def loader(seed, batch_size=args.config.get("batch_size", args.batch_size)):
    train_dataloader = DataLoader(train_set[seed],
                              batch_size=batch_size,
                              collate_fn=collate,
                              drop_last=False,
                              shuffle=True,
                              num_workers=1)

    val_dataloader =  DataLoader(val_set[seed],
                             batch_size=batch_size,
                             collate_fn=collate,
                             drop_last=False,
                             shuffle=False,
                             num_workers=1)

    test_dataloader = DataLoader(test_set[seed],
                             batch_size=batch_size,
                             collate_fn=collate,
                             drop_last=False,
                             shuffle=False,
                             num_workers=1)
    return train_dataloader, val_dataloader, test_dataloader

def max_norm(model, max_norm_val=3):
    for name, param in model.named_parameters():
        with torch.no_grad():
            if 'bias' not in name:
                norm = param.norm(2, dim=0, keepdim=True).clamp(min=max_norm_val / 2)
                desired = torch.clamp(norm, max=max_norm_val)
                param *= desired / norm                

if arguments.task_type=="Classification":
    def train_epoch(train_dataloader, model, optimizer, device):
        epoch_train_loss = 0
        iterations = 0
        model.train() # Prepare model for training
        for i, (mol_dgl_graph, labels, masks, globals) in enumerate(train_dataloader):
            mol_dgl_graph=mol_dgl_graph.to(device)
            labels=labels.to(device)
            masks=masks.to(device)
            globals=globals.to(device)       
            prediction = model(mol_dgl_graph, globals)
            loss_train = loss_func(prediction, labels, masks, arguments.task_type, arguments.num_tasks)
            optimizer.zero_grad(set_to_none=True)
            loss_train.backward()
            optimizer.step()
            if args.max_norm_status:
                max_norm(model, max_norm_val=args.config.get("max_norm_val", 3))
            epoch_train_loss += loss_train.detach().item()
            iterations += 1
        epoch_train_loss /= iterations
        return epoch_train_loss
else:
    def train_epoch(train_dataloader, model, optimizer, device):
        epoch_train_loss = 0
        mse_sum = nn.MSELoss(reduction='sum') # MSE with sum instead of mean
        model.train() # Prepare model for training
        for i, (mol_dgl_graph, labels, masks, globals) in enumerate(train_dataloader):
            mol_dgl_graph=mol_dgl_graph.to(device)
            labels=labels.to(device)
            masks=masks.to(device)
            globals=globals.to(device)           
            prediction = model(mol_dgl_graph, globals)
            loss_train = loss_func(prediction, labels, masks, arguments.task_type, arguments.num_tasks)
            optimizer.zero_grad(set_to_none=True)
            loss_train.backward()
            optimizer.step()
            if args.max_norm_status:
                max_norm(model, max_norm_val=args.config.get("max_norm_val", 3))
            loss = mse_sum(prediction, labels) # compute loss to print
            epoch_train_loss += loss.detach().item()
        epoch_train_loss /= (len(train_set[arguments.list_seeds[0]]))
        epoch_train_loss = math.sqrt(epoch_train_loss)
        return epoch_train_loss  

from prettytable import PrettyTable
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


""" Training and evaluation """

def train_evaluate():
    start_time = time.time()

    os.environ['PYTHONHASHSEED']=str(args.config.get("seed", 42))
    random.seed(args.config.get("seed", 42))
    np.random.seed(args.config.get("seed", 42))
    torch.manual_seed(args.config.get("seed", 42))
    if torch.cuda.is_available():        
        torch.cuda.manual_seed_all(args.config.get("seed", 42))
        dgl.seed(args.config.get("seed", 42))  
    device = args.device
    model = [GNN(args.config, arguments.global_size, arguments.num_tasks, args.global_feature)]
    model[0].to(device)
    optimizer = [torch.optim.Adam(model[0].parameters(), lr = args.config.get("lr", 0.0001))]
    train_dataloader, val_dataloader, test_dataloader = [None for i in range(args.n_splits)], [None for i in range(args.n_splits)], [None for i in range(args.n_splits)]
    train_dataloader[0], val_dataloader[0], test_dataloader[0] = loader(arguments.list_seeds[0], batch_size=args.config.get("batch_size", args.batch_size)) 
    for fold_idx in range(1, args.n_splits):
        model.append(copy.deepcopy(model[0]))
        model[fold_idx].to(device)
        optimizer.append(torch.optim.Adam(model[fold_idx].parameters(), lr = args.config.get("lr", 0.0001)))
        optimizer[fold_idx].load_state_dict(optimizer[0].state_dict())
        train_dataloader[fold_idx], val_dataloader[fold_idx], test_dataloader[fold_idx] = loader(arguments.list_seeds[fold_idx], batch_size=args.config.get("batch_size", args.batch_size)) 

    if arguments.task_type=="Classification":
        best_val = [0 for i in range(args.n_splits)]
    else:
        best_val = [np.Inf for i in range(args.n_splits)]
    patience_count = [1 for i in range(args.n_splits)]
    epoch = 1

    while epoch <= args.num_epochs:
        if min(patience_count) > args.patience:
            break
        for fold_idx in range(args.n_splits):
            if patience_count[fold_idx] <= args.patience:    
                model[fold_idx].train()
                loss_train_fold = train_epoch(train_dataloader[fold_idx], model[fold_idx], optimizer[fold_idx], device)
                model[fold_idx].eval()
                if arguments.task_type=="Classification":
                    score_val_fold = compute_score(model[fold_idx], val_dataloader[fold_idx], device, scaler, len(val_set[arguments.list_seeds[0]]), arguments.task_type, arguments.num_tasks, arguments.ray_tune, args.scaler_regression)
                    if score_val_fold > best_val[fold_idx]:
                        best_val[fold_idx] = score_val_fold
                        print("Save checkpoint of fold {}!".format(fold_idx+1))
                        path = os.path.join(checkpoint_path, 'checkpoint_'+str(fold_idx)+'.pth')
                        dict_checkpoint = {"score_val": score_val_fold}
                        dict_checkpoint.update({"model_state_dict": model[fold_idx].state_dict(), "optimizer_state": optimizer[fold_idx].state_dict()})
                        with open(path, "wb") as outputfile:
                            cloudpickle.dump(dict_checkpoint, outputfile)
                        patience_count[fold_idx] = 1
                    else:
                        print("Patience of fold {}:".format(fold_idx+1), patience_count[fold_idx])
                        patience_count[fold_idx] += 1
                else:
                    score_val_fold = compute_score(model[fold_idx], val_dataloader[fold_idx], device, scaler[fold_idx], len(val_set[arguments.list_seeds[0]]), arguments.task_type, arguments.num_tasks, arguments.ray_tune, args.scaler_regression)
                    if score_val_fold < best_val[fold_idx]:
                        best_val[fold_idx] = score_val_fold
                        print("Save checkpoint of fold {}!".format(fold_idx+1))
                        path = os.path.join(checkpoint_path, 'checkpoint_'+str(fold_idx)+'.pth')
                        dict_checkpoint = {"score_val": score_val_fold}
                        dict_checkpoint.update({"model_state_dict": model[fold_idx].state_dict(), "optimizer_state": optimizer[fold_idx].state_dict()})
                        with open(path, "wb") as outputfile:
                            cloudpickle.dump(dict_checkpoint, outputfile)
                        patience_count[fold_idx] = 1
                    else:
                        print("Patience of fold {}:".format(fold_idx+1), patience_count[fold_idx])
                        patience_count[fold_idx] += 1            

                print("Epoch: {}/{} | Fold: {}/{} | Training Loss: {:.3f} | Valid Score: {:.3f}".format(
                epoch, args.num_epochs, fold_idx+1, args.n_splits, loss_train_fold, score_val_fold))

        print(" ")
        print("Epoch: {}/{} | Average Valid Score: {:.3f}".format(epoch, args.num_epochs, np.mean(best_val)), "\n")
        epoch += 1

    print("Final results:")        
    print("Average Valid Score: {:.3f}".format(np.mean(best_val)), "\n")

    count_parameters(model[0])
    print(sum(param.numel() for param in model[0].parameters()))
    print(sum(param.numel() for param in model[0].parameters() if param.requires_grad))

    """Compute test set score of the final saved models""" 
    final_models = [GNN(args.config, arguments.global_size, arguments.num_tasks, args.global_feature) for i in range(args.n_splits)]
    test_scores = []
    for fold_idx in range(args.n_splits):
        path = os.path.join(checkpoint_path, 'checkpoint_'+str(fold_idx)+'.pth')
        with open(path, 'rb') as f:
            checkpoint = cloudpickle.load(f)
        final_models[fold_idx].load_state_dict(checkpoint["model_state_dict"])
        final_models[fold_idx].eval()
        if arguments.task_type=="Classification": 
            test_scores.append(compute_score(final_models[fold_idx], test_dataloader[fold_idx], device, scaler, len(test_set[arguments.list_seeds[0]])), arguments.task_type, arguments.num_tasks, arguments.ray_tune, args.scaler_regression)
        else:
            test_scores.append(compute_score(final_models[fold_idx], test_dataloader[fold_idx], device, scaler[fold_idx], len(test_set[arguments.list_seeds[0]])), arguments.task_type, arguments.num_tasks, arguments.ray_tune, args.scaler_regression)

    print("Average Test Score: {:.3f} | SEM of Test Scores: {:.3f} | STD of Test Scores: {:.3f}".format(np.mean(test_scores), np.std(test_scores)/np.sqrt(args.n_splits), np.std(test_scores)), "\n")

    print("Execution time: {:.3f} seconds".format(time.time() - start_time))

if __name__ == "__main__":
    train_evaluate()


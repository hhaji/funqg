
""" Explain GNN """

import os
import random
from scipy.stats import gmean
from datetime import datetime
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
from dataset import DGLDatasetClass
import dgl.function as fn
import torch
import torch.nn as nn
from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from functools import partial
from sklearn.model_selection import KFold
import copy
import arguments
import random 
import pickle
from sklearn import preprocessing
from captum.attr import IntegratedGradients
from arguments import args 
from gnn_explain import GNN

current_dir = args.current_dir
device ="cpu"
name_final = "Hierarchical_Quotient_type_False_Both_False_Uni_Vert_False_#quotient_2_#layers_1_127_one_hot"
arguments.name_final_zip = name_final+".zip"

for name_data in ["tox21", "sider", "bace", "freesolv", "lipo", "esol", "bbbp","clintox"]:
    args.name_data = name_data
    l1 = [args.name_data]
    if args.name_data in ["tox21", "toxcast", "clintox", "sider", "bbbp", "bace", "muv", "hiv"]:
        arguments.task_type = "Classification"  
    elif args.name_data in ["freesolv", "esol", "lipo", "qm7", "qm8", "pdbbind_r", "pdbbind_c", "pdbbind_f"]:
        arguments.task_type = "Regression"      
    if args.name_data=="tox21":
        arguments.dataset_metric = "ROC-AUC"
        arguments.num_tasks = 12
    elif args.name_data=="bbbp":
        arguments.dataset_metric = "ROC-AUC"
        arguments.num_tasks = 1   
    elif args.name_data=="bace":
        arguments.dataset_metric = "ROC-AUC"
        arguments.num_tasks = 1   
    elif args.name_data=="clintox":
        arguments.dataset_metric = "ROC-AUC"
        arguments.num_tasks = 2   
    elif args.name_data=="toxcast":
        arguments.dataset_metric = "ROC-AUC"
        arguments.num_tasks = 617   
    elif args.name_data=="sider":
        arguments.dataset_metric = "ROC-AUC"
        arguments.num_tasks = 27   
    elif args.name_data=="muv":
        arguments.dataset_metric = "PRC-AUC"
        arguments.num_tasks = 17  
    elif args.name_data=="hiv":
        arguments.dataset_metric = "ROC-AUC"
        arguments.num_tasks = 1           
    elif args.name_data=="lipo":
        arguments.dataset_metric = "RMSE"
        arguments.num_tasks = 1   
    elif args.name_data=="esol":
        arguments.dataset_metric = "RMSE"
        arguments.num_tasks = 1   
    elif args.name_data=="freesolv":
        arguments.dataset_metric = "RMSE"
        arguments.num_tasks = 1   
    elif args.name_data=="qm7":
        arguments.dataset_metric = "MAE"
        arguments.num_tasks = 1   
    elif args.name_data=="qm8":
        arguments.dataset_metric = "MAE"
        arguments.num_tasks = 12   
    elif args.name_data=="pdbbind_r" or args.name_data=="pdbbind_c" or args.name_data=="pdbbind_f":
        arguments.dataset_metric = "RMSE"
        arguments.num_tasks = 1           
    for atom_messages in [False, True]:
        args.atom_messages = atom_messages
        if args.atom_messages:
           l2 = ["funqg-mpnn"]
           if args.name_data=="tox21":
               args.config = {"GNN_Layers": 2.0, "dropout": 0.35, "dropout1": 0.4, "dropout2": 0.35, "lr": 0.001, "hidden_size": 170.0, "readout1_out": 240.0, "readout2_out": 120.0, "max_norm_val": 2.0}
           elif args.name_data=="bbbp":
               args.config = {"GNN_Layers": 0, "dropout": 0.4, "dropout1": 0.25, "dropout2": 0.2, "lr": 0.0008, "hidden_size": 140.0, "readout1_out": 280.0, "readout2_out": 290.0, "max_norm_val": 2.5}
           elif args.name_data=="bace":
               args.config = {"GNN_Layers": 0, "dropout": 0.05, "dropout1": 0.25, "dropout2": 0.1, "lr": 0.0007, "hidden_size": 150.0, "readout1_out": 100.0, "readout2_out": 280.0, "max_norm_val": 2.0}
           elif args.name_data=="clintox":
               args.config = {"GNN_Layers": 0.0, "dropout": 0.15, "dropout1": 0.3, "dropout2": 0.3, "lr": 0.0009, "hidden_size": 150.0, "readout1_out": 150.0, "readout2_out": 270.0, "max_norm_val": 2.5}
           elif args.name_data=="toxcast":
               args.config = {"GNN_Layers": 0.0, "dropout": 0.4, "dropout1": 0.4, "dropout2": 0.15, "lr": 0.001, "hidden_size": 190.0, "readout1_out": 190.0, "readout2_out": 210.0, "max_norm_val": 2.0}
           elif args.name_data=="sider": 
               args.config = {"GNN_Layers": 1.0, "dropout": 0.1, "dropout1": 0.4, "dropout2": 0.05, "lr": 0.001, "hidden_size": 130.0, "readout1_out": 230.0, "readout2_out": 300.0, "max_norm_val": 2.0}         
           elif args.name_data=="muv":
               args.config = {"GNN_Layers": 1.0, "dropout": 0.15, "dropout1": 0.35, "dropout2": 0.15, "lr": 0.0005, "hidden_size": 140.0, "readout1_out": 270.0, "readout2_out": 220.0, "max_norm_val": 2.5}
           elif args.name_data=="hiv":
               args.config = {"GNN_Layers": 2.0, "dropout": 0.1, "dropout1": 0.3, "dropout2": 0.05, "lr": 0.0008, "hidden_size": 140.0, "readout1_out": 210.0, "readout2_out": 220.0, "max_norm_val": 2.0}
           elif args.name_data=="lipo":
               args.config = {"GNN_Layers": 1.0, "dropout": 0.05, "dropout1": 0.2, "dropout2": 0.05, "lr": 0.0007, "hidden_size": 160.0, "readout1_out": 210.0, "readout2_out": 140.0, "max_norm_val": 2.0}
           elif args.name_data=="esol":
               args.config = {"GNN_Layers": 0.0, "dropout": 0.3, "dropout1": 0.05, "dropout2": 0.2, "lr": 0.0008, "hidden_size": 170.0, "readout1_out": 140.0, "readout2_out": 200.0, "max_norm_val": 2.0}
           elif args.name_data=="freesolv":
               args.config = {"GNN_Layers": 0, "dropout": 0.05, "dropout1": 0.3, "dropout2": 0.05, "lr": 0.0009, "hidden_size": 100.0, "readout1_out": 260.0, "readout2_out": 270.0, "max_norm_val": 2.0}
           elif args.name_data=="qm7":
               args.config = {"GNN_Layers": 4.0, "dropout": 0.05, "dropout1": 0.05, "dropout2": 0.05, "lr": 0.0008, "hidden_size": 170.0, "readout1_out": 260.0, "readout2_out": 290.0, "max_norm_val": 2.0}
           elif args.name_data=="qm8":
               args.config = {"GNN_Layers": 2.0, "dropout": 0.3, "dropout1": 0.2, "dropout2": 0.2, "lr": 0.0008, "hidden_size": 120.0, "readout1_out": 220.0, "readout2_out": 260.0, "max_norm_val": 2.0}
           elif args.name_data=="pdbbind_r":  
               args.config = {"GNN_Layers": 1.0, "dropout": 0.2, "dropout1": 0.15, "dropout2": 0.05, "lr": 0.001, "hidden_size": 100.0, "readout1_out": 290.0, "readout2_out": 200.0, "max_norm_val": 2.0}
           elif args.name_data=="pdbbind_c": 
               args.config = {"GNN_Layers": 1.0, "dropout": 0.05, "dropout1": 0.2, "dropout2": 0.05, "lr": 0.001, "hidden_size": 110.0, "readout1_out": 120.0, "readout2_out": 150.0, "max_norm_val": 2.5}
           elif args.name_data=="pdbbind_f":
               args.config = {"GNN_Layers": 2.0, "dropout": 0.05, "dropout1": 0.3, "dropout2": 0.15, "lr": 0.0007, "hidden_size": 140.0, "readout1_out": 290.0, "readout2_out": 250.0, "max_norm_val": 2.0}
  
        else:
           l2 = ["funqg-dmpnn"]
           if args.name_data=="tox21":
               args.config = {"GNN_Layers": 4, "dropout": 0.35, "dropout1": 0.15, "dropout2": 0.1, "lr": 0.001, "batch_size": 64, "hidden_size": 100, "readout1_out": 180, "readout2_out": 120, "max_norm_val": 2.5}
           elif args.name_data=="bbbp":
               args.config = {"GNN_Layers": 0, "dropout": 0.25, "dropout1": 0.25, "dropout2": 0.4, "lr": 0.0007, "batch_size": 64, "hidden_size": 140.0, "readout1_out": 160.0, "readout2_out": 270.0, "max_norm_val": 2}
           elif args.name_data=="bace":
               args.config = {"GNN_Layers": 0, "dropout": 0.15, "dropout1": 0.15, "dropout2": 0.1, "lr": 0.0003, "batch_size": 64, "hidden_size": 110.0, "readout1_out": 240.0, "readout2_out": 240.0, "max_norm_val": 2}
           elif args.name_data=="clintox":
               args.config = {"GNN_Layers": 0.0, "dropout": 0.1, "dropout1": 0.2, "dropout2": 0.2, "lr": 0.0004, "batch_size": 64, "hidden_size": 200.0, "readout1_out": 160.0, "readout2_out": 100.0, "max_norm_val": 2}
           elif args.name_data=="toxcast":
               args.config = {"GNN_Layers": 0.0, "dropout": 0.3, "dropout1": 0.2, "dropout2": 0.2, "lr": 0.0006, "batch_size": 64, "hidden_size": 180.0, "readout1_out": 220.0, "readout2_out": 240.0, "max_norm_val": 2.5}
           elif args.name_data=="sider": 
               args.config = {"GNN_Layers": 5.0, "dropout": 0.2, "dropout1": 0.25, "dropout2": 0.2, "lr": 0.001, "batch_size": 64, "hidden_size": 110.0, "readout1_out": 290.0, "readout2_out": 180.0, "max_norm_val": 2.5}         
           elif args.name_data=="muv":
               args.config = {"GNN_Layers": 5.0, "dropout": 0.05, "dropout1": 0.4, "dropout2": 0.05, "lr": 0.0005, "hidden_size": 140.0, "readout1_out": 160.0, "readout2_out": 200.0, "max_norm_val": 2.5}
           elif args.name_data=="hiv":
               args.config = {"GNN_Layers": 2.0, "dropout": 0.1, "dropout1": 0.35, "dropout2": 0.2, "lr": 0.0004, "hidden_size": 160.0, "readout1_out": 200.0, "readout2_out": 270.0, "max_norm_val": 2.0}
           elif args.name_data=="lipo":
               args.config = {"GNN_Layers": 5.0, "dropout": 0.15, "dropout1": 0.35, "dropout2": 0.15, "lr": 0.0006, "batch_size": 64, "hidden_size": 180.0, "readout1_out": 180.0, "readout2_out": 180.0, "max_norm_val": 2.5}
           elif args.name_data=="esol":
               args.config = {"GNN_Layers": 2.0, "dropout": 0.15, "dropout1": 0.05, "dropout2": 0.25, "lr": 0.0009, "batch_size": 64, "hidden_size": 130.0, "readout1_out": 300.0, "readout2_out": 140.0, "max_norm_val": 2}
           elif args.name_data=="freesolv":
               args.config = {"GNN_Layers": 0.0, "dropout": 0.3, "dropout1": 0.2, "dropout2": 0.25, "lr": 0.0007, "batch_size": 64, "hidden_size": 120.0, "readout1_out": 290.0, "readout2_out": 110.0, "max_norm_val": 2.5} 
           elif args.name_data=="qm7":
               args.config = {"GNN_Layers": 4.0, "dropout": 0.15, "dropout1": 0.1, "dropout2": 0.05, "lr": 0.0008, "hidden_size": 150.0, "readout1_out": 230.0, "readout2_out": 270.0, "max_norm_val": 2.5}
           elif args.name_data=="qm8":
               args.config = {"GNN_Layers": 4.0, "dropout": 0.05, "dropout1": 0.1, "dropout2": 0.25, "lr": 0.0007, "hidden_size": 160.0, "readout1_out": 200.0, "readout2_out": 240.0, "max_norm_val": 2.5}
           elif args.name_data=="pdbbind_r":  
               args.config = {"GNN_Layers": 4.0, "dropout": 0.05, "dropout1": 0.15, "dropout2": 0.05, "lr": 0.001, "hidden_size": 170.0, "readout1_out": 290.0, "readout2_out": 300.0, "max_norm_val": 2.0}
           elif args.name_data=="pdbbind_c": 
               args.config = {"GNN_Layers": 1.0, "dropout": 0.4, "dropout1": 0.35, "dropout2": 0.2, "lr": 0.0009, "hidden_size": 110.0, "readout1_out": 210.0, "readout2_out": 290.0, "max_norm_val": 2.0}
           elif args.name_data=="pdbbind_f":
               args.config = {"GNN_Layers": 8.0, "dropout": 0.05, "dropout1": 0.3, "dropout2": 0.05, "lr": 0.0001, "hidden_size": 180.0, "readout1_out": 500, "readout2_out": 190.0, "max_norm_val": 2.0}

        name_model = l2[0]
        df = pd.read_csv("results_explain.csv")
        df_one = pd.isna(df.loc[(df["name_data"]==args.name_data) & (df["name_model"]==name_model), ["result_fg"]])
        
        if df_one.iloc[0,0]:
            print(l1+l2)
            if args.atom_messages:
                best_model_path = current_dir +"data/best_model_mpnn/" + args.name_data + "/" + "checkpoint_0.pth" 
            else:
                best_model_path = current_dir +"data/best_model_dmpnn/" + args.name_data + "/" + "checkpoint_0.pth"

            folder_data_temp = current_dir +"data_temp/"
            shutil.rmtree(folder_data_temp, ignore_errors=True) 
            path_save = current_dir + "data/graph/"+args.name_data+"/"+args.division+"_"+str(0)+"/"+arguments.name_final_zip
            shutil.unpack_archive(path_save, folder_data_temp)
            path_data_temp = folder_data_temp + args.division+"_"+str(0)
            test_set = DGLDatasetClass(address=path_data_temp+"_test")

            result_fg=[]
            for index_tasks in range(arguments.num_tasks):
                os.environ['PYTHONHASHSEED']=str(0)
                random.seed(0)
                np.random.seed(0)
                torch.manual_seed(0)
                if torch.cuda.is_available():        
                    torch.cuda.manual_seed_all(0)
                dgl.seed(0)
                model =GNN(args.config, arguments.global_size, arguments.num_tasks, args.global_feature, args.atom_messages)
                with open(best_model_path, 'rb') as f:
                    dict_checkpoint=cloudpickle.load(f)
                model.load_state_dict(dict_checkpoint["model_state_dict"])
                count=0
                for t in range(len(test_set)):
                    g= test_set[t][0]
                    g.ndata['globals'] = dgl.broadcast_nodes(g, test_set[t][3].view(1,-1))
                    ig = IntegratedGradients(partial(model.forward, graph=g, index_tasks=index_tasks))
                    ig_attr_node = ig.attribute(g.ndata['v'], target=None,
                                                internal_batch_size=g.num_nodes(), n_steps=50)
                    ig_attr_node = ig_attr_node.abs().sum(dim=1)
                    ig_attr_node /= ig_attr_node.max()
                    index_best_node=np.argmax(ig_attr_node.detach().numpy())
                    best_node_feature=g.ndata["v"][index_best_node][0:100]
                    if 5 not in np.argwhere(best_node_feature>0) or len(np.argwhere(best_node_feature>0)[0])>=2:
                        count+=1
                result_fg.append(count)

            print(result_fg)
            print(np.mean(result_fg))
            print(np.mean(result_fg)/len(test_set))
            df.loc[(df["name_data"]==args.name_data) & (df["name_model"]==name_model), ["result_fg"]] = str(result_fg)
            df.loc[(df["name_data"]==args.name_data) & (df["name_model"]==name_model), ["mean_result_fg"]] = str(np.mean(result_fg))
            df.loc[(df["name_data"]==args.name_data) & (df["name_model"]==name_model), ["mean_result_fg/len_test_set"]] = str(np.mean(result_fg)/len(test_set))
            df.to_csv("results_explain.csv", index = False)





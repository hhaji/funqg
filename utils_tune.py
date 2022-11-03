

""" Utils for Hyperparameters Optimization with Ray Tune """

import os
import shutil
import cloudpickle
from ray import tune
import dgl
import torch
import torch.nn as nn
import random 
import numpy as np

import arguments
from gnn import GNN
from utils import compute_score, loss_func 
import copy
from torch.utils.data import DataLoader

from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.optuna import OptunaSearch  
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.schedulers import MedianStoppingRule
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.suggest import ConcurrencyLimiter

def max_norm(model, max_norm_val):
    for name, param in model.named_parameters():
        with torch.no_grad():
            if 'bias' not in name:
                norm = param.norm(2, dim=0, keepdim=True).clamp(min=max_norm_val / 2)
                desired = torch.clamp(norm, max=max_norm_val)
                param *= desired / norm
                          
def train_ray_tune(config, train_dataloader, model, optimizer, device, task_type, num_tasks, max_norm_status):
    model.train() # Prepare model for training
    for i, (mol_dgl_graph, labels, masks, globals) in enumerate(train_dataloader):
        mol_dgl_graph=mol_dgl_graph.to(device)
        labels=labels.to(device)
        masks=masks.to(device)
        globals=globals.to(device)
        prediction = model(mol_dgl_graph, globals)
        loss_train = loss_func(prediction, labels, masks, task_type, num_tasks)
        optimizer.zero_grad(set_to_none=True)
        loss_train.backward()
        optimizer.step()
        if max_norm_status:
            max_norm(model, max_norm_val=config.get("max_norm_val", 3))
   
""" Trainable Class for Quasi-NVC """

class TrainableCV(tune.Trainable):
    def setup(self, config=None, data=None, scaler=None, val_size=None, test_size=None,
           global_size=None, num_tasks=None, global_feature=None,
                n_splits=None, batch_size=None, list_seeds=None,
                task_type=None, training_iteration=None, ray_tune=None,
                scaler_regression=None, max_norm_status=None, atom_messages=None):
        os.environ['PYTHONHASHSEED']=str(config.get("seed", 42))
        random.seed(config.get("seed", 42))
        np.random.seed(config.get("seed", 42))
        torch.manual_seed(config.get("seed", 42))
        if torch.cuda.is_available():        
            torch.cuda.manual_seed_all(config.get("seed", 42))
            dgl.seed(config.get("seed", 42))  
        self.scaler = scaler
        self.val_size = val_size
        self.test_size = test_size
        self.global_size = global_size
        self.num_tasks = num_tasks
        self.global_feature = global_feature
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.list_seeds = list_seeds
        self.task_type = task_type
        self.training_iter= training_iteration
        self.ray_tune = ray_tune
        self.scaler_regression = scaler_regression
        self.max_norm_status = max_norm_status
        self.atom_messages = atom_messages
        self.device = "cpu" 
        self.model = [GNN(config, self.global_size, self.num_tasks, self.global_feature, self.atom_messages)]
        for fold_idx in range(1, self.n_splits):
            self.model.append(copy.deepcopy(self.model[0]))
        if torch.cuda.is_available():
            self.device = "cuda:0"
            if torch.cuda.device_count() > 1:
                for fold_idx in range(self.n_splits):
                    self.model[fold_idx] = nn.DataParallel(self.model[fold_idx])

        self.model[0].to(self.device)
        self.optimizer = [torch.optim.Adam(self.model[0].parameters(), lr = round(config.get("lr", 0.0001),4))]
        ''''''
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
            
        def loader_cv(seed, fold_idx, batch_size=config.get("batch_size", self.batch_size)):
            train_dataloader= DataLoader(data_cross_validation[(seed, fold_idx, 1)],
                                batch_size=batch_size,
                                collate_fn=collate,
                                drop_last=False,
                                shuffle=True)
            val_dataloader= DataLoader(data_cross_validation[(seed, fold_idx, 2)],
                                batch_size=batch_size,
                                collate_fn=collate,
                                drop_last=False,
                                shuffle=False)
            test_dataloader = DataLoader(data_cross_validation[(seed, fold_idx, 3)],
                                batch_size=batch_size,
                                collate_fn=collate,
                                drop_last=False,
                                shuffle=False)     
            return train_dataloader, val_dataloader, test_dataloader

        data_cross_validation={}
        seed = self.list_seeds[0]
        for fold_idx in range(self.n_splits):
            data_cross_validation[(seed, fold_idx, 1)],data_cross_validation[(seed, fold_idx, 2)],data_cross_validation[(seed, fold_idx, 3)]=\
            data[(seed, fold_idx, 1)],data[(seed, fold_idx, 2)], data[(seed, fold_idx, 3)]
            data[(seed, fold_idx, 1)], data[(seed, fold_idx, 2)], data[(seed, fold_idx, 3)]= loader_cv(seed, fold_idx) 
        ''''''
        self.train_dataloader, self.val_dataloader, self.test_dataloader = [data[(self.list_seeds[0], 0, 1)]], [data[(self.list_seeds[0], 0, 2)]], [data[(self.list_seeds[0], 0, 3)]]
        for fold_idx in range(1, self.n_splits):
            self.model[fold_idx].to(self.device)
            self.optimizer.append(torch.optim.Adam(self.model[fold_idx].parameters(), lr = round(config.get("lr", 0.0001),4)))
            self.optimizer[fold_idx].load_state_dict(self.optimizer[0].state_dict())
            self.train_dataloader.append(data[(self.list_seeds[0], fold_idx, 1)])
            self.val_dataloader.append(data[(self.list_seeds[0], fold_idx, 2)])
            self.test_dataloader.append(data[(self.list_seeds[0], fold_idx, 3)])
        if self.task_type=="Classification":
            self.best_val = [0 for i in range(self.n_splits)]
            self.best_score = 0
        else:
            self.best_val = [np.Inf for i in range(self.n_splits)]
            self.best_score = np.Inf            
        self.step_trial = 0
 
    def step(self):
        score_folds = []
        self.step_trial += 1
        val_dataloader = []
        if self.step_trial < self.training_iter:
            for fold_idx in range(self.n_splits):
                val_dataloader.append(self.val_dataloader[fold_idx])
                val_size = self.val_size
        else:
            for fold_idx in range(self.n_splits):
                val_dataloader.append(self.test_dataloader[fold_idx])
                val_size = self.test_size            

        for fold_idx in range(self.n_splits):
            train_ray_tune(self.config, self.train_dataloader[fold_idx], self.model[fold_idx], self.optimizer[fold_idx], self.device, self.task_type, self.num_tasks, self.max_norm_status)
            if self.task_type=="Classification":
                score_val_fold = compute_score(self.model[fold_idx], val_dataloader[fold_idx], self.device, self.scaler, val_size, self.task_type, self.num_tasks, self.ray_tune, self.scaler_regression)
                score_folds.append(score_val_fold)
            else:
                score_val_fold = compute_score(self.model[fold_idx], val_dataloader[fold_idx], self.device, self.scaler[fold_idx], val_size, self.task_type, self.num_tasks, self.ray_tune, self.scaler_regression)
                score_folds.append(score_val_fold)

        score_val = round(np.mean(score_folds),3)
        result = {"step": self.step_trial, "metric_ray": score_val}
        if self.step_trial < self.training_iter:
            if self.task_type=="Classification" and result["metric_ray"] >= self.best_score:
                result.update(should_checkpoint=True)
                self.best_val = score_folds
                self.best_score = result["metric_ray"]
            elif self.task_type=="Regression" and result["metric_ray"] <= self.best_score:
                result.update(should_checkpoint=True)
                self.best_val = score_folds
                self.best_score = result["metric_ray"]    
        else:
            result.update(should_checkpoint=True)
            self.best_val = score_folds
            self.best_score = result["metric_ray"]
        return result
 
    def save_checkpoint(self, checkpoint_dir=None):
        print("Save Checkpoint!")
        path = os.path.join(checkpoint_dir, "checkpoint.pth")
        dict_checkpoint = {"metric_ray": self.best_score}
        for fold_idx in range(self.n_splits):
            dict_checkpoint.update({"model_state_dict_{}".format(fold_idx): self.model[fold_idx].state_dict(),
                                    "optimizer_state_{}".format(fold_idx): self.optimizer[fold_idx].state_dict()})
        with open(path, "wb") as outputfile:
            cloudpickle.dump(dict_checkpoint, outputfile)
        return path
 
    def load_checkpoint(self, checkpoint_path):
        print("Load from Checkpoint!")
        with open(checkpoint_path, "rb") as inputfile:
            checkpoint = cloudpickle.load(inputfile)
        for fold_idx in range(self.n_splits):
            self.model[fold_idx].load_state_dict(checkpoint["model_state_dict_{}".format(fold_idx)])
            self.optimizer[fold_idx].load_state_dict(checkpoint["optimizer_state_{}".format(fold_idx)])

    
""" Schedulers And Search Algorithms of Ray Tune for Hyperparameters Optimization """

def scheduler_fn(name_scheduler=None, training_iter=None, mode_ray=None):  

    if name_scheduler==None:
        scheduler = None  

    if name_scheduler=="asha":
        scheduler = AsyncHyperBandScheduler(
            time_attr="training_iteration",
            max_t=training_iter,
            metric="metric_ray",
            mode=mode_ray,  
            reduction_factor=2, 
            grace_period=4,
            brackets=5,
            )

    if name_scheduler=="bohb":
        scheduler = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=training_iter,
        reduction_factor=8, 
        stop_last_trials=True,
        metric="metric_ray",  
        mode=mode_ray)

    if name_scheduler=="median":
        scheduler = MedianStoppingRule(
        time_attr="training_iteration",
        grace_period=10,
        min_samples_required=10,
        hard_stop = True,
        metric="metric_ray",  
        mode=mode_ray)

    return scheduler

def search_alg_fn(name_search_alg=None, max_concur=None, mode_ray=None):

    if name_search_alg==None:
        search_alg = None

    if name_search_alg=="bohb":
        search_alg = TuneBOHB(
        # space=config_space,  # If you want to set the space manually
        metric="metric_ray",  
        mode=mode_ray,
        )
        search_alg = tune.suggest.ConcurrencyLimiter(search_alg, max_concurrent=max_concur)

    if name_search_alg=="hyperopt":
        search_alg = HyperOptSearch(
            # space=config,
            metric="metric_ray", 
            mode=mode_ray,
            n_initial_points=60,
            )

    if name_search_alg=="optuna":
        search_alg = OptunaSearch(
            metric="metric_ray", 
            mode=mode_ray,
            )

    return search_alg

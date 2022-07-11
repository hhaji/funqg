
""" Custom PyTorch Datasets """

import torch
import dgl
from sklearn import preprocessing 

""" Classification Dataset """
class DGLDatasetClass(torch.utils.data.Dataset):
    def __init__(self, address):
            self.address=address+".bin"
            self.list_graphs, train_labels_masks_globals = dgl.load_graphs(self.address)
            num_graphs =len(self.list_graphs)
            self.labels = train_labels_masks_globals["labels"].view(num_graphs,-1)
            self.masks = train_labels_masks_globals["masks"].view(num_graphs,-1)
            self.globals = train_labels_masks_globals["globals"].view(num_graphs,-1)
    def __len__(self):
        return len(self.list_graphs)
    def __getitem__(self, idx):
        return  self.list_graphs[idx], self.labels[idx], self.masks[idx], self.globals[idx]

""" Regression Dataset """
class DGLDatasetReg(torch.utils.data.Dataset):
    def __init__(self, address, transform=None, train=False, scaler=None, scaler_regression=None):
            self.train = train 
            self.scaler = scaler
            self.data_set, train_labels_masks_globals = dgl.load_graphs(address+".bin")
            num_graphs = len(self.data_set)
            self.labels = train_labels_masks_globals["labels"].view(num_graphs,-1)
            self.masks = train_labels_masks_globals["masks"].view(num_graphs,-1)
            self.globals = train_labels_masks_globals["globals"].view(num_graphs,-1)
            self.transform = transform
            self.scaler_regression = scaler_regression
    def scaler_method(self):
        if self.train:
            scaler = preprocessing.StandardScaler().fit(self.labels) 
            self.scaler = scaler
        return self.scaler
    def __len__(self):
        return len(self.data_set)
    def __getitem__(self, idx):       
        if self.scaler_regression:
            """ With Scaler"""
            return  self.data_set[idx], torch.tensor(self.scaler.transform(self.labels)[idx]).float(), self.masks[idx], self.globals[idx]
        else:
            """ Without Scaler """
            return  self.data_set[idx], self.labels[idx].float(), self.masks[idx], self.globals[idx]

""" Tune Regression Dataset """
class TuneDatasetReg(torch.utils.data.Dataset):
    def __init__(self, graphs_list, labels, masks, globals, train=False, scaler=None, scaler_regression=None):
            self.train = train 
            self.scaler = scaler
            self.data_set = graphs_list
            self.labels = labels
            self.masks = masks
            self.globals = globals
            self.scaler_regression = scaler_regression
    def scaler_method(self):
        if self.train:
            scaler = preprocessing.StandardScaler().fit(self.labels) 
            self.scaler = scaler
        return self.scaler
    def __len__(self):
        return len(self.data_set)
    def __getitem__(self, idx):        
        if self.scaler_regression:
            """ With Scaler"""
            return  self.data_set[idx], torch.tensor(self.scaler.transform(self.labels)[idx]).float(), self.masks[idx], self.globals[idx]
        else:
            """ Without Scaler """
            return  self.data_set[idx], self.labels[idx].float(), self.masks[idx], self.globals[idx]            
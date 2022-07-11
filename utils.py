
""" Functions to Compute Scores of Models """

import torch
import torch.nn as nn
import math
from sklearn.metrics import roc_auc_score

def compute_score(model, data_loader, device, scaler, val_size, task_type, num_tasks, ray_tune, scaler_regression):
    if task_type=="Classification":
        model.eval()
        metric = roc_auc_score
        state = torch.get_rng_state()
        with torch.no_grad():
            prediction_all= torch.empty(0, device=device)
            labels_all= torch.empty(0, device=device)
            masks_all= torch.empty(0, device=device)
            for i, (mol_dgl_graph, labels, masks, globals) in enumerate(data_loader):
                mol_dgl_graph=mol_dgl_graph.to(device)
                labels=labels.to(device)
                masks=masks.to(device)
                globals=globals.to(device)
                prediction = model(mol_dgl_graph, globals).to(device)
                prediction = torch.sigmoid(prediction).to(device)
                prediction_all = torch.cat((prediction_all, prediction), 0)
                labels_all = torch.cat((labels_all, labels), 0)
                masks_all = torch.cat((masks_all, masks), 0)
            average = torch.tensor([0.], device=device)
            for i in range(num_tasks):
                a1 = prediction_all[:, i][masks_all[:,i]==1]
                a2 = labels_all[:, i][masks_all[:,i]==1]
                try:
                    t = metric(a2.int().cpu(), a1.cpu()).item()
                except ValueError:
                    t = 0
                average += t
        if ray_tune==False:
            torch.set_rng_state(state)
        return average.item()/num_tasks
    else:
        model.eval()
        mse_sum = nn.MSELoss(reduction='sum') # MSE with sum instead of mean, i.e., sum_i[(y_i)^2-(y'_i)^2]
        final_loss = 0
        state = torch.get_rng_state()
        with torch.no_grad():
            for i, (mol_dgl_graph, labels, masks, globals) in enumerate(data_loader):
                mol_dgl_graph=mol_dgl_graph.to(device)
                labels=labels.to(device)
                masks=masks.to(device)
                globals=globals.to(device)
                prediction = model(mol_dgl_graph, globals).to(device)  
                if scaler_regression:
                    prediction = torch.tensor(scaler.inverse_transform(prediction.detach().cpu())).to(device)    
                    labels = torch.tensor(scaler.inverse_transform(labels.cpu())).to(device)                                       
                loss = mse_sum(prediction, labels)
                final_loss += loss.item()
            final_loss /= val_size
            final_loss = math.sqrt(final_loss) # RMSE
        if ray_tune==False:
            torch.set_rng_state(state)   
        return final_loss



""" Loss Function """

def loss_func(output, label, mask, task_type, num_tasks):
    pos_weight = torch.ones((1, num_tasks))
    pos_weight
    if task_type=="Classification":
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
        loss = mask*criterion(output,label)
        loss = loss.sum() / mask.sum()
        return loss

    elif task_type=="Regression": 
        criterion = nn.MSELoss(reduction='none')
        loss = mask*criterion(output,label)
        loss = loss.sum() / mask.sum()
        return loss

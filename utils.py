
""" Loss Functions and Compute Scores Functions """

import torch
import torch.nn as nn
import math
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def prc_auc(targets, preds):
    """
    Computes the area under the precision-recall curve.
    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed prc-auc.
    """
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)


def compute_score(model, data_loader, device, scaler, val_size, task_type, num_tasks, ray_tune, scaler_regression, dataset_metric):
    if task_type=="Classification":
        model.eval()
        if dataset_metric=="ROC-AUC":
            metric = roc_auc_score
        elif dataset_metric=="PRC-AUC":
            metric = prc_auc
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
                pred_task = prediction_all[:, i][masks_all[:,i]==1]
                target_task = labels_all[:, i][masks_all[:,i]==1]
                # if args.name_data=="muv":
                nan = False
                if all(target == 0 for target in target_task) or all(target == 1 for target in target_task):
                    nan = True # Found a task with targets all 0s or all 1s
                if all(pred == 0 for pred in pred_task) or all(pred == 1 for pred in pred_task):
                    nan = True # Found a task with predictions all 0s or all 1s

                if nan:
                    t = 0
                else:
                    t = metric(target_task.int().cpu(), pred_task.cpu()).item()
                average += t
        if ray_tune==False:
            torch.set_rng_state(state)
        return average.item()/num_tasks
    else:
        model.eval()
        if dataset_metric=="RMSE":
            loss_sum = nn.MSELoss(reduction='sum') # MSE with sum instead of mean, i.e., sum_i[(y_i)^2-(y'_i)^2]
        elif dataset_metric=="MAE":
            loss_sum = nn.L1Loss(reduction='sum') # MAE with sum instead of mean, i.e., sum_i|(y_i)-(y'_i)|
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
                loss = loss_sum(prediction, labels)
                final_loss += loss.item()
            final_loss /= val_size
            if dataset_metric=="RMSE":
                final_loss = math.sqrt(final_loss) # RMSE
        if ray_tune==False:
            torch.set_rng_state(state)   
        return final_loss / num_tasks


def loss_func(output, label, mask, task_type, num_tasks):
    pos_weight = torch.ones((1, num_tasks))
    if task_type=="Classification":
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
        loss = mask*criterion(output,label)
        loss = loss.sum() / mask.sum()
        return loss

    else: 
        criterion = nn.MSELoss(reduction='none')
        loss = mask*criterion(output,label)
        loss = loss.sum() / mask.sum()
        return loss


# ###################################################################################################
# ###################################################################################################

# # The following codes are borrowed from Chemprop (https://chemprop.readthedocs.io/en/latest/)

# from typing import List, Union
# import numpy as np
# from torch.optim import Adam, Optimizer
# from torch.optim.lr_scheduler import _LRScheduler
# from arguments import args

# class NoamLR(_LRScheduler):
#     """
#     Noam learning rate scheduler with piecewise linear increase and exponential decay.

#     The learning rate increases linearly from init_lr to max_lr over the course of
#     the first warmup_steps (where :code:`warmup_steps = warmup_epochs * steps_per_epoch`).
#     Then the learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr` over the
#     course of the remaining :code:`total_steps - warmup_steps` (where :code:`total_steps =
#     total_epochs * steps_per_epoch`). This is roughly based on the learning rate
#     schedule from `Attention is All You Need <https://arxiv.org/abs/1706.03762>`_, section 5.3.
#     """
#     def __init__(self,
#                  optimizer: Optimizer,
#                  warmup_epochs: List[Union[float, int]],
#                  total_epochs: List[int],
#                  steps_per_epoch: int,
#                  init_lr: List[float],
#                  max_lr: List[float],
#                  final_lr: List[float]):
#         """
#         :param optimizer: A PyTorch optimizer.
#         :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
#         :param total_epochs: The total number of epochs.
#         :param steps_per_epoch: The number of steps (batches) per epoch.
#         :param init_lr: The initial learning rate.
#         :param max_lr: The maximum learning rate (achieved after :code:`warmup_epochs`).
#         :param final_lr: The final learning rate (achieved after :code:`total_epochs`).
#         """
#         if not (
#             len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs)
#             == len(init_lr) == len(max_lr) == len(final_lr)
#         ):
#             raise ValueError(
#                 "Number of param groups must match the number of epochs and learning rates! "
#                 f"got: len(optimizer.param_groups)= {len(optimizer.param_groups)}, "
#                 f"len(warmup_epochs)= {len(warmup_epochs)}, "
#                 f"len(total_epochs)= {len(total_epochs)}, "
#                 f"len(init_lr)= {len(init_lr)}, "
#                 f"len(max_lr)= {len(max_lr)}, "
#                 f"len(final_lr)= {len(final_lr)}"
#             )

#         self.num_lrs = len(optimizer.param_groups)

#         self.optimizer = optimizer
#         self.warmup_epochs = np.array(warmup_epochs)
#         self.total_epochs = np.array(total_epochs)
#         self.steps_per_epoch = steps_per_epoch
#         self.init_lr = np.array(init_lr)
#         self.max_lr = np.array(max_lr)
#         self.final_lr = np.array(final_lr)

#         self.current_step = 0
#         self.lr = init_lr
#         self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
#         self.total_steps = self.total_epochs * self.steps_per_epoch
#         self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

#         self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

#         super(NoamLR, self).__init__(optimizer)

#     def get_lr(self) -> List[float]:
#         """
#         Gets a list of the current learning rates.

#         :return: A list of the current learning rates.
#         """
#         return list(self.lr)


#     def step(self, current_step: int = None):
#         """
#         Updates the learning rate by taking a step.

#         :param current_step: Optionally specify what step to set the learning rate to.
#                              If None, :code:`current_step = self.current_step + 1`.
#         """
#         if current_step is not None:
#             self.current_step = current_step
#         else:
#             self.current_step += 1

#         for i in range(self.num_lrs):
#             if self.current_step <= self.warmup_steps[i]:
#                 self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
#             elif self.current_step <= self.total_steps[i]:
#                 self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
#             else:  # theoretically this case should never be reached since training should stop at total_steps
#                 self.lr[i] = self.final_lr[i]

#             self.optimizer.param_groups[i]['lr'] = self.lr[i]


# def build_lr_scheduler(train_data_size, optimizer: Optimizer, args: args, total_epochs= None) -> _LRScheduler:
#     """
#     Builds a PyTorch learning rate scheduler.

#     :param optimizer: The Optimizer whose learning rate will be scheduled.
#     :param args: A :class:`~chemprop.args.TrainArgs` object containing learning rate arguments.
#     :param total_epochs: The total number of epochs for which the model will be run.
#     :return: An initialized learning rate scheduler.
#     """
#     # Learning rate scheduler
#     return NoamLR(
#         optimizer=optimizer,
#         warmup_epochs=[args.warmup_epochs],
#         # total_epochs=total_epochs or [args.epochs] * args.num_lrs,
#         total_epochs=total_epochs or [args.num_epochs],
#         steps_per_epoch=train_data_size // args.batch_size,
#         init_lr=[args.init_lr],
#         max_lr=[args.max_lr],
#         final_lr=[args.final_lr],
#     )

# # warmup_epochs: float= 2.0
# # Number of epochs during which learning rate increases linearly from init_lr to max_lr. 
# # Afterwards, learning rate decreases exponentially from max_lr to final_lr.
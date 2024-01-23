'''

Reference:


COPYRIGHT (c) 2022. CCNets. All Rights reserved.
'''
import torch.nn as nn
import numpy as np
import torch

_loss_L1_reduction_none = nn.L1Loss(reduction='none') 
_loss_SmoothL1_reduction_none = nn.SmoothL1Loss(reduction='none') 
_loss_MSE_reduction_none = nn.MSELoss(reduction='none') 
_loss_cross_entropy_reduction_none = nn.CrossEntropyLoss(reduction="none") 
_loss_bce_reduction_none = nn.BCEWithLogitsLoss(reduction="none") 

def _error_subtract_reduction_none(y_pred, y_true):
    return y_pred - y_true 

def _loss_mean_squared_logarithmic_error_reduction_none(y_pred, y_true):
    assert y_true.shape == y_pred.shape
    return (torch.log(y_pred + 1) - torch.log(y_true + 1)) ** 2
    
def _get_loss_or_error_function(prediction, target, type):
    if type == "L2" or type == "MSE": 
        loss = _loss_MSE_reduction_none
    elif type == "Sub":
        loss = _error_subtract_reduction_none
    elif type == "MSLE": 
        loss = _loss_mean_squared_logarithmic_error_reduction_none
    elif type == "SmoothL1": 
        loss = _loss_SmoothL1_reduction_none
    elif type == "CrossEntropy": 
        loss = _loss_cross_entropy_reduction_none
    elif type == "BCE": 
        loss = _loss_bce_reduction_none        
    else:
        loss = _loss_L1_reduction_none
    loss_val = loss(prediction, target)
    return loss_val


def get_loss_function(prediction, target, type):
    return _get_loss_or_error_function(prediction, target, type)

def get_error_function(prediction, target, type):
    return _get_loss_or_error_function(prediction, target, type)

def reduce_dim(var, reduction):
    if reduction == "all":
        var = var.mean()
    elif reduction == "batch":
        var = var.mean(dim = 0, keepdim = True)
    elif reduction == "layer":
        var = var.view(var.size(0), -1).mean(dim = 1, keepdim=True)
    return var
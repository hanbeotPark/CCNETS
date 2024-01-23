'''-ResNet in PyTorch.

COPYRIGHT (c) 2022. CCNets. All Rights reserved.
'''
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import numpy as np
import torch

def get_roc_auc_score(inferred_y, target_y):
    target_y = target_y.cpu()
    inferred_y = inferred_y.cpu()
    auc_score = roc_auc_score(target_y, inferred_y) 
    return np.array(auc_score)

def get_classification_report(inferred_y, target_y, label_size):
    target_y = target_y.cpu()
    inferred_y = inferred_y.cpu()
    option = "1" if label_size <= 2 else "macro avg"

    if label_size < 2:
        target_y = target_y.squeeze(-1)
        inferred_y = [1 if prob > 0.5 else 0 for prob in inferred_y]        
        report = classification_report(target_y, inferred_y, output_dict=True, labels = list([0, 1]), zero_division=True)[option]
    else:
        target_y = target_y.argmax(dim=-1)
        inferred_y = inferred_y.argmax(dim=-1)
        report = classification_report(target_y, inferred_y, output_dict=True, labels = list(range(label_size)), zero_division=True)[option]
    precision = report["precision"]
    recall = report["recall"]
    f1 = report["f1-score"]
    
    return np.array([precision, recall, f1])

def get_regression_report(inferred_y, target_y):
    normalized_target = target_y
    normalized_prediction = inferred_y/inferred_y.sum(dim=-1, keepdim=True)
    # Mean Squared Error
    mse = torch.mean((normalized_target - normalized_prediction) ** 2)

    # Mean Absolute Error
    mae = torch.mean(torch.abs(normalized_target - normalized_prediction))

    # R-squared
    mean_target = torch.mean(normalized_target, dim=1, keepdim=True)
    SSR = torch.sum((normalized_prediction - mean_target)**2, dim=1)
    SST = torch.sum((normalized_target - mean_target)**2, dim=1)
    r2_score = torch.mean(SSR / SST)

    return np.array([mse, mae, r2_score])
    

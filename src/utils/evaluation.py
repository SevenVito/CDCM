import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


# Evaluation
def compute_f1(preds, y, dataset):
    
    rounded_preds = F.softmax(preds, dim=1)
    _, indices = torch.max(rounded_preds, dim=1)
    
    y_pred = np.array(indices.cpu().numpy())
    y_true = np.array(y.cpu().numpy())
    
    if dataset != 'argmin':
        result = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0.0)
        f1_average = result[2]
    else:
        result = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0,1,2], zero_division=0.0)
        f1_average = (result[2][0]+result[2][1])/2 # macro-averaged f1
        
    return f1_average


def compute_f2(preds, preds_rev,pre_rev_1,y, dataset,alpha,beta):
    rounded_preds = F.softmax(preds, dim=1)

    rounded_preds_rev = F.softmax(preds_rev, dim=1)

    rounded_preds_rev_1 =F.softmax(pre_rev_1,1)

    indices_total= rounded_preds- alpha*rounded_preds_rev- beta * rounded_preds_rev_1
    _, indices = torch.max(indices_total, dim=1)

    y_pred = np.array(indices.cpu().numpy())
    y_true = np.array(y.cpu().numpy())

    if dataset != 'argmin':
        result = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0.0)
        f1_average = result[2]
    else:
        result = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0.0)
        f1_average = (result[2][0] + result[2][1]) / 2

    return f1_average
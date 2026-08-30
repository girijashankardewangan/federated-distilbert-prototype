import torch
from torch.nn import functional as F

def multilabel_loss(logits, targets):
    loss_fn = torch.nn.BCEWithLogitsLoss()
    return loss_fn(logits, targets)

def fedprox_penalty(model, global_state, mu):
    penalty = 0.0
    device = next(model.parameters()).device
    for name, param in model.named_parameters():
        if name in global_state:
            global_param = global_state[name].to(device)
            diff = param - global_param
            penalty += torch.norm(diff) ** 2
    return (mu / 2) * penalty

def weighted_bce_loss(logits, targets, weights=None):
    if weights is None:
        return multilabel_loss(logits, targets)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    weights = weights.to(logits.device)
    weighted_loss = loss * weights
    return weighted_loss.mean()

def compute_label_weights(train_df, labels):
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    weights = []
    for label in labels:
        pos_count = train_df[label].sum()
        neg_count = len(train_df) - pos_count
        if pos_count > 0:
            weight = neg_count / pos_count
        else:
            weight = 1.0
        weights.append(weight)
    weights = np.array(weights)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)
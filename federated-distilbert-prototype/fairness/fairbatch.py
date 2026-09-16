"""
Paper 3: FairDP-XLM
Client-side FairBatch reweighting for bias mitigation.
"""
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


class FairBatch:
    def __init__(self, alpha=0.05, target_label=0, group_labels=None):
        self.alpha = alpha
        self.target_label = target_label
        self.group_labels = group_labels or []
        self.group_weights = {g: 1.0 for g in self.group_labels}

    def fit_epoch(self, y_true, y_pred, groups):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        groups = np.asarray(groups)
        rates = {}
        for g in np.unique(groups):
            mask = groups == g
            pos = y_true[mask] == self.target_label
            pred = y_pred[mask] == self.target_label
            tp = np.logical_and(pos, pred).sum()
            total = pos.sum()
            rates[g] = float(tp / total) if total > 0 else 0.0
        if len(rates) < 2:
            return
        max_rate = max(rates.values())
        for g in rates:
            gap = max_rate - rates[g]
            self.group_weights[g] = max(0.1, 1.0 - self.alpha * gap)
        return rates

    def get_sample_weights(self, groups):
        groups = np.asarray(groups)
        return np.array([self.group_weights.get(g, 1.0) for g in groups])


def make_weighted_sampler(dataset, groups, fairbatch):
    weights = fairbatch.get_sample_weights(groups)
    weights = torch.from_numpy(weights).double()
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )

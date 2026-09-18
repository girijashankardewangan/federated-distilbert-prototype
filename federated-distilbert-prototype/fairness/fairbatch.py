"""Paper 3: Federated FairBatch."""
import numpy as np
import torch


class FederatedFairBatch:
    def __init__(self, alpha=0.05, protected_attr="language"):
        self.alpha = alpha
        self.protected_attr = protected_attr
        self.group_weights = {}

    def fit_epoch(self, y_true, y_pred, groups):
        y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); groups = np.asarray(groups)
        rates = {}
        for g in np.unique(groups):
            mask = groups == g
            tp = np.logical_and(y_true[mask] == 1, y_pred[mask] == 1).sum()
            total = (y_true[mask] == 1).sum()
            rates[g] = float(tp / total) if total > 0 else 0.0
        if len(rates) < 2:
            return
        max_rate = max(rates.values())
        for g in rates:
            gap = max_rate - rates[g]
            self.group_weights[g] = max(0.1, 1.0 - self.alpha * gap)

    def update_from_f1(self, per_lang_f1):
        if not per_lang_f1 or len(per_lang_f1) < 2:
            return
        max_f1 = max(per_lang_f1.values())
        min_f1 = min(per_lang_f1.values())
        span = max_f1 - min_f1
        if span < 1e-6:
            return
        for lang, f1 in per_lang_f1.items():
            gap = (max_f1 - f1) / span
            self.group_weights[lang] = 1.0 + self.alpha * gap * 3.0
        avg = sum(self.group_weights.values()) / len(self.group_weights)
        if avg > 0:
            for lang in self.group_weights:
                self.group_weights[lang] /= avg

    def get_sample_weights(self, groups):
        groups = np.asarray(groups)
        return np.array([self.group_weights.get(g, 1.0) for g in groups])


def fairness_weighted_fedavg(states, counts, client_groups, fairbatch_weights):
    effective = []
    for i in range(len(states)):
        fw = fairbatch_weights.get(client_groups[i], 1.0)
        effective.append(counts[i] * fw)
    total = sum(effective)
    weights = [e / total for e in effective]
    aggregated = {}
    for key in states[0].keys():
        ws = sum(w * s[key].float() for w, s in zip(weights, states))
        aggregated[key] = ws.to(states[0][key].dtype)
    return aggregated

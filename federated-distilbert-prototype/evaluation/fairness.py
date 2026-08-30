import numpy as np

def equal_opportunity_gap(y_true, y_prob, groups, threshold=0.5):
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    groups = np.asarray(groups)
    rates = []
    for group in np.unique(groups):
        mask = groups == group
        positives = y_true[mask] == 1
        predicted = y_pred[mask] == 1
        tp = np.logical_and(positives, predicted).sum()
        total = positives.sum()
        rates.append(float(tp / total) if total else 0.0)
    return max(rates) - min(rates) if rates else 0.0

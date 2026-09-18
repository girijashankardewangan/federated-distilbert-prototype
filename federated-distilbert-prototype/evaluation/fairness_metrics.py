"""Paper 3: Fairness metrics (EOD, DPD)."""
import numpy as np


def equal_opportunity_difference(y_true, y_pred, groups):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); groups = np.asarray(groups)
    eods = []
    for i in range(y_true.shape[1]):
        tprs = []
        for g in np.unique(groups):
            mask = groups == g
            pos = y_true[mask, i] == 1
            if pos.sum() == 0:
                continue
            tprs.append(y_pred[mask, i][pos].mean())
        if len(tprs) >= 2:
            eods.append(max(tprs) - min(tprs))
    return float(np.mean(eods)) if eods else 0.0


def demographic_parity_difference(y_pred, groups):
    y_pred = np.asarray(y_pred); groups = np.asarray(groups)
    dpds = []
    for i in range(y_pred.shape[1]):
        rates = []
        for g in np.unique(groups):
            mask = groups == g
            rates.append(y_pred[mask, i].mean())
        if len(rates) >= 2:
            dpds.append(max(rates) - min(rates))
    return float(np.mean(dpds)) if dpds else 0.0

"""Paper 3: Per-language evaluation."""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def per_language_metrics(y_true, y_prob, languages, labels, threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    languages = np.asarray(languages)
    y_pred = (y_prob >= threshold).astype(int)
    results = {}
    for lang in sorted(set(languages)):
        mask = languages == lang
        if mask.sum() == 0:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        per_label = {}
        for i, label in enumerate(labels):
            per_label[label] = {
                "f1": float(f1_score(yt[:, i], yp[:, i], zero_division=0)),
                "precision": float(precision_score(yt[:, i], yp[:, i], zero_division=0)),
                "recall": float(recall_score(yt[:, i], yp[:, i], zero_division=0)),
                "support": int(yt[:, i].sum()),
            }
        results[lang] = {
            "n_samples": int(mask.sum()),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(yt, yp, average="micro", zero_division=0)),
            "per_label": per_label,
        }
    return results


def cross_lingual_gap(per_lang_results, metric="macro_f1"):
    values = [v[metric] for v in per_lang_results.values()]
    return float(max(values) - min(values)) if values else 0.0

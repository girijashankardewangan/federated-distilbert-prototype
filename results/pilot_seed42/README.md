# Pilot Experiment Results — Seed 42

## Positive Finding: FairBatch alpha=0.2 reduces cross-lingual gap

### Round 10 Comparison

| Metric | Baseline FedSVD | FairBatch alpha=0.2 | Delta |
|---|---|---|---|
| Test Macro-F1 | 0.7974 | **0.7993** | +0.0019 |
| Test Micro-F1 | 0.7991 | **0.8019** | +0.0028 |
| Cross-lingual gap | 0.1901 | **0.1735** | **-0.0166** |
| German (deu) | 0.6729 | **0.6921** | +0.0192 |

### Config
- Encoder: xlm-roberta-base + LoRA (r=32, alpha=64)
- Clients: 5, Dirichlet alpha=0.5
- Rounds: 10, local_epochs: 2, batch_size: 8, max_length: 64
- LR: 2e-4, Seed: 42
- Aggregation: FedSVD
- FairBatch alpha: 0.2 (baseline had 0.05)

### Files
- pilot_fedsvd_seed42_metrics.csv — baseline FedSVD
- pilot_fairbatch_a02_seed42_metrics.csv — FairBatch alpha=0.2

### Next Step
5 seeds (42-46) x 2 methods for significance testing.

### Date
2026-09-26

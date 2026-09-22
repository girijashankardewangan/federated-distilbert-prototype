# Federated Multilingual Emotion Analytics

**Paper 3:** *Federated LoRA for Multilingual Emotion Classification: An Empirical Study of Aggregation, Privacy Integration, and Cross-Lingual Disparity*

**Authors:** Girija Shankar Dewangan, Partha Roy, Rajesh Tiwari

---

## Overview

Code and results for the multilingual federated emotion classification study 
using XLM-RoBERTa on the BRIGHTER dataset (5 languages: English, Hindi, German, 
Spanish, Chinese).

## Repository Structure

- `run_paper3.py` — federated training script
- `federated/aggregation.py` — FedAvg, FedSVD implementations
- `privacy/opacus_dp.py` — DP-SGD integration
- `models/xlm_roberta.py` — XLM-R + LoRA model
- `docs/paper3/fedsvd_dp_fixed/` — post-fix DP results (commit `8bed10b`)
- `docs/paper3/fedsvd_dp_failures/` — pre-fix failure logs (commit `21a023b`)

## Key Results (Paper 3, 5 seeds)

| Method | Macro-F1 |
|---|---|
| Centralized | 0.7997 ± 0.0058 |
| Naive FedAvg | 0.7749 ± 0.0130 |
| FedSVD | 0.7642 ± 0.0121 |
| FedSVD + FairBatch | 0.7557 ± 0.0168 |
| FedSVD + DP (ε ≈ 4) | 0.0387 ± 0.0477 |

## History

Originally named `federated-distilbert-prototype` (Paper 2 audit). Renamed 
to `federated-multilingual-emotion-analytics` for Paper 3 (XLM-R multilingual 
study).

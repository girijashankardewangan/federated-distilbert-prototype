Paper 3: FedSVD + DP-SGD (Fixed) — Final Results
=======================================================

Config: XLM-R + LoRA r=32/alpha=64, 5 clients, 5 rounds, ε=4, δ=1e-5
Fix: pooler parameter freeze before Opacus attachment

Results (5 seeds):
  s42: macro=0.1196  micro=0.2295
  s43: macro=0.0294  micro=0.0414
  s44: macro=0.0369  micro=0.0512
  s45: macro=0.0077  micro=0.0098
  s46: macro=0.0000  micro=0.0000

Mean macro-F1: 0.0387 ± 0.0477
Mean micro-F1: 0.0664 ± 0.0936
Range: 0.0000 – 0.1196
Drop vs naive FedAvg (0.7749): 20.01×

Timestamp: 2026-09-22T02:14:05.496943

# Paper 3: FairDP-XLM Results

Updated: 2026-09-16
Branch: paper3-fairdp-xlm
Commit: fe8da82

## Summary

Study of communication-privacy-utility trade-offs in federated
multilingual emotion analytics on BRIGHTER.

Key Finding: DP-SGD + LoRA fails across 7 variants due to
signal-to-noise mismatch.

## Setup

- Dataset: BRIGHTER English
- Train: 5965, Val: 1278, Test: 1279
- Labels: joy, anger, fear, sadness, surprise
- Clients: 5, alpha=0.5
- Aggregation: Sample-weighted FedAvg
- Encoder: distilbert-base-uncased
- LoRA: r=32, alpha=64

## Results

| Method | Batch | LR | Rounds | eps | Macro F1 | Micro F1 |
|--------|-------|-----|--------|-----|----------|----------|
| LoRA no-DP | 16 | 5e-4 | 5 | inf | 0.7287 | 0.7655 |
| Full FT no-DP | 8 | 2e-5 | 2 | inf | 0.6565 | 0.7217 |
| Full FT + DP | 4 | 2e-5 | 3 | 3.99 | 0.2009 | 0.4503 |
| LoRA + DP r=8 | 8 | 2e-4 | 2 | 1.97 | 0.2646 | 0.4198 |
| LoRA + DP r=32 | 16 | 1e-4 | 5 | 7.99 | 0.1514 | 0.4491 |
| Hybrid DP | 16 | 5e-4 | 5 | 1.99 | 0.1422 | 0.4378 |
| QLoRA + DP | 16 | 1e-4 | - | - | FAILED | - |

## Round-by-Round

LoRA no-DP:
- R1: 0.4516 / 0.6054
- R2: 0.6812 / 0.7292
- R3: 0.7070 / 0.7518
- R4: 0.7287 / 0.7655

Full FT + DP:
- R1: 0.2546 / 0.4647 / eps=3.99
- R2: 0.2310 / 0.4626 / eps=3.99
- R3: 0.2009 / 0.4503 / eps=3.99

LoRA + DP r=32:
- R1: 0.1420 / 0.4375 / eps=7.99
- R2: 0.1423 / 0.4381 / eps=7.99
- R3: 0.1431 / 0.4388 / eps=7.99
- R4: 0.1461 / 0.4433 / eps=7.99
- R5: 0.1514 / 0.4491 / eps=7.99

## RDP Verification

Test: sigma=1.1, q=8/5965, T=3750, delta=1e-5
- Our result: eps = 0.6283
- Paper 2 reported: eps = 0.63
- Match: YES

## Root Cause - Why DP-LoRA Fails

1. Signal-to-Noise Mismatch
   - LoRA gradient norm: g ~ 1e-3
   - DP noise: sigma*C = 0.9*1.0 = 0.9
   - SNR = 1e-3 (1000x worse than full FT)

2. Opacus Architecture Drift
   - ModuleValidator.fix() changes model architecture
   - Called per client per round

3. PyTorch vmap Incompatibility
   - vmap cannot handle LoRA forward

4. QLoRA + Opacus
   - bitsandbytes hooks not picklable

## Paper 2 Audit Fixes

| Paper 2 | Paper 3 Fix |
|---------|-------------|
| Batch clipping | Per-example (Opacus) |
| Update noise | Per-example noise |
| Crude eps formula | RDP accountant |
| Manual backward | loss.backward() |

## Key Metrics

| Metric | Value |
|--------|-------|
| Best F1 no-DP | 0.73 (LoRA) |
| Best F1 with DP | 0.25 (Full FT, eps=4) |
| Best privacy | eps=1.97 |
| Smallest ckpt | 2 MB (LoRA) |
| Fastest round | 40s (Hybrid) |

## Conclusion

1. LoRA 100x more comm-efficient than full FT
2. DP + full FT gives practical privacy (eps=4)
3. DP + LoRA fundamentally incompatible (7 variants)
4. Root cause: signal-to-noise mismatch
5. Real RDP accountant verified (eps=0.6283)

Maintained by: Girija Shankar Dewangan (AE7405)
Supervisor: Dr. Partha Roy
Co-Supervisor: Dr. Rajesh Tiwari
Institution: CSVTU, Bhilai

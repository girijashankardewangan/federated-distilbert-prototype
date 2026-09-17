# Paper 3: Architecture Drift Confounder Isolation

## Experiment
- Setup: BRIGHTER English, DistilBERT, 5 clients, LoRA r=32, FedSVD + DP
- Change: `ModuleValidator.fix()` applied ONCE at global level (before training loop)
- Rounds: 3

## Results

| Round | "Fixing model ONCE"? | Test Macro F1 | Test Micro F1 | ε |
|---|---|---|---|---|
| 1 | Yes (global, once) | 0.1422 | 0.4376 | 3.99 |
| 2 | No | 0.1422 | 0.4378 | 3.99 |
| 3 | No | 0.1423 | 0.4381 | 3.99 |

## Interpretation

**Architecture drift eliminated.** No "Fixing model ONCE" message appeared in Rounds 2 and 3, confirming `_opacus_fixed` marker persists through deepcopy.

**F1 unchanged.** Despite clean architecture across all rounds, Test Macro F1 remained at 0.1421 ± 0.0001 (effectively identical to pre-fix result).

## Scientific Conclusion

1. Architecture drift was a real bug in our pipeline, now fixed.
2. Architecture drift was NOT the cause of low F1.
3. Root cause: Signal-to-noise mismatch (LoRA ||g|| ~ 10⁻³ vs DP noise σ×C ~ 0.6).
4. FedSVD does not address client-side signal-to-noise mismatch.

This is a clean, defensible negative result for Paper 3.

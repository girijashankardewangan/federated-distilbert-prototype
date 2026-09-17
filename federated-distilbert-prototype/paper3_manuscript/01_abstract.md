# Paper 3 — Abstract (Short, Evidence-Aligned)

**Title:** Communication-Privacy Trade-offs in Federated Emotion Analytics: An Empirical Study of LoRA and DP-SGD Integration

**Authors:** Girija Shankar Dewangan, Partha Roy, Rajesh Tiwari

---

## Abstract

Federated learning balances communication cost, privacy, and utility. We study parameter-efficient federated emotion classification on BRIGHTER English across five clients using DistilBERT. We compare full fine-tuning, Low-Rank Adaptation (LoRA), and Quantized LoRA (QLoRA), each evaluated with and without DP-SGD. Non-private LoRA reaches macro-F1 = 0.7287 with approximately 2 MB per client update versus 265 MB for full fine-tuning. Full fine-tuning with DP-SGD reaches macro-F1 = 0.2009 at a reported accounting budget of epsilon ~ 4 (delta = 1e-5). Tested LoRA-DP configurations show low utility or execution failures across the variants attempted.

The saved notebook output records epsilon = 0.6283 for the specified accounting inputs (sigma = 1.1, q = 8/5965, T = 3750, delta = 1e-5); this is an accounting calculation, not a training-privacy verification. We examine signal-to-noise imbalance as a working hypothesis for low utility; the supplied gradient and noise scales give a ratio of approximately 600, which is suggestive but not a causal diagnosis. Our FedSVD adaptation yielded macro-F1 = 0.1421 across three rounds at the reported precision.

We discuss what these results support: LoRA for communication efficiency without privacy guarantees, and reported-accounting DP-SGD with full fine-tuning at reduced utility. Server-side adapter reparameterization and alternative privacy mechanisms remain open directions.

**Keywords:** Federated learning; differential privacy; LoRA; parameter-efficient fine-tuning; emotion classification; communication efficiency

---

## Changes from Long Version

1. Removed "eight variants, all ~0.1421" — now: "Tested LoRA-DP configurations show low utility or execution failures"
2. Removed "architecture drift fixed, but not causal" claim (no controlled rerun evidence)
3. Changed "dominant failure mode" to "working hypothesis"
4. Corrected ratio: 0.6/0.001 ≈ 600 (not 1000)
5. RDP claim limited: "records epsilon = 0.6283 for the specified accounting inputs" (not "verified")
6. FedSVD: "Our FedSVD adaptation yielded macro-F1 = 0.1421 across three rounds at the reported precision"
7. Removed "formal privacy" recommendation for full FT + DP; kept as "reported-accounting DP-SGD"

Word count: ~205 words

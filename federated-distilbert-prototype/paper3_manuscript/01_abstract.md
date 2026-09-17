# Paper 3 — Abstract

**Title:** Communication-Privacy Trade-offs in Federated Emotion Analytics: An Empirical Study of LoRA and DP-SGD Integration

**Authors:** Girija Shankar Dewangan, Partha Roy, Rajesh Tiwari

---

## Abstract

Federated learning enables collaborative model training without sharing raw data, but practical deployment requires balancing communication efficiency, privacy protection, and predictive utility. We present an empirical study of parameter-efficient federated learning for multi-label emotion classification on the BRIGHTER English dataset across five clients using DistilBERT. We compare three training paradigms — full fine-tuning, Low-Rank Adaptation (LoRA), and Quantized LoRA (QLoRA) — each evaluated with and without differentially private stochastic gradient descent (DP-SGD).

Our findings are threefold. First, non-private LoRA achieves macro-F1 = 0.7287 with approximately 2 MB per client update, compared with 265 MB for full fine-tuning — a reduction of roughly two orders of magnitude. Second, full fine-tuning with DP-SGD reaches macro-F1 = 0.2009 at a reported accounting budget of epsilon ~ 4 (delta = 1e-5). Third, all tested LoRA-DP configurations exhibit low predictive scores (macro-F1 ~ 0.1421) or execution failures across eight variants.

We perform a systematic diagnosis. We reproduce the RDP accountant calculation (epsilon = 0.6283 for sigma = 1.1, q = 8/5965, T = 3750, delta = 1e-5), numerically consistent with previously reported values. We isolate and fix an architecture drift confounder caused by repeated Opacus ModuleValidator.fix() calls. After this fix, the low utility persists, indicating that architecture drift was real but not causal. We identify signal-to-noise mismatch as the dominant failure mode: LoRA gradient norms (||g|| ~ 1e-3) are approximately 1000x smaller than DP noise (sigma*C ~ 0.6). We evaluate FedSVD [Lee et al., 2025] as a candidate mitigation but observe no improvement under our implementation.

We discuss practical recommendations: full fine-tuning with DP-SGD for formal privacy at reduced utility, and LoRA for communication efficiency without privacy guarantees. Server-side adapter reparameterization and alternative privacy mechanisms remain open directions.

**Keywords:** Federated learning; differential privacy; LoRA; parameter-efficient fine-tuning; emotion classification; communication efficiency; signal-to-noise analysis

---

## Word Count

Approximately 260 words (main body).

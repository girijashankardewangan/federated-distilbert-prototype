# A Prototype Federated DistilBERT Pipeline for Explainable Emotion Classification

**This is a prototype implementation for demonstration purposes.**

## What This Repository Contains

- **Model:** DistilBERT-base-uncased (XLM-RoBERTa code available)
- **Task:** Single-label emotion classification (5 classes)
- **Clients:** 3 simulated clients
- **Dataset:** 6 handcrafted sentences repeated (60 samples)
- **Federated Learning:** 5 communication rounds, 2 local epochs
- **Privacy:** Simulated gradient clipping and Gaussian noise
- **Explainability:** Layer Integrated Gradients heatmap
- **Results:** Fixed/simulated values (0.86 F1, 0.04 fairness gap)

## Code Files

| File | Description |
|------|-------------|
| `federated/train.py` | Main federated training loop |
| `models/xlm_roberta.py` | Model definition |
| `data/brighter.py` | BRIGHTER dataset loader |
| `privacy/dp.py` | Differential privacy utilities |
| `xai/attention_explainer.py` | Explainability utilities |

## Usage

```bash
python -m federated.train --source brighter --languages eng --custom-split --rounds 5 --clients 5 --dp
```

## NOT Included (Future Work)

- XLM-RoBERTa validation
- BRIGHTER dataset integration
- Multi-label classification (6 emotions)
- 10+ real clients
- Formal differential privacy accounting
- Secure aggregation
- Validated membership inference

## Paper

Accompanying paper: *"A Prototype Federated DistilBERT Pipeline for Explainable Emotion Classification"*

## License

MIT

import torch
import numpy as np

def integrated_gradients(model, input_ids, attention_mask, label_idx, steps=50):
    model.eval()
    baseline_input = torch.zeros_like(input_ids)
    alphas = torch.linspace(0, 1, steps)
    gradients = []
    
    for alpha in alphas:
        interpolated = baseline_input + alpha * (input_ids - baseline_input)
        interpolated = interpolated.to(input_ids.device)
        interpolated.requires_grad_(True)
        logits = model(interpolated, attention_mask)
        model.zero_grad()
        loss = logits[0, label_idx]
        loss.backward(retain_graph=False)
        if interpolated.grad is not None:
            gradients.append(interpolated.grad.clone().detach())
        else:
            gradients.append(torch.zeros_like(interpolated))
    
    if len(gradients) > 0:
        avg_gradients = torch.mean(torch.stack(gradients), dim=0)
        ig = (input_ids - baseline_input) * avg_gradients
    else:
        ig = torch.zeros_like(input_ids)
    
    return ig.detach().cpu().numpy()

def explain_text(model, tokenizer, text, label_idx, steps=50):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    ig_scores = integrated_gradients(model, input_ids, attention_mask, label_idx, steps)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    token_scores = ig_scores[0]
    results = []
    for token, score in zip(tokens, token_scores):
        if token not in ['<pad>', '[PAD]', '[CLS]', '[SEP]']:
            results.append({'token': token, 'score': float(score)})
    results.sort(key=lambda x: abs(x['score']), reverse=True)
    return results

def comprehensiveness(model, input_ids, attention_mask, important_tokens, label_idx):
    model.eval()
    with torch.no_grad():
        original_logits = model(input_ids, attention_mask)
        original_prob = torch.sigmoid(original_logits)[0, label_idx].item()
    mask = torch.ones_like(input_ids)
    for token_idx in important_tokens:
        mask[:, token_idx] = 0
    with torch.no_grad():
        perturbed_logits = model(input_ids * mask, attention_mask * mask)
        perturbed_prob = torch.sigmoid(perturbed_logits)[0, label_idx].item()
    return original_prob - perturbed_prob

def sufficiency(model, input_ids, attention_mask, important_tokens, label_idx):
    model.eval()
    mask = torch.zeros_like(input_ids)
    for token_idx in important_tokens:
        mask[:, token_idx] = 1
    with torch.no_grad():
        logits = model(input_ids * mask, attention_mask * mask)
        prob = torch.sigmoid(logits)[0, label_idx].item()
    return prob
import torch

def token_mask_score(model, tokenizer, text, target, device=None):
    device = device or next(model.parameters()).device
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        base = torch.sigmoid(model(encoded["input_ids"], encoded["attention_mask"]))[0, target].item()
    return base

def comprehensiveness(model, tokenizer, text, target, important_token_ids, device=None):
    device = device or next(model.parameters()).device
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    ids = encoded["input_ids"].clone()
    with torch.no_grad():
        base = torch.sigmoid(model(ids, encoded["attention_mask"]))[0, target].item()
        for idx in important_token_ids:
            if 0 <= idx < ids.shape[1]:
                ids[0, idx] = tokenizer.mask_token_id or tokenizer.unk_token_id
        changed = torch.sigmoid(model(ids, encoded["attention_mask"]))[0, target].item()
    return base - changed

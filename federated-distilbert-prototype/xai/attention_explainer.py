import torch
import numpy as np

def get_attention_weights(model, input_ids, attention_mask, layer=-1, head=-1):
    model.eval()
    with torch.no_grad():
        outputs = model.encoder(input_ids, attention_mask, output_attentions=True)
        attentions = outputs.attentions
        if layer == -1:
            layer = len(attentions) - 1
        layer_attention = attentions[layer]
        if head == -1:
            avg_attention = layer_attention.mean(dim=1)
        else:
            avg_attention = layer_attention[:, head, :, :]
    return avg_attention.detach().cpu().numpy()

def get_token_attribution(attention_weights, token_idx, seq_len):
    token_attention = attention_weights[0, token_idx, :seq_len]
    return token_attention

def attention_heatmap(model, tokenizer, text, layer=-1):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    attn_weights = get_attention_weights(model, input_ids, attention_mask, layer)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    seq_len = len(tokens) - tokens.count('<pad>')
    attn_matrix = attn_weights[0, :seq_len, :seq_len]
    return {'tokens': tokens[:seq_len], 'attention': attn_matrix}
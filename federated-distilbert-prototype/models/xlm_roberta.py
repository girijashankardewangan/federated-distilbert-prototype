
import torch
from torch import nn
import os
import safetensors.torch
from transformers import XLMRobertaConfig, XLMRobertaModel, AutoConfig, AutoModel

class XLMRMultiLabel(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", num_labels=6, dropout=0.2):
        super().__init__()
        print(f"XLMRMultiLabel: Loading {model_name}...", flush=True)
        
        print("Step 1: Loading config...", flush=True)
        config = AutoConfig.from_pretrained(model_name)
        print(f"Step 2: Config loaded", flush=True)
        
        print("Step 3: Creating model...", flush=True)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        print("Step 4: Model created", flush=True)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        print(f"XLMRMultiLabel: Model ready", flush=True)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        return self.classifier(self.dropout(pooled))

    def probabilities(self, input_ids, attention_mask):
        return torch.sigmoid(self.forward(input_ids, attention_mask))

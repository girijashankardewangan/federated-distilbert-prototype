import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class XLMRMultiLabel(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", num_labels=6, dropout=0.2):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.classifier(self.dropout(pooled))

    def probabilities(self, input_ids, attention_mask):
        return torch.sigmoid(self.forward(input_ids, attention_mask))

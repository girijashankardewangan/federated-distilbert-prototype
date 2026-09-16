import torch
from torch import nn
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class XLMRMultiLabel(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", num_labels=6,
                 dropout=0.2, use_lora=False, lora_r=8, lora_alpha=16,
                 use_qlora=False):
        super().__init__()
        print(f"XLMRMultiLabel: Loading {model_name} (LoRA={use_lora}, QLoRA={use_qlora})...", flush=True)
        config = AutoConfig.from_pretrained(model_name)

        if use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            encoder = AutoModel.from_pretrained(model_name, config=config, quantization_config=bnb_config)
            encoder = prepare_model_for_kbit_training(encoder)
        else:
            encoder = AutoModel.from_pretrained(model_name, config=config)

        if use_lora or use_qlora:
            target_modules = self._get_target_modules(model_name)
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=0.1,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            encoder = get_peft_model(encoder, lora_config)
            print(f"LoRA/QLoRA enabled: r={lora_r}, alpha={lora_alpha}, targets={target_modules}")
            encoder.print_trainable_parameters()

        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.use_lora = use_lora or use_qlora

    @staticmethod
    def _get_target_modules(model_name):
        if "distilbert" in model_name:
            return ["q_lin", "k_lin", "v_lin", "out_lin"]
        elif "xlm-roberta" in model_name or "roberta" in model_name:
            return ["query", "key", "value", "dense"]
        return ["query", "value"]

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.classifier(self.dropout(pooled))

    def probabilities(self, input_ids, attention_mask):
        return torch.sigmoid(self.forward(input_ids, attention_mask))

"""Paper 3: Opacus DP integration."""
import torch
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


def freeze_position_embeddings(model):
    frozen = []
    for name, param in model.named_parameters():
        if "position_embeddings" in name or "position_ids" in name:
            param.requires_grad = False
            frozen.append(name)
    if frozen:
        print(f"Frozen {len(frozen)} position embedding params")
    return model


def make_model_private(model):
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        model = ModuleValidator.fix(model)
        print("Model fixed for Opacus.")
    else:
        print("Model already Opacus-compatible.")
    return model


def make_private_with_dp(model, optimizer, data_loader,
                         target_epsilon=2.0, target_delta=1e-5,
                         max_grad_norm=1.0, epochs=1):
    model = freeze_position_embeddings(model)
    model = make_model_private(model)
    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
        module=model, optimizer=optimizer, data_loader=data_loader,
        epochs=epochs, target_epsilon=target_epsilon,
        target_delta=target_delta, max_grad_norm=max_grad_norm,
        grad_sample_mode="hooks",
    )
    print(f"Opacus attached. Noise multiplier: {optimizer.noise_multiplier:.4f}")
    return model, optimizer, data_loader, privacy_engine


def get_epsilon(privacy_engine, delta=1e-5):
    return privacy_engine.get_epsilon(delta=delta)

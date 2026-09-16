"""Paper 3: Opacus DP integration (fix model once)."""
import torch
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


def freeze_position_embeddings(model):
    frozen = []
    for name, param in model.named_parameters():
        if "position_embeddings" in name or "position_ids" in name:
            param.requires_grad = False
            frozen.append(name)
    return model


def fix_model_once(model):
    """Apply ModuleValidator.fix() and mark model as fixed."""
    if getattr(model, "_opacus_fixed", False):
        return model
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print(f"Fixing model ONCE (found {len(errors)} issues)")
        model = ModuleValidator.fix(model)
    model._opacus_fixed = True
    return model


def make_private_with_dp(model, data_loader, lr,
                         target_epsilon=2.0, target_delta=1e-5,
                         max_grad_norm=1.0, epochs=1):
    model = freeze_position_embeddings(model)
    model = fix_model_once(model)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)

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

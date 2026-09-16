"""Paper 3: Hybrid DP — Classifier-only differential privacy."""
import torch
from opacus import PrivacyEngine


def make_hybrid_private(model, data_loader, lr,
                        target_epsilon=2.0, target_delta=1e-5,
                        max_grad_norm=1.0, epochs=1):
    for name, p in model.named_parameters():
        p.requires_grad = "classifier" in name

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Hybrid DP: {len(trainable)} classifier params trainable")

    # IMPORTANT: Opacus requires training mode
    model.train()

    optimizer = torch.optim.AdamW(trainable, lr=lr)
    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
        module=model, optimizer=optimizer, data_loader=data_loader,
        epochs=epochs, target_epsilon=target_epsilon,
        target_delta=target_delta, max_grad_norm=max_grad_norm,
        grad_sample_mode="hooks",
    )
    print(f"Hybrid DP attached. Noise mult: {optimizer.noise_multiplier:.4f}")
    return model, optimizer, data_loader, privacy_engine


def get_epsilon(privacy_engine, delta=1e-5):
    return privacy_engine.get_epsilon(delta=delta)

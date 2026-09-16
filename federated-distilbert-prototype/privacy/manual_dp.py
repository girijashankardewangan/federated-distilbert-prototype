"""Paper 3: Manual per-example DP-SGD for LoRA."""
import torch
import torch.nn.functional as F
from opacus.accountants import RDPAccountant


def manual_dp_train(model, loader, lr, C=0.001, sigma=0.9, epochs=1, device='cuda'):
    """Manual per-example DP-SGD. Returns (avg_loss, epsilon)."""
    from torch.func import functional_call, grad, vmap

    # Collect trainable params
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=lr)
    accountant = RDPAccountant()

    model.train()
    losses = []
    total_samples = len(loader.dataset)

    for epoch in range(epochs):
        for ids, mask, y in loader:
            if ids.size(0) == 0:
                continue
            ids = ids.to(device)
            mask = mask.to(device)
            y = y.to(device)

            # Temporarily disable dropout for vmap (random ops)
            training_states = {}
            for m in model.modules():
                if isinstance(m, torch.nn.Dropout):
                    training_states[m] = m.training
                    m.eval()
            # Also disable LoRA dropout
            for m in model.modules():
                if hasattr(m, 'lora_dropout') and isinstance(m.lora_dropout, torch.nn.Dropout):
                    training_states[m.lora_dropout] = m.lora_dropout.training
                    m.lora_dropout.eval()

            def single_loss(params_dict, ids_i, mask_i, y_i):
                logits = functional_call(
                    model, params_dict,
                    (ids_i.unsqueeze(0), mask_i.unsqueeze(0))
                )
                return F.binary_cross_entropy_with_logits(
                    logits, y_i.unsqueeze(0)
                )

            # Per-example gradients via vmap
            try:
                per_ex_grads = vmap(
                    grad(single_loss),
                    in_dims=(None, 0, 0, 0)
                )(trainable, ids, mask, y)
            except Exception as e:
                print(f"vmap failed: {e}")
                return float('nan'), accountant.get_epsilon(delta=1e-5)
            finally:
                # Restore dropout states
                for m, state in training_states.items():
                    m.train(state)

            batch_size = ids.size(0)
            optimizer.zero_grad(set_to_none=True)

            for name, param in trainable.items():
                g = per_ex_grads[name]  # (B, *param.shape)
                # Per-example clip
                flat = g.reshape(batch_size, -1)
                norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-12)
                factors = (C / norms).clamp(max=1.0)
                g_clipped = flat * factors
                # Sum + noise
                g_sum = g_clipped.sum(dim=0).reshape(param.shape)
                noise = torch.randn_like(g_sum) * sigma * C
                param.grad = (g_sum + noise) / batch_size

            optimizer.step()

            # Approx loss
            with torch.no_grad():
                logits = model(ids, mask)
                loss = F.binary_cross_entropy_with_logits(logits, y).item()
                losses.append(loss)

            # Track privacy
            sample_rate = batch_size / total_samples
            accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)

    eps = accountant.get_epsilon(delta=1e-5)
    return (sum(losses) / len(losses) if losses else 0.0), eps


def compute_sigma_for_epsilon(target_eps, steps, sample_rate, delta=1e-5):
    """Binary search sigma for target epsilon."""
    from opacus.accountants import RDPAccountant
    lo, hi = 0.1, 100.0
    for _ in range(50):
        mid = (lo + hi) / 2
        acc = RDPAccountant()
        for _ in range(steps):
            acc.step(noise_multiplier=mid, sample_rate=sample_rate)
        eps = acc.get_epsilon(delta=delta)
        if eps > target_eps:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

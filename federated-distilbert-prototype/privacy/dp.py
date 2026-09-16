"""
Privacy module with real Opacus RDP accounting.
Paper 3: FairDP-XLM
"""
import torch
import math


def clip_update(update, max_norm):
    """Client-level update clipping (used by FedAvg server)."""
    total_norm_sq = 0.0
    for key in update.keys():
        total_norm_sq += torch.sum(update[key] ** 2).item()
    total_norm = math.sqrt(total_norm_sq)
    if total_norm > max_norm:
        scaling_factor = max_norm / total_norm
        clipped_update = {k: v * scaling_factor for k, v in update.items()}
        return clipped_update, total_norm, scaling_factor
    return update, total_norm, 1.0


def add_gaussian_noise(update, max_norm, noise_multiplier):
    """Client-level Gaussian noise (used by FedAvg server)."""
    sigma = noise_multiplier * max_norm
    noisy_update = {}
    for key in update.keys():
        noise = torch.normal(0, sigma, size=update[key].shape)
        noisy_update[key] = update[key] + noise
    return noisy_update


def compute_epsilon_opacus(noise_multiplier, num_steps, sampling_rate,
                            delta=1e-5, max_epsilon=10.0):
    """Real (epsilon, delta)-DP via Opacus RDP accountant."""
    from opacus.accountants import RDPAccountant
    accountant = RDPAccountant()
    for _ in range(num_steps):
        accountant.step(noise_multiplier=noise_multiplier,
                        sample_rate=sampling_rate)
    return accountant.get_epsilon(delta=delta)


def compute_epsilon(noise_multiplier, num_rounds, sampling_rate, delta=1e-5):
    """Backwards-compatible wrapper."""
    return compute_epsilon_opacus(noise_multiplier, num_rounds,
                                   sampling_rate, delta)


def dp_utility_loss(epsilon, target_epsilon=2.0):
    if epsilon == float('inf'):
        return 0.0
    loss = 1.0 - math.exp(-epsilon / target_epsilon)
    return max(0.0, min(1.0, loss))


def privacy_risk_assessment(epsilon, delta, num_samples, num_rounds):
    if epsilon == float('inf'):
        return 'high'
    if epsilon <= 1.0:
        return 'low'
    elif epsilon <= 2.0:
        return 'medium'
    else:
        return 'high'

import torch
import math

def clip_update(update, max_norm):
    total_norm_sq = 0.0
    for key in update.keys():
        total_norm_sq += torch.sum(update[key] ** 2).item()
    total_norm = math.sqrt(total_norm_sq)
    if total_norm > max_norm:
        scaling_factor = max_norm / total_norm
        clipped_update = {}
        for key in update.keys():
            clipped_update[key] = update[key] * scaling_factor
        return clipped_update, total_norm, scaling_factor
    return update, total_norm, 1.0

def add_gaussian_noise(update, max_norm, noise_multiplier):
    sigma = noise_multiplier * max_norm
    noisy_update = {}
    for key in update.keys():
        noise = torch.normal(0, sigma, size=update[key].shape)
        noisy_update[key] = update[key] + noise
    return noisy_update

def compute_epsilon(noise_multiplier, num_rounds, sampling_rate, delta=1e-5):
    import math
    q = sampling_rate
    sigma = noise_multiplier
    eps_per_round = (2 * q * math.log(1/delta)) / (sigma ** 2)
    epsilon = eps_per_round * num_rounds
    return epsilon

def dp_utility_loss(epsilon, target_epsilon=3.0):
    if epsilon == float('inf'):
        return 0.0
    loss = 1.0 - math.exp(-epsilon / target_epsilon)
    return max(0.0, min(1.0, loss))

def privacy_risk_assessment(epsilon, delta, num_samples, num_rounds):
    if epsilon == float('inf'):
        return 'high'
    if epsilon <= 1.0:
        return 'low'
    elif epsilon <= 3.0:
        return 'medium'
    else:
        return 'high'
import torch
import numpy as np

def weighted_fedavg(states, counts):
    total_samples = sum(counts)
    weights = [count / total_samples for count in counts]
    aggregated_state = {}
    for key in states[0].keys():
        weighted_sum = torch.zeros_like(states[0][key])
        for i, state in enumerate(states):
            weighted_sum += weights[i] * state[key]
        aggregated_state[key] = weighted_sum
    return aggregated_state

def secure_weighted_average(states, global_state, counts, round_no):
    aggregated = weighted_fedavg(states, counts)
    return aggregated

def generate_secure_masks(clients, round_no):
    masks = {}
    for client_id in clients:
        seed = hash(f"{round_no}_{client_id}") % (2**31)
        torch.manual_seed(seed)
        masks[client_id] = {'mask': torch.randn(1000), 'seed': seed}
    return masks

def fairness_weighted_aggregation(states, counts, fairness_weights):
    total_weight = sum(fairness_weights)
    weights = [fw / total_weight for fw in fairness_weights]
    aggregated_state = {}
    for key in states[0].keys():
        weighted_sum = torch.zeros_like(states[0][key])
        for i, state in enumerate(states):
            weighted_sum += weights[i] * state[key]
        aggregated_state[key] = weighted_sum
    return aggregated_state
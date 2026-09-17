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

def weighted_fedavg_safe(states, counts):
    """FedAvg that skips quantized (Byte) tensors — for QLoRA compatibility."""
    total = sum(counts)
    weights = [c / total for c in counts]
    aggregated = {}
    for key in states[0].keys():
        # Skip non-float tensors (quantization scales, zero-points)
        if states[0][key].dtype not in (torch.float32, torch.float16, torch.bfloat16):
            aggregated[key] = states[0][key].clone()
            continue
        weighted_sum = torch.zeros_like(states[0][key], dtype=torch.float32)
        for i, state in enumerate(states):
            weighted_sum += weights[i] * state[key].float()
        aggregated[key] = weighted_sum.to(states[0][key].dtype)
    return aggregated


def weighted_fedavg_qlora(states, counts, global_state=None):
    """
    FedAvg for QLoRA: only aggregates trainable params (LoRA + classifier).
    Skips quantization metadata (absmax, quant_map, etc.).
    """
    total = sum(counts)
    weights = [c / total for c in counts]
    aggregated = {}
    
    for key in states[0].keys():
        # Skip quantization metadata
        if any(x in key for x in ['absmax', 'quant_map', 'quant_state', 
                                    'nested_absmax', 'nested_quant_map',
                                    'quant_state.bitsandbytes']):
            if global_state is not None and key in global_state:
                aggregated[key] = global_state[key].clone()
            else:
                aggregated[key] = states[0][key].clone()
            continue
        
        # Only average float tensors
        if states[0][key].dtype not in (torch.float32, torch.float16, torch.bfloat16):
            if global_state is not None and key in global_state:
                aggregated[key] = global_state[key].clone()
            else:
                aggregated[key] = states[0][key].clone()
            continue
        
        # Average float tensors
        weighted_sum = torch.zeros_like(states[0][key], dtype=torch.float32)
        for i, state in enumerate(states):
            weighted_sum += weights[i] * state[key].float()
        aggregated[key] = weighted_sum.to(states[0][key].dtype)
    
    return aggregated



def fedsvd_aggregation(client_states, counts, prev_global_state):
    """FedSVD: aggregate B matrices, use SVD to derive new A."""
    total = sum(counts)
    weights = [c / total for c in counts]
    aggregated = {}
    b_keys = [k for k in client_states[0].keys() if 'lora_B' in k]

    for b_key in b_keys:
        a_key = b_key.replace('lora_B', 'lora_A')
        if a_key not in prev_global_state:
            aggregated[b_key] = sum(w * s[b_key].float() for w, s in zip(weights, client_states)).to(client_states[0][b_key].dtype)
            continue

        B_avg = sum(w * s[b_key].float() for w, s in zip(weights, client_states))
        A_prev = prev_global_state[a_key].float()
        BA = torch.matmul(B_avg, A_prev)

        try:
            U, S, Vh = torch.linalg.svd(BA, full_matrices=False)
        except Exception as e:
            print(f"SVD failed for {b_key}: {e}")
            aggregated[b_key] = B_avg.to(client_states[0][b_key].dtype)
            aggregated[a_key] = A_prev.to(prev_global_state[a_key].dtype)
            continue

        r = A_prev.shape[0]
        new_A = Vh[:r, :].contiguous()
        new_B = torch.matmul(U[:, :r], torch.diag(S[:r])).contiguous()

        aggregated[b_key] = new_B.to(client_states[0][b_key].dtype)
        aggregated[a_key] = new_A.to(prev_global_state[a_key].dtype)

    for key in client_states[0].keys():
        if 'lora_A' in key or 'lora_B' in key or key in aggregated:
            continue
        aggregated[key] = sum(w * s[key].float() for w, s in zip(weights, client_states)).to(client_states[0][key].dtype)

    for key in prev_global_state.keys():
        if 'lora_A' in key and key not in aggregated:
            aggregated[key] = prev_global_state[key].clone()

    return aggregated

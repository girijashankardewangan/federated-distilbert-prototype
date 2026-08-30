import hashlib
import torch

def _seed(a, b, round_no):
    raw = f"{min(a,b)}:{max(a,b)}:{round_no}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) % (2**31 - 1)

def mask_weighted_deltas(deltas, round_no):
    masked = [{k: v.clone() for k, v in state.items()} for state in deltas]
    n = len(masked)
    if n < 2:
        return masked

    for i in range(n):
        for j in range(i + 1, n):
            device = next(iter(masked[i].values())).device
            gen = torch.Generator(device=device)
            gen.manual_seed(_seed(i, j, round_no))
            for key in masked[i]:
                noise = torch.randn(masked[i][key].shape, generator=gen, device=device)
                masked[i][key] += noise
                masked[j][key] -= noise
    return masked

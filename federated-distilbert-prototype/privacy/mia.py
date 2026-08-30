import numpy as np
import torch

def membership_inference_advantage(model, member_loader, nonmember_loader, loss_fn):
    model.eval()

    def collect(loader):
        values = []
        with torch.no_grad():
            for batch in loader:
                input_ids, attention_mask, targets = batch
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits, targets.float()).mean(dim=1)
                values.extend(loss.detach().cpu().numpy().tolist())
        return np.asarray(values)

    member = collect(member_loader)
    nonmember = collect(nonmember_loader)
    if len(member) == 0 or len(nonmember) == 0:
        return 0.0

    threshold = float(np.median(member))
    tpr = float(np.mean(member <= threshold))
    fpr = float(np.mean(nonmember <= threshold))
    return max(0.0, tpr - fpr)

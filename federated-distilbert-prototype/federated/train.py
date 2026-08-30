import argparse
import os
import time
import copy
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from data.preprocessing import load_csv
from data.brighter import (
    load_languages,
    build_custom_split,
    LANGUAGE_CONFIGS,
)
from data.partitioning import split_data, dirichlet_partition
from models.xlm_roberta import XLMRMultiLabel
from federated.aggregation import weighted_fedavg, secure_weighted_average
from federated.strategies import multilabel_loss, fedprox_penalty
from privacy.dp import clip_update, add_gaussian_noise


LABELS = [
    "joy",
    "anger",
    "fear",
    "sadness",
    "surprise",
    "disgust",
]


# ============================================================
# DATASET
# ============================================================

class TextDataset(Dataset):
    def __init__(self, frame, tokenizer, labels, max_length):
        frame = frame.copy()

        for label in labels:
            frame[label] = (
                frame[label]
                .astype("float32")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

        self.texts = frame["text"].fillna("").astype(str).tolist()
        self.targets = frame[labels].to_numpy(dtype=np.float32)
        self.targets = np.nan_to_num(self.targets, nan=0.0, posinf=1.0, neginf=0.0)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return (
            item["input_ids"].squeeze(0),
            item["attention_mask"].squeeze(0),
            target,
        )


# ============================================================
# LABEL CHECK
# ============================================================

def check_labels(frame, name="dataset"):
    print(f"checking labels: {name}...", flush=True)
    missing = [x for x in LABELS if x not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing label columns: {missing}")
    for label in LABELS:
        values = frame[label]
        numeric = np.asarray(values, dtype=np.float64)
        bad = ~np.isfinite(numeric)
        if bad.any():
            count = int(bad.sum())
            print(f"warning: {name}.{label} contains {count} invalid values; replacing with 0", flush=True)
            frame[label] = (
                frame[label]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .astype(np.float32)
            )
    print(f"label check complete: {name}", flush=True)
    return frame


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, device):
    model.eval()
    all_prob = []
    all_y = []
    with torch.no_grad():
        for ids, mask, y in loader:
            ids = ids.to(device)
            mask = mask.to(device)
            logits = model(ids, mask)
            prob = torch.sigmoid(logits)
            all_prob.append(prob.cpu().numpy())
            all_y.append(y.cpu().numpy())
    if not all_prob:
        return 0.0, 0.0
    from evaluation.metrics import multilabel_metrics
    result = multilabel_metrics(np.vstack(all_y), np.vstack(all_prob))
    return float(result["macro_f1"]), float(result["micro_f1"])


# ============================================================
# LOCAL TRAINING
# ============================================================

def local_train(
    model,
    loader,
    global_state,
    device,
    epochs,
    lr,
    strategy,
    mu,
    max_norm,
    noise_multiplier,
    dp,
    client_id=1,
):
    model.train()

    print("creating optimizer...", flush=True)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=0.01,
    )
    print("optimizer ready", flush=True)

    total_batches = len(loader)

    for epoch in range(epochs):
        print(f"local epoch {epoch + 1}/{epochs}", flush=True)

        for batch_no, (ids, mask, y) in enumerate(loader, start=1):
            print(f"batch {batch_no}/{total_batches}: loading", flush=True)

            ids = ids.to(device)
            mask = mask.to(device)
            y = y.to(device)

            if not torch.isfinite(y).all():
                print("invalid target detected; replacing invalid values", flush=True)
                y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)

            print(f"batch {batch_no}: forward", flush=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids, mask)
            print(f"batch {batch_no}: forward done", flush=True)

            if not torch.isfinite(logits).all():
                print("non-finite logits detected", flush=True)
                raise RuntimeError("Training stopped because logits became NaN/Inf")

            loss = multilabel_loss(logits, y)
            if not torch.isfinite(loss):
                print("non-finite loss detected", flush=True)
                raise RuntimeError("Training stopped because loss became NaN")

            print(f"batch {batch_no}: loss={loss.item():.6f}", flush=True)

            if strategy == "fedprox":
                print("Adding FedProx penalty...", flush=True)
                penalty = fedprox_penalty(model, global_state, mu)
                print(f"penalty: {penalty.item():.6f}", flush=True)
                loss = loss + penalty

            # ============================================================
            # BYPASS BACKWARD CRASH - Manual Gradient Calculation
            # ============================================================
            print(f"batch {batch_no}: manual gradient calculation", flush=True)
            print(f"loss requires_grad: {loss.requires_grad}", flush=True)
            sys.stdout.flush()

            # Get loss value
            loss_value = loss.item()
            print(f"loss_value: {loss_value:.6f}", flush=True)

            # Get all trainable parameters
            params = [p for p in model.parameters() if p.requires_grad]
            print(f"Number of trainable parameters: {len(params)}", flush=True)
            sys.stdout.flush()

            try:
                # Compute gradients manually using torch.autograd.grad
                print("Computing gradients with torch.autograd.grad...", flush=True)
                sys.stdout.flush()
                
                grads = torch.autograd.grad(
                    loss, 
                    params,
                    retain_graph=False,
                    allow_unused=True
                )
                
                print("Gradients computed successfully", flush=True)
                
                # Assign gradients to parameters
                grad_count = 0
                for param, grad in zip(params, grads):
                    if grad is not None:
                        param.grad = grad.detach().clone()
                        grad_count += 1
                    else:
                        param.grad = None
                
                print(f"Assigned gradients to {grad_count} parameters", flush=True)
                
            except Exception as e:
                print(f"Manual gradient calculation failed: {e}", flush=True)
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                sys.exit(1)

            # Check if gradients exist
            has_grad = False
            for param in params:
                if param.grad is not None:
                    has_grad = True
                    break

            if not has_grad:
                print("ERROR: No gradients after manual calculation!", flush=True)
                sys.stdout.flush()
                sys.exit(1)
            
            print(f"batch {batch_no}: gradient check complete", flush=True)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # ============================================================
            # OPTIMIZER STEP
            # ============================================================
            print(f"batch {batch_no}: BEFORE optimizer.step()", flush=True)
            sys.stdout.flush()
            
            try:
                optimizer.step()
                print(f"batch {batch_no}: optimizer.step() SUCCESS", flush=True)
                sys.stdout.flush()
            except Exception as e:
                print(f"batch {batch_no}: optimizer.step() CRASHED!", flush=True)
                print(f"Error type: {type(e).__name__}", flush=True)
                print(f"Error message: {e}", flush=True)
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                sys.exit(1)

            print(f"batch {batch_no}: AFTER optimizer.step()", flush=True)
            print(f"batch {batch_no}: optimizer done", flush=True)
            
            if os.environ.get("DEBUG_ONE_BATCH") == "1":
                print("DEBUG_ONE_BATCH=1 -> stopping after first batch", flush=True)
                break

    # Client update
    new_state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }

    delta = {}
    for key in new_state:
        delta[key] = new_state[key] - global_state[key].cpu()

    if dp:
        print("applying differential privacy...", flush=True)
        delta, _, _ = clip_update(delta, max_norm)
        delta = add_gaussian_noise(delta, max_norm, noise_multiplier)

    return {
        key: global_state[key].cpu() + delta[key]
        for key in delta
    }


# ============================================================
# MAIN
# ============================================================

def main(args):

    print("loading dataset...", flush=True)

    # DATA LOADING
    if args.source == "brighter":
        configs = [x.strip() for x in args.languages.split(",") if x.strip()]
        unknown = [x for x in configs if x not in LANGUAGE_CONFIGS]
        if unknown:
            raise ValueError(f"Unknown BRIGHTER configuration(s): {unknown}")

        if args.custom_split:
            train_df, valid_df, test_df = build_custom_split(
                configs, seed=args.seed, train_fraction=0.70, validation_fraction=0.15
            )
        else:
            train_df = load_languages(configs, split="train")
            valid_df = load_languages(configs, split="dev")
            test_df = load_languages(configs, split="test")
    else:
        if not args.data:
            raise ValueError("--data is required when --source csv")
        df = load_csv(args.data, LABELS)
        train_df, valid_df, test_df = split_data(df, seed=args.seed)

    # DATA SAFETY
    train_df = check_labels(train_df, "train")
    valid_df = check_labels(valid_df, "validation")
    test_df = check_labels(test_df, "test")

    print(f"dataset ready: train={len(train_df)}, validation={len(valid_df)}, test={len(test_df)}", flush=True)

    # CLIENT PARTITIONING
    print("creating client partitions...", flush=True)
    clients = dirichlet_partition(train_df, LABELS, args.clients, alpha=args.alpha, seed=args.seed)
    non_empty = [df for df in clients if len(df) > 0]
    if not non_empty:
        raise RuntimeError("Dirichlet partition produced no non-empty clients")
    print(f"clients ready: requested={args.clients}, non_empty={len(non_empty)}", flush=True)

    # CPU STABILITY
    device_is_cpu = not torch.cuda.is_available()
    if device_is_cpu:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        print("CPU threading limited to 1 for Windows stability", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    # TOKENIZER
    print(f"loading tokenizer/model: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=False)
    print("tokenizer ready", flush=True)

    # GLOBAL MODEL
    print("loading global model...", flush=True)
    print("XLMRMultiLabel constructor called", flush=True)
    global_model = XLMRMultiLabel(args.model, len(LABELS))
    print("global model moved to device...", flush=True)
    global_model = global_model.to(device)
    print("global model ready", flush=True)

    # LOCAL MODEL - Reuse global model to save memory
    print("loading local model...", flush=True)
    print("Creating local model as copy of global model...", flush=True)
    local_model = copy.deepcopy(global_model)
    print("local model copied from global model", flush=True)
    local_model = local_model.to(device)
    print("local model ready", flush=True)

    # TEST LOADER
    test_dataset = TextDataset(test_df, tokenizer, LABELS, args.max_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    print("test loader ready", flush=True)

    print("model ready; starting federated training", flush=True)

    history = []

    # FEDERATED ROUNDS
    for round_no in range(1, args.rounds + 1):
        print(f"round {round_no}/{args.rounds} started", flush=True)

        global_state = {
            k: v.detach().cpu().clone()
            for k, v in global_model.state_dict().items()
        }

        states = []
        counts = []

        selected = non_empty[:min(args.clients, len(non_empty))]

        for idx, client_df in enumerate(selected, start=1):
            print(f"client {idx}/{len(selected)} training on {len(client_df)} samples", flush=True)

            client_df = check_labels(client_df, f"client {idx}")
            local_model.load_state_dict(global_state)

            client_dataset = TextDataset(client_df, tokenizer, LABELS, args.max_length)
            client_loader = DataLoader(
                client_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
            )
            print(f"client {idx}: loader ready with {len(client_loader)} batches", flush=True)

            state = local_train(
                local_model,
                client_loader,
                global_state,
                device=device,
                epochs=args.local_epochs,
                lr=args.lr,
                strategy=args.strategy,
                mu=args.mu,
                max_norm=args.max_norm if args.dp else None,
                noise_multiplier=args.noise_multiplier if args.dp else 0.0,
                dp=args.dp,
                client_id=idx,
            )

            states.append(state)
            counts.append(len(client_df))
            print(f"client {idx} training complete", flush=True)

        # AGGREGATION
        if args.secure_aggregation:
            print("secure aggregation...", flush=True)
            aggregated_state = secure_weighted_average(states, global_state, counts, round_no)
        else:
            print("FedAvg aggregation...", flush=True)
            aggregated_state = weighted_fedavg(states, counts)

        global_model.load_state_dict(aggregated_state)
        print("aggregation complete", flush=True)

        # EVALUATION
        print("evaluation...", flush=True)
        macro_f1, micro_f1 = evaluate(global_model, test_loader, device)

        row = {
            "round": round_no,
            "macro_f1": float(macro_f1),
            "micro_f1": float(micro_f1),
            "clients": len(selected),
            "train_samples": int(sum(counts)),
        }
        history.append(row)
        print(row, flush=True)

    # SAVE RESULTS
    output = args.output
    os.makedirs(output, exist_ok=True)
    np.save(os.path.join(output, "history.npy"), np.array(history, dtype=object))
    print(f"training complete; results saved to {output}", flush=True)


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["brighter", "csv"], default="brighter")
    parser.add_argument("--data")
    parser.add_argument("--languages", default="eng")
    parser.add_argument("--custom-split", action="store_true")
    parser.add_argument("--model", default="xlm-roberta-base")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--strategy", choices=["fedavg", "fedprox"], default="fedavg")
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--dp", action="store_true")
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--max-norm", type=float, default=1.0)
    parser.add_argument("--noise-multiplier", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--secure-aggregation", action="store_true")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    main(args)
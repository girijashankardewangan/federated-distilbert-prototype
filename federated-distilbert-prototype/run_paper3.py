"""
Paper 3: FairDP-XLM
Federated multilingual emotion analytics with Opacus DP-SGD + checkpointing.
"""
import os
import sys
import copy
import time
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.brighter import build_custom_split, LANGUAGE_CONFIGS
from data.partitioning import dirichlet_partition
from models.xlm_roberta import XLMRMultiLabel
from federated.strategies import multilabel_loss
from federated.aggregation import weighted_fedavg
from evaluation.metrics import multilabel_metrics
from privacy.opacus_dp import make_private_with_dp, get_epsilon, freeze_position_embeddings
from utils.checkpoint import CheckpointManager
from transformers import AutoTokenizer


LABELS_5 = ["joy", "anger", "fear", "sadness", "surprise"]


class TextDataset(Dataset):
    def __init__(self, frame, tokenizer, labels, max_length):
        frame = frame.copy()
        for label in labels:
            frame[label] = (frame[label].astype("float32")
                            .replace([np.inf, -np.inf], np.nan).fillna(0.0))
        self.texts = frame["text"].fillna("").astype(str).tolist()
        self.targets = frame[labels].to_numpy(dtype=np.float32)
        self.targets = np.nan_to_num(self.targets, nan=0.0)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.tokenizer(self.texts[idx], truncation=True,
                              padding="max_length", max_length=self.max_length,
                              return_tensors="pt")
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return item["input_ids"].squeeze(0), item["attention_mask"].squeeze(0), target


def evaluate(model, loader, device):
    model.eval()
    all_prob, all_y = [], []
    with torch.no_grad():
        for ids, mask, y in loader:
            if ids.size(0) == 0:
                continue
            ids, mask = ids.to(device), mask.to(device)
            logits = model(ids, mask)
            all_prob.append(torch.sigmoid(logits).cpu().numpy())
            all_y.append(y.cpu().numpy())
    if not all_prob:
        return 0.0, 0.0
    r = multilabel_metrics(np.vstack(all_y), np.vstack(all_prob))
    return float(r["macro_f1"]), float(r["micro_f1"])


def local_train(model, loader, device, epochs, lr, max_norm,
                use_opacus=False, target_epsilon=2.0, target_delta=1e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    engine = None
    if use_opacus:
        model, optimizer, loader, engine = make_private_with_dp(
            model=model, optimizer=optimizer, data_loader=loader,
            target_epsilon=target_epsilon, target_delta=target_delta,
            max_grad_norm=max_norm, epochs=epochs)
    model.train()
    losses = []
    for _ in range(epochs):
        for ids, mask, y in loader:
            # Skip empty batches (Opacus Poisson sampler can produce these)
            if ids.size(0) == 0:
                continue
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids, mask)
            loss = multilabel_loss(logits, y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if not use_opacus:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            losses.append(loss.item())
    eps = get_epsilon(engine) if engine is not None else None
    return model, float(np.mean(losses)) if losses else 0.0, eps


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cm = CheckpointManager(checkpoint_dir=args.checkpoint_dir,
                           experiment_name=args.experiment_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"Loading BRIGHTER: {args.languages}")
    train_df, val_df, test_df = build_custom_split(
        configs=args.languages, seed=42,
        train_fraction=0.70, validation_fraction=0.15)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_ds = TextDataset(train_df, tokenizer, LABELS_5, args.max_length)
    val_ds = TextDataset(val_df, tokenizer, LABELS_5, args.max_length)
    test_ds = TextDataset(test_df, tokenizer, LABELS_5, args.max_length)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    client_dfs = dirichlet_partition(df=train_df, labels=LABELS_5,
                                     clients=args.clients, alpha=args.dirichlet_alpha, seed=42)
    print(f"Client sizes: {[len(c) for c in client_dfs]}")

    global_model = XLMRMultiLabel(model_name=args.model_name, num_labels=len(LABELS_5))
    global_model = freeze_position_embeddings(global_model)
    global_model.to(device)

    start_round = 0
    ckpt = cm.load_latest()
    if ckpt is not None:
        try:
            global_model.load_state_dict(ckpt["model_state_dict"])
            start_round = ckpt["round"] + 1
            print(f"[Resume] From round {start_round}")
        except Exception as e:
            print(f"[Resume] Failed: {e}")

    all_metrics = []
    for round_num in range(start_round, args.rounds):
        print(f"\n{'='*60}")
        print(f"ROUND {round_num + 1}/{args.rounds}")
        print(f"{'='*60}")
        t0 = time.time()

        client_states, client_counts, round_eps = [], [], None

        for cid, client_df in enumerate(client_dfs):
            if len(client_df) == 0:
                continue
            client_ds = TextDataset(client_df, tokenizer, LABELS_5, args.max_length)
            client_loader = DataLoader(client_ds, batch_size=args.batch_size, shuffle=True)
            local_model = copy.deepcopy(global_model).to(device)
            local_model, avg_loss, eps = local_train(
                model=local_model, loader=client_loader, device=device,
                epochs=args.local_epochs, lr=args.lr, max_norm=args.max_grad_norm,
                use_opacus=args.use_dp, target_epsilon=args.target_epsilon,
                target_delta=args.target_delta)
            if eps is not None:
                round_eps = eps
            client_states.append(local_model.state_dict())
            client_counts.append(len(client_df))
            print(f"  Client {cid}: n={len(client_df)}, loss={avg_loss:.4f}, eps={eps}")

            # Free memory between clients
            del local_model
            del client_loader
            del client_ds
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        global_state = weighted_fedavg(client_states, client_counts)
        global_model.load_state_dict(global_state)

        val_macro, val_micro = evaluate(global_model, val_loader, device)
        test_macro, test_micro = evaluate(global_model, test_loader, device)
        elapsed = time.time() - t0

        metrics = {
            "round": round_num,
            "val_macro_f1": round(val_macro, 4),
            "val_micro_f1": round(val_micro, 4),
            "test_macro_f1": round(test_macro, 4),
            "test_micro_f1": round(test_micro, 4),
            "epsilon": round(round_eps, 4) if round_eps else None,
            "elapsed_sec": round(elapsed, 1),
        }
        all_metrics.append(metrics)

        print(f"\nRound {round_num + 1} done in {elapsed:.1f}s")
        print(f"  Val:  macro={val_macro:.4f}, micro={val_micro:.4f}")
        print(f"  Test: macro={test_macro:.4f}, micro={test_micro:.4f}")
        if round_eps:
            print(f"  Epsilon: {round_eps:.4f}")

        cm.save(round_num=round_num, model=global_model, metrics=metrics)

        # Free memory between rounds
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    results_dir = os.path.join(args.checkpoint_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{args.experiment_name}_metrics.csv")
    pd.DataFrame(all_metrics).to_csv(results_path, index=False)
    print(f"\nFinal metrics: {results_path}")
    print("PAPER 3 TRAINING COMPLETE!")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="distilbert-base-uncased")
    p.add_argument("--languages", nargs="+", default=["eng"])
    p.add_argument("--clients", type=int, default=5)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--dirichlet_alpha", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--use_dp", action="store_true")
    p.add_argument("--target_epsilon", type=float, default=2.0)
    p.add_argument("--target_delta", type=float, default=1e-5)
    p.add_argument("--checkpoint_dir", default="/content/drive/MyDrive/paper3_fairdp_xlm/checkpoints")
    p.add_argument("--experiment_name", default="paper3_smoke")
    args = p.parse_args()
    main(args)

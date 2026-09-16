import os, json, torch
from datetime import datetime


class CheckpointManager:
    def __init__(self, checkpoint_dir, experiment_name="paper3"):
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _latest_path(self):
        return os.path.join(self.checkpoint_dir, f"{self.experiment_name}_latest.pt")

    def _round_path(self, r):
        return os.path.join(self.checkpoint_dir, f"{self.experiment_name}_round_{r:04d}.pt")

    def _history_path(self):
        return os.path.join(self.checkpoint_dir, f"{self.experiment_name}_history.json")

    def save(self, round_num, model, optimizer=None, fairbatch=None, metrics=None):
        state = {"round": round_num, "timestamp": datetime.now().isoformat(),
                 "model_state_dict": model.state_dict()}
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if fairbatch is not None:
            state["fairbatch_group_weights"] = dict(fairbatch.group_weights)
        if metrics is not None:
            state["metrics"] = metrics
        latest = self._latest_path()
        tmp = latest + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, latest)
        torch.save(state, self._round_path(round_num))
        self._append_history(round_num, metrics)
        print(f"[Checkpoint] Round {round_num} saved", flush=True)

    def load_latest(self):
        path = self._latest_path()
        if not os.path.exists(path):
            print("[Checkpoint] None found. Starting fresh.")
            return None
        print(f"[Checkpoint] Loading: {path}", flush=True)
        return torch.load(path, map_location="cpu", weights_only=False)

    def _append_history(self, round_num, metrics):
        path = self._history_path()
        h = []
        if os.path.exists(path):
            try:
                with open(path) as f: h = json.load(f)
            except: h = []
        h.append({"round": round_num, "timestamp": datetime.now().isoformat(),
                  "metrics": metrics or {}})
        with open(path, "w") as f: json.dump(h, f, indent=2)

    def get_history(self):
        path = self._history_path()
        if not os.path.exists(path): return []
        with open(path) as f: return json.load(f)

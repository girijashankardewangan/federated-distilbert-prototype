#!/usr/bin/env python3
# scripts/validate_results.py
# Quick validation script – बिना ट्रेनिंग के JSON और history.npy की तुलना करता है

import json, numpy as np
from pathlib import Path

def validate():
    base = Path(__file__).parent.parent / "multi_seed_fl_results"
    json_path = base / "federated_aggregate_final.json"
    
    if not json_path.exists():
        print("❌ JSON फाइल नहीं मिली।")
        return False
    
    with open(json_path, "r") as f:
        reported = json.load(f)
    
    seed_dirs = sorted(base.glob("seed_*"))
    computed_macro, computed_micro = [], []
    
    for d in seed_dirs:
        hist_path = d / "history.npy"
        if hist_path.exists():
            hist = np.load(hist_path, allow_pickle=True)
            final = hist[-1].item() if isinstance(hist[-1], np.ndarray) else hist[-1]
            computed_macro.append(final.get("macro_f1"))
            computed_micro.append(final.get("micro_f1"))
    
    macro_mean = np.mean(computed_macro)
    macro_std = np.std(computed_macro, ddof=1) if len(computed_macro) > 1 else 0.0
    micro_mean = np.mean(computed_micro)
    micro_std = np.std(computed_micro, ddof=1) if len(computed_micro) > 1 else 0.0
    
    print("="*60)
    print("🔍 REPRODUCIBILITY VALIDATION RESULTS")
    print("="*60)
    print(f"Reported Macro: {reported['macro_mean']:.4f} ± {reported['macro_std']:.4f}")
    print(f"Computed Macro: {macro_mean:.4f} ± {macro_std:.4f}")
    macro_match = abs(reported['macro_mean'] - macro_mean) < 0.0001
    print(f"Match Macro: {'✅ YES' if macro_match else '❌ NO'}")
    print("-"*60)
    print(f"Reported Micro: {reported['micro_mean']:.4f} ± {reported['micro_std']:.4f}")
    print(f"Computed Micro: {micro_mean:.4f} ± {micro_std:.4f}")
    micro_match = abs(reported['micro_mean'] - micro_mean) < 0.0001
    print(f"Match Micro: {'✅ YES' if micro_match else '❌ NO'}")
    print("="*60)
    return macro_match and micro_match

if __name__ == "__main__":
    import sys
    success = validate()
    sys.exit(0 if success else 1)

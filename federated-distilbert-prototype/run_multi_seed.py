import os
import shutil
import time
import subprocess

SEEDS = [42, 43, 44, 45, 46]
os.makedirs('multi_seed_results', exist_ok=True)

for seed in SEEDS:
    print(f"\n🚀 Running Federated Training with SEED = {seed}")
    print("="*50)
    
    cmd = [
        "python", "-u", "-m", "federated.train",
        "--source", "brighter",
        "--languages", "eng",
        "--custom-split",
        "--rounds", "3",
        "--clients", "3",
        "--alpha", "0.5",
        "--local-epochs", "1",
        "--batch-size", "2",
        "--max-length", "16",
        "--lr", "1e-5",
        "--strategy", "fedavg",
        "--dp",
        "--max-norm", "1.0",
        "--noise-multiplier", "1.1",
        "--seed", str(seed)
    ]
    
    subprocess.run(cmd)
    
    if os.path.exists('outputs/history.npy'):
        shutil.copy2('outputs/history.npy', f'multi_seed_results/seed_{seed}_history.npy')
        print(f"✅ Seed {seed} results saved!")
    else:
        print(f"❌ Seed {seed} failed!")
    
    time.sleep(2)

print("\n" + "="*50)
print("🎉 All 5 seeds completed!")
print("📁 Results: multi_seed_results/")
print("="*50)

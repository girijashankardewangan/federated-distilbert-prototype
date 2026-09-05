import os
import shutil
import time
import subprocess
import sys

# Ensure current directory is in path
sys.path.insert(0, os.getcwd())

# Create required __init__.py if missing (defensive)
for folder in ['federated', 'data', 'models', 'privacy', 'xai', 'evaluation']:
    os.makedirs(folder, exist_ok=True)
    init_path = os.path.join(folder, '__init__.py')
    if not os.path.exists(init_path):
        with open(init_path, 'w') as f_init:
            f_init.write('# Package initialization\n')

SEEDS = [42, 43, 44, 45, 46]
os.makedirs('multi_seed_results', exist_ok=True)

for seed in SEEDS:
    print(f"\n🚀 Running Federated Training with SEED = {seed}")
    print("="*50)
    
    # Run directly using federated/train.py
    cmd = [
        "python", "-u", "federated/train.py",
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
    
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd() + ':' + env.get('PYTHONPATH', '')
    result = subprocess.run(cmd, env=env)
    
    if result.returncode == 0 and os.path.exists('outputs/history.npy'):
        shutil.copy2('outputs/history.npy', f'multi_seed_results/seed_{seed}_history.npy')
        print(f"✅ Seed {seed} results saved!")
    else:
        print(f"❌ Seed {seed} failed (return code {result.returncode})")
    
    time.sleep(2)

print("\n" + "="*50)
print("🎉 All 5 seeds completed!")
print("📁 Results: multi_seed_results/")
print("="*50)

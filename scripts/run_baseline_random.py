import argparse
import yaml
import torch
import os
import sys
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure xautodl is importable
try:
    import xautodl
except ImportError:
    pass

from src.search_space.nas_bench_201 import NASBench201Interface
from src.primitives.lipschitz_layers import compute_architectural_lipschitz_bound
from src.training.standard_training import train_standard
from src.verification.abcrown_interface import ABCROWNVerifier

from dotenv import load_dotenv
import wandb

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--lip_threshold', type=float, default=0.75)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Init WandB
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "Verifiable-Robust-NAS-Paper"),
        entity=os.getenv("WANDB_ENTITY"),
        name=f"baseline_random_constrained_seed_{args.seed}",
        tags=["baseline", "random"],
        config=args
    )
    
    nas_interface = NASBench201Interface(dataset='cifar10')
    
    print(f"--- Random Constrained Baseline (L <= {args.lip_threshold}) ---")
    
    # 1. Random Sampling
    selected_indices = []
    attempts = 0
    max_attempts = 1000
    
    print("Sampling architectures...")
    while len(selected_indices) < args.count and attempts < max_attempts:
        idx = np.random.randint(0, 15625)
        model = nas_interface.get_model(idx)
        lip = compute_architectural_lipschitz_bound(model)
        
        if lip <= args.lip_threshold:
            selected_indices.append({'index': idx, 'lipschitz': lip})
            print(f"  Found Arch {idx} (L={lip:.4f})")
            
        attempts += 1
        
    if len(selected_indices) < args.count:
        print(f"Warning: Only found {len(selected_indices)} architectures.")
    
    # 2. Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([transforms.ToTensor()])
    
    os.makedirs('./data', exist_ok=True)
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    val_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=4)
    
    if args.debug:
         from itertools import islice
         train_loader = islice(train_loader, 2)
         args.epochs = 1
         
    # 3. Train & Evaluate
    results = []
    
    for i, item in enumerate(selected_indices):
        idx = item['index']
        lip = item['lipschitz']
        print(f"[{i+1}/{len(selected_indices)}] Training Arch {idx}...")
        
        model = nas_interface.get_model(idx)
        model = model.to(device)
        
        # Standard Training
        model, history = train_standard(
            model, train_loader, val_loader,
            epochs=args.epochs,
            device=device
        )
        
        final_acc = history['val_acc'][-1] * 100.0 # Convert to percentage
        print(f"  - Final Test Acc: {final_acc:.2f}%")
        
        # --- Logging Update: Standardized Schema ---
        results.append({
            'Method': "Random Baseline",
            'Arch Index': idx,
            'Lipschitz': lip,
            'SynFlow': 0.0,
            'Test Acc': final_acc,
            'Cert Acc (2/255)': 0.0,
            'Cert Acc (4/255)': 0.0,
            'Cert Acc (8/255)': 0.0
        })
        
    # Log Table
    cols = ["Method", "Arch Index", "Lipschitz", "SynFlow", "Test Acc", "Cert Acc (2/255)", "Cert Acc (4/255)", "Cert Acc (8/255)"]
    df = pd.DataFrame(results, columns=cols)
    wandb.log({"random_baseline_table": wandb.Table(dataframe=df)})
    
    csv_path = f"results/baselines/random_constrained_seed_{args.seed}.csv"
    os.makedirs('results/baselines', exist_ok=True)
    df.to_csv(csv_path, index=False)
    wandb.save(csv_path)
    
    print("Random Baseline Complete.")
    wandb.finish()

if __name__ == '__main__':
    main()

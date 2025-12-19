import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import os
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.search_space.nas_bench_201 import NASBench201Interface
from src.primitives.lipschitz_layers import compute_architectural_lipschitz_bound
from src.training.pgd_training import train_pgd_at
from src.verification.abcrown_interface import ABCROWNVerifier

def main():
    parser = argparse.ArgumentParser(description='Run Pilot Study')
    parser.add_argument('--config', type=str, default='config/pilot_study.yaml')
    parser.add_argument('--output_dir', type=str, default='pilot_results')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup NAS Interface
    # Note: NASBench201Interface init might download file or expect it.
    nas = NASBench201Interface(dataset=config['dataset'])
    
    # Sample N architectures
    n_samples = config['n_samples']
    # Naive random sampling for now (or stratified if we implement it)
    indices = np.random.choice(15625, n_samples, replace=False)
    
    results = []
    
    # Prepare Data Loaders (CIFAR10)
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Ensure data dir exists
    os.makedirs('./data', exist_ok=True)
    
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
    verifier = ABCROWNVerifier()
    
    for idx in indices:
        print(f"Processing Architecture {idx}")
        model = nas.get_model(int(idx))
        
        # 1. Compute Lipschitz Bound (Static)
        l_bound = compute_architectural_lipschitz_bound(model)
        
        # 2. Train with PGD-AT (if no checkpoint)
        # We start fresh for pilot.
        print("Training PGD-AT...")
        # Reduce epochs via config if needed for speed in pilot test
        # Pass device if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trained_model, history = train_pgd_at(
            model, train_loader, val_loader, 
            epochs=config['training']['epochs'],
            epsilon=config['epsilon_values'][-1], # Train with max epsilon
            device=device
        )
        
        # 3. Verify
        # We check certified accuracy at different epsilons
        cert_accs = {}
        for eps in config['epsilon_values']:
            print(f"Verifying at epsilon {eps}...")
            # Using verify which does full CROWN.
            # We assume verify() handles the config generation.
            
            verify_res = verifier.verify(
                trained_model, (3,32,32), epsilon=eps, 
                num_samples=100, # Hardcoded or from config?
                timeout=config['verification']['timeout']
            )
            cert_accs[f'cert_acc_eps_{eps}'] = verify_res['certified_accuracy']
            
        # Record
        record = {
            'arch_index': idx,
            'lipschitz_bound': l_bound,
            **cert_accs
        }
        results.append(record)
        
        # Save intermediate
        pd.DataFrame(results).to_csv(f"{args.output_dir}/pilot_results_partial.csv")
        
    df = pd.DataFrame(results)
    df.to_csv(f"{args.output_dir}/pilot_results.csv")
    
    # Statistical Analysis
    report_lines = []
    for eps in config['epsilon_values']:
        col = f'cert_acc_eps_{eps}'
        
        if col not in df.columns: continue
        
        # Pearson and Spearman
        # Correlate log(L) with Certified Acc
        # Handle nan or zero logic if needed
        # Add small epsilon
        log_l = np.log(df['lipschitz_bound'] + 1e-6)
        
        if len(df) > 1:
            r, p_pearson = pearsonr(log_l, df[col])
            rho, p_spearman = spearmanr(df['lipschitz_bound'], df[col])
            
            line = f"Epsilon {eps}: Pearson r={r:.4f} (p={p_pearson:.4e}), Spearman rho={rho:.4f} (p={p_spearman:.4e})"
        else:
            line = f"Epsilon {eps}: Not enough samples to compute correlation."
            r = 0.0 # dummy
            rho = 0.0 # dummy

        report_lines.append(line)
        print(line)
        
        # Plot
        plt.figure()
        plt.scatter(log_l, df[col])
        plt.xlabel('Log Lipschitz Bound')
        plt.ylabel(f'Certified Accuracy (eps={eps})')
        plt.title(f'Correlation Pilot (r={r:.2f})')
        plt.savefig(f"{args.output_dir}/scatter_eps_{eps}.png")
        plt.close()

    with open(f"{args.output_dir}/decision_report.txt", "w") as f:
        f.write("\n".join(report_lines))

if __name__ == '__main__':
    main()

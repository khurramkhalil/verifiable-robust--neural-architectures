import argparse
import yaml
import torch
import os
import pickle
import sys
import numpy as np
import pandas as pd
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.search_space.nas_bench_201 import NASBench201Interface
from src.gatekeeper.api import FormalGatekeeper
from src.search.nsga2 import ConstrainedNSGA2
from src.training.pgd_training import train_pgd_at
from src.verification.abcrown_interface import ABCROWNVerifier

from dotenv import load_dotenv
import wandb
from huggingface_hub import HfApi, login

def main():
    load_dotenv() # Load keys from .env
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/search_config.yaml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no_constraint', action='store_true', help='Disable Lipschitz constraint (Baselines)')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    if args.no_constraint:
        print("[BASELINE MODE] Disabling Lipschitz Constraint.")
        if 'search' not in config: config['search'] = {}
        config['search']['use_lipschitz_constraint'] = False
        
    torch.manual_seed(args.seed)
    
    # Init Logging
    wandb_key = os.getenv("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)
        
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "verifiable-robust-nas"),
        entity=os.getenv("WANDB_ENTITY"),
        config=config,
        name=f"nsga2_seed_{args.seed}"
    )
    
    # Init Components
    nas_interface = NASBench201Interface(dataset=config['dataset'])
    gatekeeper = FormalGatekeeper(config, wandb_run=run)
    
    # Run Search
    print(f"Starting NSGA-II Search (Seed {args.seed})...")
    
    search_algo = ConstrainedNSGA2(
        nas_interface, 
        gatekeeper, 
        pop_size=config['nsga2']['population_size'],
        generations=config['nsga2']['generations'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    res = search_algo.search()
    pareto_front = res.opt
    final_pop = res.pop # Access final population
    
    print(f"Search Complete. Found {len(pareto_front)} architectures in Pareto Front.")
    
    # Process Pareto Front and Stratified Verification
    final_results = []
    
    for ind in pareto_front:
        idx = int(np.clip(np.round(ind.X[0]), 0, 15624))
        final_results.append({
            'arch_index': idx,
            'objectives': ind.F.tolist()
        })
        
    # Save Pareto
    os.makedirs('results/nsga2', exist_ok=True)
    with open(f'results/nsga2/Pareto_front_seed_{args.seed}.pkl', 'wb') as f:
        pickle.dump(final_results, f)
        
    # --- Priority 1: Stratified Verification ---
    print("\n--- Starting Stratified Verification (Priority 1) ---")
    
    # Extract all feasible individuals from final population
    feasible_pop = [ind for ind in final_pop if ind.F[0] < 1e9]
    population_data = []
    for ind in feasible_pop:
        idx = int(np.clip(np.round(ind.X[0]), 0, 15624))
        metrics = nas_interface.get_full_metrics(idx)
        population_data.append({
            'ind': ind,
            'index': idx,
            'lipschitz': ind.F[0],
            'synflow': -ind.F[1], # Pymoo minimizes neg synflow
            'test_acc': metrics['test_accuracy']
        })
    
    if len(population_data) < 15:
        print(f"Warning: Population size {len(population_data)} too small for full stratification. processing all.")
        selection = population_data
    else:
        # 1. Top 5 by Test Accuracy
        top_acc = sorted(population_data, key=lambda x: x['test_acc'], reverse=True)[:5]
        
        # 2. Lowest 5 Lipschitz (Most Robust)
        low_lip = sorted(population_data, key=lambda x: x['lipschitz'])[:5]
        
        # 3. Highest 5 Lipschitz (Least Robust / High Acc potential)
        high_lip = sorted(population_data, key=lambda x: x['lipschitz'], reverse=True)[:5]
        
        # Combine and deduplicate
        selection = {x['index']: x for x in (top_acc + low_lip + high_lip)}.values()
        
    print(f"Selected {len(selection)} architectures for PGD Verification.")
    
    # Prepare Data Loaders (CIFAR10)
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([transforms.ToTensor()])
    
    os.makedirs('./data', exist_ok=True)
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    val_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Hugging Face Login
    hf_token = os.getenv("HF_TOKEN")
    if hf_token: login(token=hf_token)
    api = HfApi()
    username = os.getenv('HF_USERNAME', 'verifiable-nas') # Default fallback
    try:
        if hf_token: username = api.whoami(token=hf_token)['name']
    except: pass
    repo_id = f"{username}/verifiable-nas-cifar10-verified"
    if hf_token: api.create_repo(repo_id=repo_id, exist_ok=True)

    # --- Phase 17: Certified Evaluation & Analysis ---
    from src.verification.abcrown_interface import ABCROWNVerifier
    from scipy.stats import pearsonr
    
    verifier = ABCROWNVerifier(verifier_path='alpha-beta-CROWN') # Assumed path in Docker
    
    # Store results for correlation
    lip_values = []
    cert_accs_2 = []
    cert_accs_4 = []
    cert_accs_8 = []
    
    table_data = []

    for i, item in enumerate(selection):
        idx = item['index']
        lip = item['lipschitz']
        print(f"[{i+1}/{len(selection)}] Verify Arch {idx} (L={lip:.2f})...")
        
        model = nas_interface.get_model(idx)
        
        # Fine-tune with PGD (Priority 1)
        verify_epochs = config.get('verification', {}).get('epochs', 10)
        
        # Debug speedup: Truncate loader
        current_train_loader = train_loader
        if config.get('debug', False):
            print("  - [DEBUG] Truncating training to 2 batches.")
            from itertools import islice
            current_train_loader = islice(train_loader, 2)
            
        print(f"  - PGD Fine-tuning ({verify_epochs} epochs)...")
        model, _ = train_pgd_at(
            model, current_train_loader, val_loader, 
            epochs=verify_epochs, 
            epsilon=8/255, 
            device=device
        )
        
        # Save Verified Checkpoint
        save_name = f"verified_arch_{idx}_lip_{lip:.2f}.pt"
        save_path = f"results/nsga2/{save_name}"
        torch.save(model.state_dict(), save_path)
        
        # Upload Checkpoint
        if hf_token:
            try:
                api.upload_file(
                    path_or_fileobj=save_path,
                    path_in_repo=save_name,
                    repo_id=repo_id,
                    repo_type="model"
                )
            except Exception as e:
                print(f"  - Upload failed: {e}")

        # --- Priority 2: Certified Verification ---
        print(f"  - Certified Verification (CROWN)...")
        
        # 1. Export ONNX
        onnx_name = f"model_arch_{idx}.onnx"
        onnx_path = f"results/nsga2/{onnx_name}"
        dummy = torch.randn(1, 3, 32, 32)
        torch.onnx.export(model.cpu(), dummy, onnx_path, verbose=False) # CPU for export safe
        model.to(device)
        
        # 2. Multi-Epsilon Verification
        epsilons = [2/255, 4/255, 8/255]
        results = {}
        
        # Use simple timeout/samples from config or defaults
        timeout = config.get('verification', {}).get('final_timeout', 60)
        num_samples = config.get('verification', {}).get('num_samples', 100)
        
        for eps in epsilons:
            eps_str = f"{int(eps*255)}/255"
            print(f"    - eps={eps_str}...", end="", flush=True)
            
            # Using verify method from interface
            # Note: interface expects model object usually, but here we pass ONNX path handling inside class?
            # Actually ABCROWNVerifier.verify takes 'model' object to export ONNX.
            # But we already exported. 
            # Reviewing interface: It exports 'temp.onnx' internally.
            # To avoid redundancy, we can just call verifier.verify(model, ...) which re-exports 'temp.onnx'.
            # Or assume we rely on temp file.
            
            res = verifier.verify(model, (3,32,32), epsilon=eps, num_samples=num_samples, timeout=timeout)
            results[eps_str] = res['certified_accuracy']
            print(f" Acc={res['certified_accuracy']:.2%}")
            
        # Collect for Correlation
        lip_values.append(lip)
        cert_accs_2.append(results['2/255'])
        cert_accs_4.append(results['4/255'])
        cert_accs_8.append(results['8/255'])
        
        # Add SynFlow and Test Acc to record
        synflow = item.get('synflow', 0)
        test_acc = item.get('test_acc', 0)
        
        table_data.append([idx, lip, synflow, test_acc, results['2/255'], results['4/255'], results['8/255']])

    # Analysis
    if len(lip_values) > 1:
        log_lips = np.log(np.array(lip_values) + 1e-6) # Avoid log(0)
        r_2, _ = pearsonr(log_lips, cert_accs_2)
        r_4, _ = pearsonr(log_lips, cert_accs_4)
        r_8, _ = pearsonr(log_lips, cert_accs_8)
        
        print("\n--- Correlation Analysis (Log(L) vs Cert Acc) ---")
        print(f"eps=2/255: r={r_2:.3f}")
        print(f"eps=4/255: r={r_4:.3f}")
        print(f"eps=8/255: r={r_8:.3f}")
        
        # Create DataFrame and Save CSV
        columns = ["Arch Index", "Lipschitz", "SynFlow", "Test Acc", "Cert Acc (2/255)", "Cert Acc (4/255)", "Cert Acc (8/255)"]
        df = pd.DataFrame(table_data, columns=columns)
        
        csv_filename = f"certified_results_seed_{args.seed}.csv"
        csv_path = f"results/nsga2/{csv_filename}"
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        # Log to WandB
        wandb.log({
            "correlation/r_eps_2": r_2,
            "correlation/r_eps_4": r_4,
            "correlation/r_eps_8": r_8,
            "analysis/certified_table": wandb.Table(dataframe=df)
        })
        
        # Upload CSV to WandB
        wandb.save(csv_path)

    print("Certified Evaluation & Analysis Complete.")
    run.finish()

if __name__ == '__main__':
    main()

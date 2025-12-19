import argparse
import yaml
import torch
import os
import pickle
from src.search_space.nas_bench_201 import NASBench201Interface
from src.gatekeeper.api import FormalGatekeeper
from src.search.random_search import random_search_constrained

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/search_config.yaml')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    torch.manual_seed(args.seed)
    
    nas_interface = NASBench201Interface(dataset=config['dataset'])
    gatekeeper = FormalGatekeeper(config)
    
    # Baseline 4: Random + Constraint
    print(f"Running Random Search Baseline (Seed {args.seed})...")
    
    # N samples budget
    # E.g. same as NSGA-II budget = pop * gen
    budget = config['nsga2']['population_size'] * config['nsga2']['generations']
    
    results = random_search_constrained(
        nas_interface,
        gatekeeper,
        n_samples=budget,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    os.makedirs('results/random', exist_ok=True)
    with open(f'results/random/results_seed_{args.seed}.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main()

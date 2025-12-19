import argparse
import yaml
import torch
import os
import pickle
from src.search_space.nas_bench_201 import NASBench201Interface
from src.gatekeeper.api import FormalGatekeeper
from src.search.bayesian_opt import ConstrainedBayesianOpt

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
    
    print(f"Starting BO Search (Seed {args.seed})...")
    
    bo = ConstrainedBayesianOpt(
        nas_interface,
        gatekeeper,
        n_iterations=config['bayesian_opt']['n_iterations'],
        n_initial=config['bayesian_opt']['n_initial'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    X, Y = bo.search()
    
    # Save results
    os.makedirs('results/bo', exist_ok=True)
    results = {
        'X': X,
        'Y': Y
    }
    with open(f'results/bo/results_seed_{args.seed}.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main()

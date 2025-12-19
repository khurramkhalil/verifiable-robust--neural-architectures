import random
import torch
from tqdm import tqdm
from src.primitives.lipschitz_layers import compute_architectural_lipschitz_bound
from src.metrics.zero_cost_proxies import synflow_score

def random_search_constrained(search_space, gatekeeper, n_samples, device='cpu'):
    """
    Constrained Random Search.
    Samples architectures until n_samples feasible ones are found.
    Returns: List of dicts {index, score, model}
    """
    valid_architectures = []
    
    # We sample indices. NASBench201 has ~15k models.
    # We loop until we have enough.
    
    attempts = 0
    pbar = tqdm(total=n_samples, desc="Random Search")
    
    while len(valid_architectures) < n_samples and attempts < n_samples * 10:
        index = random.randint(0, 15624)
        
        # Build model to check feasibility
        model = search_space.get_model(index)
        
        if gatekeeper.is_statically_feasible(model):
            # Evaluate objectives (SynFlow)
            model.to(device)
            score = synflow_score(model, (3,32,32), device=device)
            
            valid_architectures.append({
                'index': index,
                'synflow': score,
                'model': model
            })
            pbar.update(1)
        
        attempts += 1
        
    pbar.close()
    
    # Sort by SynFlow (descending)
    valid_architectures.sort(key=lambda x: x['synflow'], reverse=True)
    
    return valid_architectures

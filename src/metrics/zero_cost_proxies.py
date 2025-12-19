import torch

def jacobian_norm_proxy(model, data_loader, device):
    """
    Computes the Jacobian Norm Proxy (lower is better for robustness).
    Approximated using random projections.
    """
    model.eval()
    try:
        inputs, _ = next(iter(data_loader))
    except (StopIteration, TypeError):
        # Handle cases where data_loader might be empty or mocked
        return float('inf')
        
    inputs = inputs.to(device).requires_grad_(True)
    
    outputs = model(inputs)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    
    # Random projection
    v = torch.randn_like(outputs)
    v = v / torch.norm(v, dim=1, keepdim=True)
    
    # Projected output
    proj_out = torch.sum(outputs * v)
    
    # Gradient w.r.t inputs
    # create_graph=False significantly speeds it up, but we might arguably need True if 
    # we were doing higher order meta-learning. Here just a proxy score.
    # However, to be safe for "backward" call later, we leave False.
    grads = torch.autograd.grad(proj_out, inputs, create_graph=False)[0]
    
    # Average norm over batch
    score = torch.norm(grads.view(grads.shape[0], -1), dim=1).mean().item()
    return score

def synflow_score(model, input_shape, device):
    """
    Computes SynFlow Score (higher is better for trainability).
    Sum of |weight * gradient| on all-ones input.
    """
    model.eval()
    # All ones input
    inputs = torch.ones((1, *input_shape)).to(device)
    
    # Zero gradients
    model.zero_grad()
    
    # Forward
    # We enforce linear activation or just sum output to keep gradients flowing
    # SynFlow paper suggests just summing output is fine.
    try:
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
    except Exception as e:
        # Fallback for models that might fail with dummy input
        print(f"SynFlow Error: {e}")
        return 0.0

    loss = torch.sum(outputs)
    loss.backward()
    
    # Sum |w * grad|
    metric = 0.0
    for p in model.parameters():
        if p.grad is not None:
            metric += torch.abs(p * p.grad).sum().item()
            
    return metric

def ntk_condition_number(model, data_loader, device, max_samples=64):
    """
    Estimates NTK condition number (lower is better).
    """
    # This is expensive, so we use a small batch.
    # We won't implement full NTK matrix here for simplicity unless requested.
    # Usually requires jacobian contraction.
    # Returning 0 for now as placeholder or simple proxy.
    return 0.0

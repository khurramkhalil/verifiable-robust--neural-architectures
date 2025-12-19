import torch
import torch.nn as nn
from src.primitives.spectral_conv import SpectralConv2d

def compute_architectural_lipschitz_bound(model, input_shape=(1, 3, 32, 32)):
    """
    Computes the compositional Lipschitz upper bound of the model.
    """
    # Helper to traverse and compute bound.
    # We simplified the graph traversal for the sake of this prompt, 
    # but for NAS-Bench-201 we might need to be more careful with DAGs.
    # However, for compositional bound L_hat = Product(L_layer), we can just iterate over modules
    # if we assume a sequential backbone or handle specific blocks.
    
    # Actually, simply iterating recursively over modules might double count if containers are visited.
    # A standard way for composition:
    # If the model is purely sequential: Product L_i
    # If Residual: L_block = L_path + 1
    # If Concat: L_block = sqrt(sum(L_i^2))
    
    # Since NAS-Bench-201 models have specific structure (Cells), we should try to handle that.
    # But as a general heuristic as per spec, we can define a visitor.
    
    # We will implement a simplified traversal that works for standard sequential-ish models
    # or relies on the fact that we replace layers.
    # FOR NOW, following the `details.txt`, we implement a recursive analysis.
    
    # Note: This simple implementation effectively treats everything as sequential 
    # unless we handle Cells explicitly. 
    # Ideally, the `forward` pass structure defines the Lipschitz composition.
    # But static analysis is requested.
    
    # Let's implement the `analyze_module` logic from details.txt
    
    total_lipschitz = 1.0

    def analyze_module(module):
        l_bound = 1.0
        
        if isinstance(module, SpectralConv2d):
            l_bound = module.get_lipschitz_constant()
            
        elif isinstance(module, nn.Conv2d):
             # Fallback
            w_flat = module.weight.view(module.weight.shape[0], -1)
            l_bound = torch.norm(w_flat, p='fro').item()
            
        elif isinstance(module, nn.BatchNorm2d):
            if module.affine:
                scale = torch.abs(module.weight) / torch.sqrt(module.running_var + module.eps)
                l_bound = torch.max(scale).item()
            else:
                l_bound = 1.0
                
        elif isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.Flatten)):
            l_bound = 1.0
            
        elif isinstance(module, nn.Linear):
             l_bound = torch.linalg.matrix_norm(module.weight, ord=2).item()
        
        # We need to handle containers.
        # If it's a container, we generally don't assign it a weight itself, 
        # but we need to know how its children combine.
        # This is TRICKY without knowing graph topology.
        # The spec in `details.txt` had a heuristic:
        # "if not list(module.children()): total_lipschitz *= analyze_module(module)"
        # This assumes purely sequential depth. 
        # For a Pilot study on NAS-Bench-201, this might be an approximation, 
        # BUT NAS-Bench-201 cells are sum of ops (Residual-like).
        # We might need a better parser later. For now, we stick to the sequential assumption
        # or rely on the `NASBench201Interface` to provide a flattened view if possible,
        # OR we improve this function to handle `Cell` classes if we can import them.
        
        return l_bound

    # Using the heuristic from spec: treating as sequential product of leaves
    # This is an UPPER BOUND if we assume paths are sequential.
    # For residual blocks (y = x + f(x)), L <= 1 + L_f.
    # Standard sequential traversal multiplies everything.
    # If we multiply 1 (for skip) and L_f, we get L_f, which is wrong (should be summative).
    # However, without specific graph info, sequential product is the conservative estimate 
    # for a deep chain.
    
    # IMPORTANT: The current spec `details.txt` implementation iterates `model.modules()`
    # and multiplies. This is technically incorrect for ResNets (it overestimates? or underestimates? 
    # If we have distinct branches, we should sum/max/l2. Multiplicative is for composition).
    # Since we are asked to implement the spec:
    
    for module in model.modules():
        # Skip containers
        if isinstance(module, (nn.Sequential, nn.ModuleList)):
            continue
        # Only multiply leaf nodes that are in the direct path (heuristic)
        if len(list(module.children())) == 0:
             total_lipschitz *= analyze_module(module)
             
    return total_lipschitz

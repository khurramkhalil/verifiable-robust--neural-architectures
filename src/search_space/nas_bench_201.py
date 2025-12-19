import torch.nn as nn
from xautodl.models import get_cell_based_tiny_net
from nats_bench import create
from src.primitives.spectral_conv import SpectralConv2d
import numpy as np

class NASBench201Interface:
    def __init__(self, dataset='cifar10', path=None):
        # Assuming path is None, it uses default NATS location or environment variable.
        # fast_mode=True loads the API faster, but might have less info.
        # Set verbose to False to reduce noise.
        self.api = create(path, 'tss', fast_mode=True, verbose=False)
        self.dataset = dataset

    def get_model(self, arch_index):
        arch_index = int(np.clip(arch_index, 0, 15624))
        config = self.api.get_net_config(arch_index, self.dataset)
        model = get_cell_based_tiny_net(config)
        self._inject_spectral_norm(model)
        return model
    
    def get_arch_str(self, arch_index):
        # Retrieve architecture string
        arch_index = int(np.clip(arch_index, 0, 15624))
        return self.api.arch(arch_index)

    def get_full_metrics(self, arch_index):
        # Retrieve all available "elaborative" data for publication
        arch_index = int(np.clip(arch_index, 0, 15624))
        info = self.api.get_more_info(arch_index, self.dataset)
        cost = self.api.get_cost_info(arch_index, self.dataset)
        
        return {
            "test_accuracy": info['test-accuracy'],
            "train_accuracy": info['train-accuracy'],
            "train_loss": info['train-loss'],
            "training_time_total": info['train-all-time'],
            "params_mb": float(cost['params']), # Usually in MB or Count, check NATS docs. NATS says 'params' in MB generally.
            "flops_mb": float(cost['flops']), # FLOPs in M
            "latency": float(cost['latency']) 
        }

    def _inject_spectral_norm(self, module):
        """
        Recursively replace nn.Conv2d with SpectralConv2d
        """
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                # Create replacement
                new_conv = SpectralConv2d(
                    child.in_channels, child.out_channels, child.kernel_size,
                    child.stride, child.padding, child.dilation, 
                    child.groups, child.bias is not None
                )
                # Copy weights
                new_conv.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    new_conv.bias.data = child.bias.data.clone()
                
                # Replace in parent
                setattr(module, name, new_conv)
            else:
                # Recurse
                self._inject_spectral_norm(child)

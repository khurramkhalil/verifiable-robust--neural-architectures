import unittest
import torch
import torch.nn as nn
from src.gatekeeper.api import FormalGatekeeper
from src.gatekeeper.trajectory_monitor import STLTrajectoryMonitor
from src.primitives.spectral_conv import SpectralConv2d

class TestGatekeeper(unittest.TestCase):
    def test_static_feasibility(self):
        # Configure Gatekeeper with a threshold of 10.0
        config = {'gatekeeper': {'lipschitz_threshold': 10.0}}
        gk = FormalGatekeeper(config)
        
        # Case 1: Feasible Model (Lipschitz < 10)
        # We construct a model with small weights.
        # SpectralConv2d(1, 1, 1) is a 1x1 convolution (scalar multiplication).
        # We set weight to 5.0, so Spectral Norm = 5.0.
        model_low = nn.Sequential(SpectralConv2d(1, 1, 1, bias=False))
        with torch.no_grad():
            model_low[0].weight.fill_(5.0)
        
        # Run forward pass to allow SpectralConv2d to estimate sigma via power iteration.
        # We run enough iterations or just one (defaults to 3 PI steps per forward).
        # 3 steps is usually enough for 1x1 scalar.
        model_low(torch.randn(1, 1, 5, 5))
        
        # Check feasibility
        self.assertTrue(gk.is_statically_feasible(model_low), 
                        f"Model with sigma ~5.0 should be feasible (Threshold 10.0). Actual L: {model_low[0].get_lipschitz_constant()}")
        
        # Case 2: Infeasible Model (Lipschitz > 10)
        # We set weight to 15.0, so Spectral Norm = 15.0.
        model_high = nn.Sequential(SpectralConv2d(1, 1, 1, bias=False))
        with torch.no_grad():
            model_high[0].weight.fill_(15.0)
            
        # Run forward pass to update sigma
        model_high(torch.randn(1, 1, 5, 5))
        
        self.assertFalse(gk.is_statically_feasible(model_high), 
                         f"Model with sigma ~15.0 should be infeasible (Threshold 10.0). Actual L: {model_high[0].get_lipschitz_constant()}")

    def test_stl_monitor(self):
        monitor = STLTrajectoryMonitor(degradation_tol=2.0, diversity_thresh=0.5)
        
        # History 0: Trivial
        h = [{'s_acc': 0, 's_rob': 10, 'diversity': 1.0}]
        rho = monitor.evaluate(h)
        self.assertTrue(rho > 0, f"History 0 Rho: {rho}")
        
        # History 1: Acc improves (0->1), Rob constant (10->10)
        h.append({'s_acc': 1, 's_rob': 10, 'diversity': 1.0})
        rho = monitor.evaluate(h)
        self.assertTrue(rho > 0, f"History 1 Rho: {rho}")
        
        # History 2: Acc improves (1->2), Rob degrades slightly (10->11) (Delta = -1)
        h.append({'s_acc': 2, 's_rob': 11, 'diversity': 1.0})
        rho = monitor.evaluate(h)
        self.assertTrue(rho > 0, f"History 2 Rho: {rho}")
        
        # History 3: Acc improves (2->3), Rob degrades significantly (11->15) (Delta = -4)
        # Violation: -4 is NOT >= -2.
        h.append({'s_acc': 3, 's_rob': 15, 'diversity': 1.0})
        rho = monitor.evaluate(h)
        self.assertTrue(rho < 0, f"History 3 Rho: {rho}")
        
        # History 4: Diversity Fail
        h = [{'s_acc': 0, 's_rob': 10, 'diversity': 0.1}, {'s_acc': 1, 's_rob': 10, 'diversity': 0.1}]
        rho = monitor.evaluate(h)
        self.assertTrue(rho < 0, f"History 4 Rho: {rho}")

if __name__ == '__main__':
    unittest.main()

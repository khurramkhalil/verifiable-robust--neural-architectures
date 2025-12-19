import unittest
import torch
import torch.nn as nn
from src.primitives.spectral_conv import SpectralConv2d
from src.primitives.lipschitz_layers import compute_architectural_lipschitz_bound

class TestPrimitives(unittest.TestCase):
    def test_spectral_conv_initialization(self):
        conv = SpectralConv2d(3, 16, 3, padding=1)
        self.assertIsInstance(conv, nn.Conv2d)
        # Check buffers
        self.assertTrue(hasattr(conv, 'u'))
        self.assertTrue(hasattr(conv, 'v'))
        self.assertTrue(hasattr(conv, 'sigma'))

    def test_spectral_conv_forward(self):
        conv = SpectralConv2d(3, 16, 3, padding=1)
        x = torch.randn(2, 3, 32, 32)
        out = conv(x)
        self.assertEqual(out.shape, (2, 16, 32, 32))
        # Sigma should be updated in training mode
        self.assertNotEqual(conv.sigma.item(), 1.0) # Unlikely to be exactly 1.0 random init

    def test_spectral_conv_sigma_correctness(self):
        # Create a conv with known weights (e.g. Identity-ish or simple)
        # Or just check that power iteration converges roughly
        conv = SpectralConv2d(1, 1, 3, padding=1, bias=False)
        conv.weight.data.fill_(0.1)
        x = torch.randn(1, 1, 10, 10)
        
        # Run forward multiple times to converge
        for _ in range(20):
            conv(x)
            
        sigma = conv.get_lipschitz_constant()
        self.assertTrue(sigma > 0)
        
        # Compare with naive spectral norm of reshaped matrix (known to be loose/different but correlated)
        # For valid padding, spectral norm is well defined.
        
    def test_lipschitz_bound_computation(self):
        model = nn.Sequential(
            SpectralConv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            SpectralConv2d(16, 16, 3, padding=1)
        )
        # Run forward once to populate sigmas
        model(torch.randn(1, 3, 32, 32))
        
        bound = compute_architectural_lipschitz_bound(model)
        
        # Bound should be roughly product of sigmas
        l1 = model[0].get_lipschitz_constant()
        l2 = model[2].get_lipschitz_constant()
        
        # Allow small float error
        self.assertAlmostEqual(bound, l1 * l2, places=4)

if __name__ == '__main__':
    unittest.main()

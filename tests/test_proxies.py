import unittest
import torch
import torch.nn as nn
from src.metrics.zero_cost_proxies import jacobian_norm_proxy, synflow_score
from src.metrics.diversity import population_diversity

class TestProxies(unittest.TestCase):
    def test_jacobian_proxy(self):
        model = nn.Sequential(nn.Conv2d(3, 3, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 10))
        # Create dummy loader
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 10, (4,))
        loader = [(x, y)]
        
        score = jacobian_norm_proxy(model, loader, 'cpu')
        self.assertIsInstance(score, float)
        self.assertTrue(score >= 0)

    def test_synflow(self):
        model = nn.Sequential(nn.Conv2d(3, 3, 3, padding=1))
        # Initialize weights to non-zero
        for p in model.parameters():
            p.data.fill_(1.0)
            
        score = synflow_score(model, (3, 32, 32), 'cpu')
        self.assertIsInstance(score, float)
        self.assertTrue(score > 0) # Should have gradients

    def test_diversity(self):
        pop = ["ABC", "ABD", "ZZZ"]
        # Dist(ABC, ABD) = 1
        # Dist(ABC, ZZZ) = 3
        # Dist(ABD, ZZZ) = 3
        # Avg = (1+3+3)/3 = 7/3 = 2.33
        
        d = population_diversity(pop)
        self.assertAlmostEqual(d, 2.3333, places=3)
        
        self.assertEqual(population_diversity(["AAA", "AAA"]), 0.0)

if __name__ == '__main__':
    unittest.main()

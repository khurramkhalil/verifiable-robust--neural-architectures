import unittest
import torch
import torch.nn as nn
import sys
import os

# Adapt path to import local scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from subscripts import ResNet20_definition
# Since execute scripts are not modules, I will verify the class logic by redefining/importing if possible
# or better, I will verify the standard training logic we modified.

from src.training.standard_training import train_standard, _evaluate

class MockModel(nn.Module):
    def __init__(self, tuple_output=False):
        super().__init__()
        self.tuple_output = tuple_output
        self.linear = nn.Linear(10, 2)
        
    def forward(self, x):
        out = self.linear(x)
        if self.tuple_output:
            return out, out # Return tuple
        return out

class TestBaselines(unittest.TestCase):
    
    def test_standard_training_tuple_handling(self):
        """Test that train_standard handles tuple outputs (logits, aux) correctly."""
        model = MockModel(tuple_output=True)
        # Create dummy loader
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        
        # Should not raise TypeError: cross_entropy ... tuple
        try:
            model, history = train_standard(model, loader, loader, epochs=1, device='cpu')
        except Exception as e:
            self.fail(f"train_standard failed with tuple output: {e}")
            
    def test_evaluate_tuple_handling(self):
        """Test that _evaluate handles tuple outputs correctly."""
        model = MockModel(tuple_output=True)
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        
        try:
            acc = _evaluate(model, loader, device='cpu')
        except Exception as e:
            self.fail(f"_evaluate failed with tuple output: {e}")

if __name__ == '__main__':
    unittest.main()

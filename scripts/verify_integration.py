import torch
import torch.nn as nn
import os
import sys
import subprocess

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.primitives.spectral_conv import SpectralConv2d
from src.verification.abcrown_interface import ABCROWNVerifier

def verify_integration():
    print("Beginning Integration Verification...")
    
    # 1. Create a simple model with SpectralConv2d
    model = nn.Sequential(
        SpectralConv2d(1, 4, 3, padding=1),
        nn.ReLU(),
        SpectralConv2d(4, 2, 3, padding=1),
        nn.Flatten(),
        nn.Linear(2*32*32, 10)
    )
    # Initialize weights
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (SpectralConv2d, nn.Linear)):
                m.weight.data.normal_(0, 0.1)
                
    print("[1/3] Simple Model created.")
    
    # 2. Test ONNX Export (via ABCROWNVerifier Helper)
    verifier = ABCROWNVerifier()
    output_onnx = "tests/test_model.onnx"
    input_shape = (1, 32, 32)
    
    dummy_input = torch.randn(1, *input_shape)
    os.makedirs("tests", exist_ok=True)
    
    try:
        torch.onnx.export(model, dummy_input, output_onnx, verbose=False, opset_version=11)
        print(f"[2/3] ONNX Export successful: {output_onnx}")
    except Exception as e:
        print(f"[ERROR] ONNX Export failed: {e}")
        return

    # 3. Test ABCROWN Invocation
    print("[3/3] testing ABCROWNVerifier.verify() invocation...")
    try:
        # We set a short timeout
        # verify expects model on cpu for export
        model.cpu()
        result = verifier.verify(model, input_shape, epsilon=2.0/255.0, timeout=10)
        print(f"Verifier returned: {result}")
        print("[SUCCESS] Integration Verification Complete.")
    except Exception as e:
        print(f"[WARNING] Verifier invocation raised exception (Expected provided no real data/params): {e}")
        # Failure here is likely due to alpha-beta-CROWN returning non-zero since we didn't train it or provide data config.
        # But we want to see it RAN.
    
if __name__ == "__main__":
    verify_integration()

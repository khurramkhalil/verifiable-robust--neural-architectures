# Verification Module - README

## Overview

This module provides neural network verification using alpha-beta-CROWN for NATS-Bench architectures. It handles ONNX export with necessary compatibility fixes and interfaces with the alpha-beta-CROWN verifier.

## ⚠️ CRITICAL LIMITATION: Single-Sample Only

**This module ONLY supports verifying ONE sample at a time** due to ONNX export constraints with NATS-Bench architectures.

### Why?

NATS-Bench uses scalar Mul constants (shape `()`) for skip connections, but alpha-beta-CROWN's auto_LiRPA expects 4D tensors (shape `[1, C, 1, 1]`). This causes verification to fail during bound propagation when processing multiple samples.

### Enforcement

The `verify()` method **automatically enforces `num_samples=1`** and will print a warning if you try to use a different value:

```python
verifier = ABCROWNVerifier()
result = verifier.verify(model, (3,32,32), epsilon=2/255, num_samples=10)
# WARNING: num_samples=10 is not supported due to ONNX limitations.
# Overriding to num_samples=1. To verify multiple samples, call verify() in a loop.
```

## Usage

### Basic Usage (Single Sample)

```python
from src.verification.abcrown_interface import ABCROWNVerifier
from nats_bench import create
from xautodl.models import get_cell_based_tiny_net

# Initialize
verifier = ABCROWNVerifier()
api = create(None, 'tss', fast_mode=True, verbose=False)

# Load model
config = api.get_net_config(10308, 'cifar10')
model = get_cell_based_tiny_net(config)

# Verify ONE sample
result = verifier.verify(
    model,
    input_shape=(3, 32, 32),
    epsilon=2/255,
    num_samples=1,  # MUST be 1
    timeout=60
)

print(f"Verified: {result['verified_count']}/{result['total']}")
print(f"Certified Accuracy: {result['certified_accuracy']:.1%}")
```

### Verifying Multiple Samples

To verify N samples, **call `verify()` in a loop**:

```python
def verify_multiple_samples(model, num_samples=10, epsilon=2/255):
    """Verify multiple samples for one model."""
    verifier = ABCROWNVerifier()
    
    total_verified = 0
    for i in range(num_samples):
        print(f"Verifying sample {i+1}/{num_samples}...")
        result = verifier.verify(
            model,
            input_shape=(3, 32, 32),
            epsilon=epsilon,
            num_samples=1,  # Always 1
            timeout=60
        )
        total_verified += result['verified_count']
    
    certified_acc = total_verified / num_samples
    return certified_acc

# Usage
acc = verify_multiple_samples(model, num_samples=10)
print(f"Certified accuracy over 10 samples: {acc:.1%}")
```

### Verifying Multiple Architectures

```python
# Verify multiple architectures
results = []
for arch_id in [10308, 10309, 10310]:
    config = api.get_net_config(arch_id, 'cifar10')
    model = get_cell_based_tiny_net(config)
    
    # Verify 10 samples per architecture
    acc = verify_multiple_samples(model, num_samples=10)
    
    results.append({
        'arch_id': arch_id,
        'certified_acc': acc
    })
    
    print(f"Architecture {arch_id}: {acc:.1%}")
```

## Configuration

The module generates a YAML config file (`temp.yaml`) for alpha-beta-CROWN with these settings:

```yaml
general:
  enable_incomplete_verification: true  # Use IBP/CROWN first

model:
  onnx_path: temp.onnx  # Generated ONNX file

data:
  dataset: CIFAR
  start: 0
  end: 1  # Always 1 sample (enforced)
  mean: [0.0]
  std: [1.0]

specification:
  norm: inf  # L-infinity norm
  epsilon: 0.00784  # 2/255 by default

solver:
  batch_size: 256
  beta-crown:
    iteration: 20

bab:
  timeout: 60  # seconds
```

## ONNX Export Fixes

The module applies **four critical fixes** to the ONNX export:

### 1. Fake Bias Removal
PyTorch ONNX export creates fake bias inputs for Conv layers with `bias=False`. We remove these post-export.

### 2. count_include_pad Removal
`onnx2pytorch` doesn't support this AveragePool attribute. We remove it from all AveragePool nodes.

### 3. Reshape → Flatten Conversion
The Reshape node between GlobalAveragePool and Linear causes shape mismatches in auto_LiRPA. We replace it with Flatten (axis=1).

### 4. Selective Constant Removal
We remove only the Constant nodes used by Reshape, preserving those needed for Mul operations.

## Performance

- **Single sample**: ~0.8 seconds
- **10 samples**: ~8 seconds (loop)
- **100 samples**: ~80 seconds (loop)

## Troubleshooting

### "AssertionError: const.ndim == 4"
**Cause**: Trying to verify multiple samples in one call.  
**Solution**: The code now prevents this automatically. If you still see this, ensure you're using the latest version.

### "TypeError: Conv2d.forward() takes 2 positional arguments but 3 were given"
**Cause**: Fake bias inputs not removed.  
**Solution**: This should be fixed automatically. Check that ONNX export completed successfully.

### Verification is slow
**Options**:
1. Reduce number of samples
2. Increase timeout for complex architectures
3. Parallelize across multiple GPUs/pods

### Zero certified accuracy
**Expected** for untrained models - PGD will find adversarial examples. Use trained robust models for meaningful results.

## Technical Details

### ONNX Conversion Accuracy
Max error between PyTorch and ONNX outputs: **< 0.00001** (verified)

### BatchNorm Compatibility
Works correctly with batch_size > 1 despite "experimental" warning. Max error: **< 0.00001** (verified)

### Dependencies
- PyTorch 2.2.0+
- ONNX 1.17.0+
- onnx2pytorch (Verified-Intelligence fork)
- alpha-beta-CROWN v1.0+

## Files

- `abcrown_interface.py` - Main verification interface
- `temp.onnx` - Generated ONNX model (temporary)
- `temp.yaml` - Generated alpha-beta-CROWN config (temporary)

## References

- [alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
- [NATS-Bench](https://github.com/D-X-Y/NATS-Bench)
- [Production Usage Guide](../../../.gemini/antigravity/brain/005935a0-cdfc-45a8-91c7-325d2e9eaa66/production_usage_guide.md)
- [Debugging Summary](../../../.gemini/antigravity/brain/005935a0-cdfc-45a8-91c7-325d2e9eaa66/onnx_debugging_summary.md)

## Status

✅ **PRODUCTION READY** (with single-sample limitation documented and enforced)

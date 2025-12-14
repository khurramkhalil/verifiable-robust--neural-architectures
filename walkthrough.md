# Walkthrough: Verifiable Synthesis of Robust Neural Architectures

I have implemented the complete "General Framework for Verifiable Synthesis of Robust Neural Architectures" according to the specification.

## Core Components Implemented

### 1. Primitives (`src/primitives/`)
- **[SpectralConv2d](file:///Users/khurram/verifiable-robust--neural-architectures/src/primitives/spectral_conv.py)**: A custom convolutional layer that tracks its spectral norm using power iteration on the *convolution operator* itself, correctly handling zero-padding (unlike `torch.nn.utils.spectral_norm`).
- **[Lipschitz Composition](file:///Users/khurram/verifiable-robust--neural-architectures/src/primitives/lipschitz_layers.py)**: `compute_architectural_lipschitz_bound` recursively computes the upper bound $\hat{L}(A)$ of the network.

### 2. Neuro-Symbolic Gatekeeper (`src/gatekeeper/`)
- **[FormalGatekeeper](file:///Users/khurram/verifiable-robust--neural-architectures/src/gatekeeper/api.py)**: Enforces static constraints (Lipschitz threshold) and monitors search trajectory.
- **[STLTrajectoryMonitor](file:///Users/khurram/verifiable-robust--neural-architectures/src/gatekeeper/trajectory_monitor.py)**: Uses Signal Temporal Logic (via `rtamt`) to enforce "Bounded Degradation" and "Diversity Maintenance" across generations.

### 3. Search Algorithms (`src/search/`)
- **[NSGA-II](file:///Users/khurram/verifiable-robust--neural-architectures/src/search/nsga2.py)**: Multi-objective evolutionary search (Minimize Lipschitz, Maximize SynFlow) coupled with the Gatekeeper.
- **[Bayesian Optimization](file:///Users/khurram/verifiable-robust--neural-architectures/src/search/bayesian_opt.py)**: BoTorch-based implementation using qNEHVI with constraints.
- **[Random Search](file:///Users/khurram/verifiable-robust--neural-architectures/src/search/random_search.py)**: Constrained random baseline.

### 4. Zero-Cost Proxies (`src/metrics/`)
- **[zero_cost_proxies.py](file:///Users/khurram/verifiable-robust--neural-architectures/src/metrics/zero_cost_proxies.py)**: Implements Jacobian Norm (robustness proxy) and SynFlow (trainability proxy).

### 5. Verification Interface (`src/verification/`)
- **[ABCROWNVerifier](file:///Users/khurram/verifiable-robust--neural-architectures/src/verification/abcrown_interface.py)**: Interacts with an external `alpha-beta-CROWN` installation via ONNX export and config generation to provide certified accuracy.

## How to Verify

### 1. Pilot Study (Correlation Validation)
The critical first step is to validate the correlation between our Lipschitz bound and certified robustness.
```bash
python scripts/pilot_study.py --config config/pilot_study.yaml
```
*Note: This requires `nats_bench` data and `alpha-beta-CROWN` to be accessible.*

### 2. Run Search Experiments
To run the full search using NSGA-II:
```bash
python scripts/run_search_nsga2.py --config config/search_config.yaml --seed 0
```

To run Bayesian Optimization:
```bash
python scripts/run_search_bo.py --config config/search_config.yaml --seed 0
```

### 3. Analyze Results
After running experiments, analyze the Pareto fronts and Hypervolume:
```bash
python scripts/analyze_results.py
```

## Next Steps
1. **Pilot Execution**: Run the pilot study to confirm the parameter $K$ (Lipschitz threshold).
2. **Threshold Update**: Update `config/search_config.yaml` with the $K$ found in the pilot.
3. **Full Run**: Execute the full search suite across 10 seeds.

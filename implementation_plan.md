# Implementation Plan: Verifiable Synthesis of Robust Neural Architectures

## Goal Description
Implement a "General Framework for Verifiable Synthesis of Robust Neural Architectures" that integrates formal verification (Lipschitz bounds, STL monitoring) into the Neural Architecture Search (NAS) process.

## User Review Required
> [!IMPORTANT]
> - **External Dependency**: `alpha-beta-CROWN` needs to be cloned locally. The plan assumes it will be available or installed.
> - **Compute Resources**: The full search requires significant GPU resources.
> - **NAS-Bench-201**: The `nats_bench` library and dataset are required.

## Proposed Changes

### Environment & Dependencies
#### [NEW] [requirements.txt](file:///Users/khurram/verifiable-robust--neural-architectures/requirements.txt)
- List all Python dependencies (torch, nats_bench, botorch, rtamt, etc.)

### Phase 1: Formal Compositional Analysis (Layer 1)
#### [NEW] [src/primitives/spectral_conv.py](file:///Users/khurram/verifiable-robust--neural-architectures/src/primitives/spectral_conv.py)
- `SpectralConv2d`: Custom Conv2d layer with power iteration on the convolution operator (handling zero-padding correctly) to estimate spectral norm.

#### [NEW] [src/primitives/lipschitz_layers.py](file:///Users/khurram/verifiable-robust--neural-architectures/src/primitives/lipschitz_layers.py)
- `compute_architectural_lipschitz_bound`: Recursive function to compute the compositional Lipschitz upper bound of a model.

### Phase 2: Gatekeeper API (Layer 2)
#### [NEW] [src/gatekeeper/api.py](file:///Users/khurram/verifiable-robust--neural-architectures/src/gatekeeper/api.py)
- `FormalGatekeeper`: Class to check static feasibility (Lipschitz bound) and trajectory robustness (STL).

#### [NEW] [src/gatekeeper/trajectory_monitor.py](file:///Users/khurram/verifiable-robust--neural-architectures/src/gatekeeper/trajectory_monitor.py)
- `STLTrajectoryMonitor`: Implements STL properties (Bounded Degradation, Diversity Maintenance) using `rtamt`.

### Phase 3: Zero-Cost Proxies & Metrics
#### [NEW] [src/metrics/zero_cost_proxies.py](file:///Users/khurram/verifiable-robust--neural-architectures/src/metrics/zero_cost_proxies.py)
- `jacobian_norm_proxy`: Estimates robustness.
- `synflow_score`: Estimates trainability.
- `ntk_condition_number`: Estimates optimization curvature.

#### [NEW] [src/metrics/diversity.py](file:///Users/khurram/verifiable-robust--neural-architectures/src/metrics/diversity.py)
- `population_diversity`: Computes average pairwise edit distance between architecture strings.

### Phase 4: Search Space Interface
#### [NEW] [src/search_space/nas_bench_201.py](file:///Users/khurram/verifiable-robust--neural-architectures/src/search_space/nas_bench_201.py)
- `NASBench201Interface`: Wrapper around `nats_bench` to load models and inject `SpectralConv2d`.

### Phase 5: Training Protocols
#### [NEW] [src/training/pgd_training.py](file:///Users/khurram/verifiable-robust--neural-architectures/src/training/pgd_training.py)
- `pgd_attack`: Generates adversarial examples.
- `train_pgd_at`: Training loop with PGD.

#### [NEW] [src/training/standard_training.py](file:///Users/khurram/verifiable-robust--neural-architectures/src/training/standard_training.py)
- `train_standard`: Standard training loop for baselines.

### Phase 6: Verification Interface
#### [NEW] [src/verification/abcrown_interface.py](file:///Users/khurram/verifiable-robust--neural-architectures/src/verification/abcrown_interface.py)
- `ABCROWNVerifier`: Interface to run the external `alpha-beta-CROWN` verifier via ONNX export and subprocess calls.

### Phase 7: Search Algorithms
#### [NEW] [src/search/nsga2.py](file:///Users/khurram/verifiable-robust--neural-architectures/src/search/nsga2.py)
- `ConstrainedNSGA2`: NSGA-II algorithm with Gatekeeper integration.

#### [NEW] [src/search/bayesian_opt.py](file:///Users/khurram/verifiable-robust--neural-architectures/src/search/bayesian_opt.py)
- `ConstrainedBayesianOpt`: BoTorch-based Bayesian Optimization with constraints.

#### [NEW] [src/search/random_search.py](file:///Users/khurram/verifiable-robust--neural-architectures/src/search/random_search.py)
- `random_search_constrained`: Baseline random search with Gatekeeper filtering.

### Phase 8: Pilot Study
#### [NEW] [config/pilot_study.yaml](file:///Users/khurram/verifiable-robust--neural-architectures/config/pilot_study.yaml)
- Configuration for the pilot study.

#### [NEW] [scripts/pilot_study.py](file:///Users/khurram/verifiable-robust--neural-architectures/scripts/pilot_study.py)
- Script to run the pilot study (correlation analysis).

### Phase 9: Full Search Experiments
#### [NEW] [config/search_config.yaml](file:///Users/khurram/verifiable-robust--neural-architectures/config/search_config.yaml)
- Configuration for the full search.

#### [NEW] [scripts/run_search_nsga2.py](file:///Users/khurram/verifiable-robust--neural-architectures/scripts/run_search_nsga2.py)
#### [NEW] [scripts/run_search_bo.py](file:///Users/khurram/verifiable-robust--neural-architectures/scripts/run_search_bo.py)
#### [NEW] [scripts/run_baselines.py](file:///Users/khurram/verifiable-robust--neural-architectures/scripts/run_baselines.py)
- Scripts to run the respective search algorithms.

### Phase 10: Statistical Analysis & Visualization
#### [NEW] [scripts/analyze_results.py](file:///Users/khurram/verifiable-robust--neural-architectures/scripts/analyze_results.py)
- Script to analyze results, compute hypervolumes, and generate plots.

## Verification Plan

### Automated Tests
- [ ] Run `scripts/pilot_study.py` to validate the core hypothesis (Correlation > 0.3).
- [ ] Run unit tests (to be created in `tests/`) for primitives like `SpectralConv2d` and proxies.

### Manual Verification
- [ ] Verify that ONNX export works for a few sample architectures.
- [ ] Verify that `abcrown.py` can be invoked and returns results.

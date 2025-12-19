#!/bin/bash
set -e  # Exit on error

echo "========================================================"
echo "🚀 STARTING DEBUG PIPELINE (LOCAL VERIFICATION)"
echo "========================================================"

# 1. Main Constrained Search (Debug)
echo ""
echo "--- [1/6] Running Constrained NSGA-II Search (Debug) ---"
python scripts/run_search_nsga2.py --seed 0 --config config/debug_pipeline.yaml

# 2. Unconstrained Baseline (Debug)
echo ""
echo "--- [2/6] Running Unconstrained Search (Debug) ---"
python scripts/run_search_nsga2.py --seed 0 --config config/debug_pipeline.yaml --no_constraint

# 3. ResNet-20 Baseline (Debug)
echo ""
echo "--- [3/6] Running ResNet-20 Baseline (Debug) ---"
python scripts/run_baseline_resnet.py --seed 0 --debug --epochs 1 --batch_size 4

# 4. Random Baseline (Debug)
echo ""
echo "--- [4/6] Running Random Baseline (Debug) ---"
python scripts/run_baseline_random.py --seed 0 --debug --count 2 --epochs 1

# 5. Consolidate Results (Debug)
echo ""
echo "--- [5/7] Consolidating Results (Master CSV) ---"
python scripts/consolidate_results.py --seed 0 --results_dir results

# 6. Statistical Analysis (Debug)
echo ""
echo "--- [6/7] Performing Statistical Analysis ---"
python scripts/analyze_statistics.py results/MASTER_RESULTS_RUN_0.csv

# 7. Generate Figures (Debug)
echo ""
echo "--- [7/7] Generating Publication Figures ---"
python scripts/generate_figures.py \
    results/MASTER_RESULTS_RUN_0.csv \
    --output_dir results/debug_figures

echo "========================================================"
echo "✅ DEBUG PIPELINE COMPLETE. Check results/debug_figures/"
echo "========================================================"

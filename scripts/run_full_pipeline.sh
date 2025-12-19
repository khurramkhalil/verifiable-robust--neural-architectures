#!/bin/bash
set -e  # Exit on error

echo "========================================================"
echo "🚀 STARTING VERIFIABLE ROBUST NAS: FULL PIPELINE"
echo "========================================================"

# 1. Main Constrained Search (Priority 1)
echo ""
echo "--- [1/6] Running Constrained NSGA-II Search (Main) ---"
python scripts/run_search_nsga2.py --seed 0 --config config/search_config.yaml

# 2. Unconstrained Baseline (Priority 2)
echo ""
echo "--- [2/6] Running Unconstrained Search (Baseline 1) ---"
python scripts/run_search_nsga2.py --seed 0 --config config/search_config.yaml --no_constraint

# 3. ResNet-20 Baseline (Priority 2)
echo ""
echo "--- [3/6] Running ResNet-20 Baseline (Baseline 2) ---"
python scripts/run_baseline_resnet.py --seed 0

# 4. Random Baseline (Priority 2)
echo ""
echo "--- [4/6] Running Random Baseline (Baseline 3) ---"
python scripts/run_baseline_random.py --seed 0

# 5. Statistical Analysis (Priority 3)
echo ""
echo "--- [5/7] Performing Statistical Analysis ---"
# Using defaults since schemas are now standardized
python scripts/analyze_statistics.py results/nsga2/certified_results_seed_0.csv

# 6. Generate Figures (Priority 4)
echo ""
# 6. Consolidate Results (Priority 5)
echo ""
echo "--- [6/7] Consolidating Results (Master CSV) ---"
python scripts/consolidate_results.py --seed 0

# 7. Generate Figures (Priority 4)
# Now uses the master file generated in step 6
echo ""
echo "--- [7/7] Generating Publication Figures ---"
python scripts/generate_figures.py \
    results/MASTER_RESULTS_RUN_0.csv \
    --output_dir results/figures

echo "========================================================"
echo "✅ PIPELINE COMPLETE. Artifacts available in results/"
echo "========================================================"

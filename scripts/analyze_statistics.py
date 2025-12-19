import argparse
import pandas as pd
import numpy as np
import scipy.stats as stats
import sys
import os

def identify_pareto_front(objectives):
    """
    Finds the Pareto front for minimization of all objectives.
    objectives: (n_samples, n_objectives) numpy array
    Returns: Boolean array, True if Pareto optimal.
    """
    population_size = objectives.shape[0]
    pareto_front = np.ones(population_size, dtype=bool)
    for i in range(population_size):
        for j in range(population_size):
            if all(objectives[j] <= objectives[i]) and any(objectives[j] < objectives[i]):
                pareto_front[i] = False
                break
    return pareto_front

def main():
    parser = argparse.ArgumentParser(description="Statistical Analysis of NAS Results")
    parser.add_argument('csv_file', type=str, help="Path to wandb export or results CSV")
    parser.add_argument('--synflow_col', type=str, default='SynFlow', help='Column name for SynFlow')
    parser.add_argument('--lip_col', type=str, default='Lipschitz', help='Column name for Lipschitz')
    parser.add_argument('--acc_col', type=str, default='Test Acc', help='Column name for Test Accuracy')
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: File {args.csv_file} not found.")
        return

    df = pd.read_csv(args.csv_file)
    print(f"Loaded {len(df)} rows from {args.csv_file}")
    
    # 10. Analyze existing test accuracy vs Lipschitz
    if args.lip_col in df and args.acc_col in df:
        # Correlation test
        # We drop NaNs to ensure clean stat
        sub = df.dropna(subset=[args.lip_col, args.acc_col])
        r, p = stats.pearsonr(sub[args.lip_col], sub[args.acc_col])
        print(f"\n10. Lipschitz vs Test Acc:")
        print(f"  r={r:.3f}, p={p:.4f}")
        if r > 0:
            print("  -> Positive correlation (Expected: Smooth models train better).")
        else:
            print("  -> Negative correlation.")
    else:
        print(f"\nSkipping Lipschitz analysis (Cols missing: {args.lip_col}, {args.acc_col})")

    # 11. SynFlow validation
    if args.synflow_col in df and args.acc_col in df:
        sub = df.dropna(subset=[args.synflow_col, args.acc_col])
        # Log10 transform as requested, handling potentialzeros
        syn_scores = sub[args.synflow_col].values
        # Ensure positive for log
        syn_log = np.log10(np.maximum(syn_scores, 0) + 1)
        
        r_syn, p_syn = stats.pearsonr(syn_log, sub[args.acc_col])
        print(f"\n11. SynFlow vs Test Acc:")
        print(f"  r={r_syn:.3f}, p={p_syn:.4f}")
        
        if r_syn > 0.5:
            print("  -> SynFlow is a good proxy ✅")
        elif r_syn < 0.3:
            print("  -> SynFlow failed as proxy ❌")
        else:
            print("  -> SynFlow is a weak proxy (0.3 < r < 0.5)")
    else:
         print(f"\nSkipping SynFlow analysis (Cols missing: {args.synflow_col}, {args.acc_col})")
         
    # 12. Multi-objective analysis
    # Objectives: Maximize Test Accuracy -> Minimize -Test Accuracy
    #             Minimize Lipschitz -> Minimize Lipschitz
    if args.lip_col in df and args.acc_col in df:
        objectives = np.column_stack([
            -df[args.acc_col].values, # Minimize negative acc
            df[args.lip_col].values    # Minimize Lipschitz
        ])
        
        is_pareto = identify_pareto_front(objectives)
        pareto_count = np.sum(is_pareto)
        
        print(f"\n12. Multi-Objective Analysis:")
        print(f"  Pareto Optimal Models: {pareto_count}/{len(df)}")
        
        best_acc_idx = df[args.acc_col].idxmax()
        best_model = df.iloc[best_acc_idx]
        print(f"  Best Accuracy Model: Acc={best_model[args.acc_col]:.2f}%, Lip={best_model[args.lip_col]:.4f}")
        print(f"  Is Best Acc model Pareto Optimal? {is_pareto[best_acc_idx]}")
        
    print("\nStatistical Analysis Complete.")

if __name__ == '__main__':
    main()

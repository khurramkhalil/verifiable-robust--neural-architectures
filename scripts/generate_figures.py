import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_lip_vs_acc(df_main, output_dir):
    """
    13. Scatter plot: Lipschitz vs Test Accuracy (Color by SynFlow)
    """
    if df_main.empty: return

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Ensure columns exist
    if 'SynFlow' not in df_main.columns:
        df_main['SynFlow'] = 0
        
    # Log transform SynFlow for better color scale
    if 'Log SynFlow' not in df_main.columns:
        df_main.loc[:, 'Log SynFlow'] = np.log10(np.maximum(df_main['SynFlow'], 1e-6) + 1)

    # Filter outliers (ResNet baseline with huge Lipschitz) for visualization
    plot_df = df_main[df_main['Lipschitz'] < 1000].copy()
    
    sns.scatterplot(
        data=plot_df, 
        x='Lipschitz', 
        y='Test Acc', 
        hue='Log SynFlow',
        palette='viridis', 
        size='Log SynFlow',
        sizes=(20, 200),
        alpha=0.8
    )
    
    # Annotate best architecture
    if not df_main.empty:
        best_acc_idx = df_main['Test Acc'].idxmax()
        best_model = df_main.loc[best_acc_idx]
        plt.annotate(
            f"Best Acc: {best_model['Test Acc']:.2f}%",
            (best_model['Lipschitz'], best_model['Test Acc']),
            xytext=(10, 10), textcoords='offset points',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
        )
    
    plt.title("Lipschitz Constant vs Test Accuracy")
    plt.xlabel("Lipschitz Bound (Lower is smoother)")
    plt.ylabel("Test Accuracy (%)")
    plt.savefig(os.path.join(output_dir, '13_lipschitz_vs_test_acc.png'), dpi=300)
    plt.close()
    print(f"Generated 13_lipschitz_vs_test_acc.png")

def plot_lip_vs_certified(df_main, output_dir):
    """
    14. Scatter plot: Lipschitz vs Certified Accuracy (MAIN RESULT)
    """
    if df_main.empty: return
    
    plt.figure(figsize=(10, 6))
    
    # Melt dataframe for multiple epsilons
    eps_cols = [c for c in df_main.columns if 'Cert Acc' in c]
    if not eps_cols:
        print("Skipping Certified Plot: No 'Cert Acc' columns found.")
        return

    melted = df_main.melt(
        id_vars=['Lipschitz', 'Test Acc'], 
        value_vars=eps_cols,
        var_name='Epsilon', 
        value_name='Certified Accuracy'
    )
    
    # Filter outliers
    melted = melted[melted['Lipschitz'] < 1000]
    
    sns.scatterplot(
        data=melted,
        x='Lipschitz',
        y='Certified Accuracy',
        hue='Epsilon',
        style='Epsilon',
        s=100,
        alpha=0.8
    )
    
    plt.title("Certified Robustness vs Lipschitz Constant")
    plt.xlabel("Lipschitz Bound")
    plt.ylabel("Certified Accuracy (%)")
    plt.legend(title='Epsilon')
    plt.savefig(os.path.join(output_dir, '14_lipschitz_vs_certified_acc.png'), dpi=300)
    plt.close()
    print(f"Generated 14_lipschitz_vs_certified_acc.png")

def plot_pareto_comparison(df_all, output_dir):
    """
    15. Pareto front comparison: Constrained vs Unconstrained
    """
    # Filter methods
    df = df_all[df_all['Method'].isin(['Constrained Search', 'Unconstrained Search'])]
    if df.empty:
        print("Skipping Pareto Comparison: No search data found.")
        return
    
    plt.figure(figsize=(10, 6))
    
    sns.scatterplot(
        data=df,
        x='Lipschitz',
        y='Test Acc',
        hue='Method',
        style='Method',
        s=100
    )
    
    plt.title("Pareto Front Comparison: Constrained vs Unconstrained Search")
    plt.xlabel("Lipschitz Bound")
    plt.ylabel("Test Accuracy (%)")
    plt.savefig(os.path.join(output_dir, '15_pareto_comparison.png'), dpi=300)
    plt.close()
    print(f"Generated 15_pareto_comparison.png")

def plot_ablation(df_all, output_dir):
    """
    16. Ablation study visualization
    """
    plt.figure(figsize=(10, 6))
    
    # Map method names for clarity if needed, or just use raw
    # Methods: Constrained Search, Unconstrained Search, ResNet-20 Baseline, Random Baseline
    
    # Test Acc boxplot
    sns.boxplot(x='Method', y='Test Acc', data=df_all) 
    plt.title("Ablation Study: Test Accuracy Distribution")
    plt.ylabel("Test Accuracy (%)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '16_ablation_study.png'), dpi=300)
    plt.close()
    print(f"Generated 16_ablation_study.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_csv', type=str, help="Master results CSV")
    parser.add_argument('--output_dir', type=str, default='figures')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if os.path.exists(args.results_csv):
        df_all = pd.read_csv(args.results_csv)
        
        # Subsets
        df_constrained = df_all[df_all['Method'] == 'Constrained Search']
        
        # 13.
        plot_lip_vs_acc(df_constrained, args.output_dir)
        
        # 14.
        plot_lip_vs_certified(df_constrained, args.output_dir)
        
        # 15.
        plot_pareto_comparison(df_all, args.output_dir)
        
        # 16.
        plot_ablation(df_all, args.output_dir)
        
    else:
        print(f"Results file {args.results_csv} not found.")

if __name__ == '__main__':
    main()

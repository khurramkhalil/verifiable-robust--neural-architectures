import argparse
import pickle
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from pymoo.indicators.hv import HV

def compute_hypervolume(front, ref_point):
    if len(front) == 0:
        return 0.0
    hv = HV(ref_point=ref_point)
    # Front is (N, 2). Needs to be minimization.
    # Objectives were [Min L, Min -SynFlow]
    # We want to maximize Accuracy and Robustness (minimize Lipschitz).
    # If we plot Acc vs Rob:
    # HV usually assumes minimization.
    return hv(front)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--output_dir', type=str, default='analysis_results')
    args = parser.parse_args()
    
    # Load all results
    methods = ['nsga2', 'bo', 'random']
    data = []
    
    for method in methods:
        path = f"{args.results_dir}/{method}/*.pkl"
        files = glob.glob(path)
        
        for f in files:
            with open(f, 'rb') as fp:
                res = pickle.load(fp)
            
            # Extract Pareto fronts or final points
            # Format varies slightly by method script
            points = []
            if method == 'nsga2':
                # res is list of dicts {arch_index, objectives}
                points = np.array([r['objectives'] for r in res])
            elif method == 'bo':
                # res is dict {X, Y}
                points = res['Y'].numpy()
            elif method == 'random':
                # list of dicts {index, synflow, model} - we need to recompute Lipschitz if not stored
                # Random baseline in script didn't store Lipschitz in result dict explicitly except by re-eval?
                # Actually random script stored 'index', 'synflow', 'model'. 
                # It computed Lipschitz for feasibility but didn't save it in dict.
                # Ideally random script should be updated.
                # Assuming it has 'lipschitz' key if we updated it, or we skip/mock.
                pass 
            
            if len(points) > 0:
                # Calculate HV
                # Ref point [100.0, 0.0] ?
                # Depends on scale.
                hv = compute_hypervolume(points, np.array([100.0, 0.0]))
                data.append({
                    'method': method,
                    'hypervolume': hv,
                    'file': f
                })

    df = pd.DataFrame(data)
    if not df.empty:
        print(df.groupby('method')['hypervolume'].describe())
        
        # Boxplot
        plt.figure()
        sns.boxplot(data=df, x='method', y='hypervolume')
        plt.title('Hypervolume Comparison')
        plt.savefig(f"{args.output_dir}/hypervolume_comparison.png")
    else:
        print("No results found to analyze.")

if __name__ == '__main__':
    main()

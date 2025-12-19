import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.repair.rounding import RoundingRepair
from src.metrics.zero_cost_proxies import synflow_score
from src.primitives.lipschitz_layers import compute_architectural_lipschitz_bound
from src.metrics.diversity import population_diversity
import torch
import wandb

class NASProblem(ElementwiseProblem):
    def __init__(self, search_space, gatekeeper, device):
        # 1 variable: architecture index
        # We model it as continuous integer for pymoo, round it later
        # Range 0 to 15624
        super().__init__(n_var=1, n_obj=2, n_ieq_constr=0, xl=0, xu=15624)
        self.search_space = search_space
        self.gatekeeper = gatekeeper
        self.device = device
        self.history = []

    def _evaluate(self, x, out, *args, **kwargs):
        index = int(np.clip(np.round(x[0]), 0, 15624))
        model = self.search_space.get_model(index)
        
        # Constraints: Statically feasible
        if not self.gatekeeper.is_statically_feasible(model):
            # Peninsula: Bad fitness
            out["F"] = [1e10, 1e10] # Min Lipschitz, Max SynFlow (Min -SynFlow)
            return

        # Objectives:
        # 1. Minimize Lipschitz (rob_proxy)
        # 2. Maximize SynFlow (acc_proxy) -> Minimize -SynFlow
        
        l_bound = compute_architectural_lipschitz_bound(model)
        
        model.to(self.device)
        syn = synflow_score(model, (3,32,32), self.device)
        
        out["F"] = [l_bound, -syn]
        
        # Store for diversity calc?
        # Pymoo handles population logic.
        
class ConstrainedNSGA2:
    def __init__(self, search_space, gatekeeper, pop_size=24, generations=15, device='cpu'):
        self.search_space = search_space
        self.gatekeeper = gatekeeper
        self.pop_size = pop_size
        self.generations = generations
        self.device = device
        
    def search(self):
        problem = NASProblem(self.search_space, self.gatekeeper, self.device)
        
        algorithm = NSGA2(
            pop_size=self.pop_size,
            sampling=IntegerRandomSampling(),
            crossover=SBX(),
            mutation=PolynomialMutation(repair=RoundingRepair()),
            eliminate_duplicates=True
        )
        
        # Custom callback to handle STL trajectory and diversity
        # Since pymoo 0.6 has a callback interface
        
        class STLCallback:
            def __init__(self, gatekeeper, interface):
                self.gatekeeper = gatekeeper
                self.interface = interface
                
            def __call__(self, algorithm):
                # Retrieve population
                pop_indices = [int(np.clip(np.round(ind.X[0]), 0, 15624)) for ind in algorithm.pop]
                pop_objs = algorithm.pop.get("F")
                
                # Get architecture strings for diversity
                arch_strs = [self.interface.get_arch_str(i) for i in pop_indices]
                div_score = population_diversity(arch_strs)
                
                # Metrics for history: 
                # acc = max SynFlow (min -SynFlow, so min of col 1, negated)
                # rob = min Lipschitz (min of col 0)
                
                # F needs to be collected properly.
                # If some are infeasible, their F is 1e10. 
                # We filter feasible ones.
                
                feasible_mask = pop_objs[:, 0] < 1e9
                if not np.any(feasible_mask):
                    # All infeasible?
                     metrics = {'s_acc': 0, 's_rob': 1000, 'diversity': div_score}
                else:
                    # Zip indices with objects
                    feasible_indices = np.array(pop_indices)[feasible_mask]
                    feasible_values = pop_objs[feasible_mask]
                    
                    # Create tuple (obj0, obj1, index)
                    feasible_triplets = []
                    for val, idx in zip(feasible_values, feasible_indices):
                        feasible_triplets.append([val[0], val[1], idx])
                    feasible_triplets = np.array(feasible_triplets)

                    best_syn = -np.min(feasible_triplets[:, 1]) # Max SynFlow
                    best_rob = np.min(feasible_triplets[:, 0]) # Min Lipschitz
                    
                    metrics = {
                        's_acc': best_syn,
                        's_rob': best_rob,
                        'diversity': div_score
                    }
                
                # Update trajectory
                robustness = self.gatekeeper.update_and_check_trajectory(metrics)
                
                if robustness < 0:
                    print(f"STL Violation detected! Rho: {robustness}")

                if self.gatekeeper.wandb_run:
                    # Log metrics
                    
                    # Retrieve full elaborative data for logging/publication
                    if 'feasible_indices' in locals():
                         full_metrics_list = [self.interface.get_full_metrics(idx) for idx in feasible_indices]
                         feasible_objs = feasible_triplets # Use triplets which have [lip, -syn, idx]
                    else:
                         full_metrics_list = []
                         feasible_objs = []
                    
                    if len(feasible_objs) == len(full_metrics_list):
                        scatter_data = []
                        for o, m in zip(feasible_objs, full_metrics_list):
                            scatter_data.append([
                                int(o[2]),      # Architecture Index (New)
                                o[0],           # Lipschitz
                                -o[1],          # SynFlow
                                m['test_accuracy'],
                                m['train_accuracy'],
                                m['train_loss'],
                                m['params_mb'],
                                m['flops_mb'],
                                m['latency'],
                                m['training_time_total']
                            ])
                    else:
                        scatter_data = []
                    
                    if scatter_data:
                        columns = [
                            "Arch Index", 
                            "Lipschitz Bound", "SynFlow Score", 
                            "Test Accuracy", "Train Accuracy", "Train Loss",
                            "Params (MB)", "FLOPs (M)", "Latency (s)", "Training Time (s)"
                        ]
                        table = wandb.Table(data=scatter_data, columns=columns)
                        
                        log_dict = {
                            "generation": algorithm.n_gen,
                            "train/best_synflow": metrics['s_acc'],
                            "train/min_lipschitz": metrics['s_rob'],
                            "train/diversity": metrics['diversity'],
                            "train/stl_robustness": robustness,
                            "search/population_scatter": wandb.plot.scatter(table, "Lipschitz Bound", "SynFlow Score", title="Population Evolution"),
                            "search/population_table": table
                        }
                        self.gatekeeper.wandb_run.log(log_dict)
        
        res = minimize(problem,
                       algorithm,
                       ('n_gen', self.generations),
                       callback=STLCallback(self.gatekeeper, self.search_space),
                       verbose=True)
                       
        return res # Return full result object, not just opt

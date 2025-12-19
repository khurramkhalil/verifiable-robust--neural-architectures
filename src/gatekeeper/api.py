from src.primitives.lipschitz_layers import compute_architectural_lipschitz_bound
from src.gatekeeper.trajectory_monitor import STLTrajectoryMonitor

class FormalGatekeeper:
    def __init__(self, config, wandb_run=None):
        self.lipschitz_threshold = config.get('gatekeeper', {}).get('lipschitz_threshold')
        
        deg_tol = config.get('gatekeeper', {}).get('degradation_tolerance', 5.0)
        div_thresh = config.get('gatekeeper', {}).get('diversity_threshold', 0.1)
        
        self.monitor = STLTrajectoryMonitor(
            degradation_tol=deg_tol,
            diversity_thresh=div_thresh
        )
        self.history = []
        self.wandb_run = wandb_run

    def is_statically_feasible(self, model):
        """
        Check if the architecture satisfies the static Lipschitz bound.
        This is a 'Hard' constraint applied before training.
        """
        L_hat = compute_architectural_lipschitz_bound(model)
        
        # If threshold is not set (e.g. during Pilot), we accept all
        if self.lipschitz_threshold is None:
            return True
        
        # L_hat must be <= threshold
        return L_hat <= self.lipschitz_threshold

    def update_and_check_trajectory(self, generation_metrics):
        """
        Update history and check STL properties.
        generation_metrics: dict {'s_acc', 's_rob', 'diversity'}
        
        Note: s_rob here should be the Lipschitz value (lower is better), 
        as handled by trajectory_monitor to invert it for logic.
        """
        self.history.append(generation_metrics)
        return self.monitor.evaluate(self.history)

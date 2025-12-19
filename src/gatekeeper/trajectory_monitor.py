import rtamt

class STLTrajectoryMonitor:
    def __init__(self, degradation_tol=5.0, diversity_thresh=0.2):
        self.degradation_tol = degradation_tol
        self.diversity_thresh = diversity_thresh
        
        # Use Offline specification
        self.spec = rtamt.StlDiscreteTimeOfflineSpecification()
        
        # Declare variables
        self.spec.declare_var('acc_delta', 'float')
        self.spec.declare_var('rob_delta', 'float')
        self.spec.declare_var('diversity', 'float')
        self.spec.declare_var('diversity_thresh', 'float')
        
        self.spec.spec = 'always ((diversity >= diversity_thresh) and ((acc_delta > 0) -> (rob_delta >= -{})))'.format(self.degradation_tol)
        
        try:
            self.spec.parse()
        except Exception as e:
            print(f"STL Parse Error: {e}")

    def evaluate(self, history):
        """
        Evaluate the trajectory using offline monitoring.
        history: List of dicts {'s_acc', 's_rob', 'diversity'}
        """
        if len(history) < 2:
            return 1.0 # Trivially satisfied

        # Prepare trace for rtamt offline (Standard Format with 'time' key often safer/saner for some versions)
        # Attempting standard Dict format: {'time': [...], 'var': [...]}
        
        times = []
        acc_deltas = []
        rob_deltas = []
        diversities = []
        div_threshs = []
        
        for t in range(len(history)):
            curr = history[t]
            times.append(float(t)) # rtamt typically likes float time
            
            if t == 0:
                acc_delta = 0.0
                rob_delta = 0.0
            else:
                prev = history[t-1]
                acc_delta = curr['s_acc'] - prev['s_acc']
                rob_delta = -(curr['s_rob'] - prev['s_rob'])
            
            acc_deltas.append(acc_delta)
            rob_deltas.append(rob_delta)
            diversities.append(curr['diversity'])
            div_threshs.append(self.diversity_thresh)
            
        trace = {
            'time': times,
            'acc_delta': acc_deltas,
            'rob_delta': rob_deltas,
            'diversity': diversities,
            'diversity_thresh': div_threshs
        }
        
        try:
            # evaluate expects dataset. Dict of lists is valid in many versions.
            res = self.spec.evaluate(trace)
            # Result is list of [time, value].
            if len(res) > 0:
                # Return robustess at t=0
                return res[0][1]
            return 0.0
        except Exception as e:
            print(f"STL Eval Error: {e}")
            return -1.0

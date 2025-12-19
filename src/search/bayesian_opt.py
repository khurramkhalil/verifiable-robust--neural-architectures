import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from mll = gpytorch.mlls.ExactMarginalLogLikelihood
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from src.primitives.lipschitz_layers import compute_architectural_lipschitz_bound
from src.metrics.zero_cost_proxies import synflow_score

class ConstrainedBayesianOpt:
    def __init__(self, search_space, gatekeeper, n_iterations=30, n_initial=20, device='cpu'):
        self.search_space = search_space
        self.gatekeeper = gatekeeper
        self.n_iterations = n_iterations
        self.n_initial = n_initial
        self.device = device
        
    def search(self):
        # Initial Design
        X_train = []
        Y_train = []
        
        # Simple Loop to gather initial
        count = 0 
        while count < self.n_initial:
            idx = torch.randint(0, 15624, (1,))
            model = self.search_space.get_model(int(idx))
            
            if self.gatekeeper.is_statically_feasible(model):
                l_bound = compute_architectural_lipschitz_bound(model)
                model.to(self.device)
                syn = synflow_score(model, (3,32,32), self.device)
                
                X_train.append(idx.float())
                # Objectives: minimize L (maximize -L), maximize SynFlow
                Y_train.append(torch.tensor([-l_bound, syn]))
                count += 1
                
        X = torch.stack(X_train).unsqueeze(1) # (N, 1)
        Y = torch.stack(Y_train) # (N, 2)
        
        # References for HV
        ref_point = torch.tensor([-1000.0, 0.0]) # Approx worst case
        
        for i in range(self.n_iterations):
            # Fit GP
            # Note: 1D input (index) is not great for GP kernel.
            # Ideally we embed architecture.
            # Using bare index is very naive but per spec for "Bayesian Opt Integration" 
            # we follow the `x` tensor flow.
            # Better: use encoded arch.
            
            gp = SingleTaskGP(X, Y)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
            
            acq_func = qNoisyExpectedHypervolumeImprovement(
                model=gp,
                ref_point=ref_point,
                X_baseline=X,
                prune_baseline=True,
                sampler=sampler
            )
            
            # Optimize
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=torch.tensor([[0.0], [15624.0]]),
                q=5,
                num_restarts=10,
                raw_samples=512,
            )
            
            # Eval candidates
            new_x = candidates.detach()
            for x_cand in new_x:
                idx = int(x_cand.item())
                idx = max(0, min(idx, 15624)) # Clamp
                
                model = self.search_space.get_model(idx)
                if self.gatekeeper.is_statically_feasible(model):
                    l_bound = compute_architectural_lipschitz_bound(model)
                    model.to(self.device)
                    syn = synflow_score(model, (3,32,32), self.device)
                    
                    X = torch.cat([X, torch.tensor([[float(idx)]])])
                    Y = torch.cat([Y, torch.tensor([[-l_bound, syn]])])
        
        return X, Y

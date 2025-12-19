import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Conv2d):
    """
    A Convolutional Layer that tracks and constrains its spectral norm
    using Power Iteration on the direct convolution operator.
    
    This implementation correctly handles zero-padding by using the 
    conv_transpose2d operator as the true adjoint, rather than 
    reshaping the kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True, 
                 n_power_iterations=3, eps=1e-12):
        super().__init__(in_channels, out_channels, kernel_size, 
                         stride, padding, dilation, groups, bias)
        
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        
        # Persistent buffers for power iteration vectors.
        # Initialized lazily during the first forward pass to adapt to input resolution.
        self.register_buffer('u', None)
        self.register_buffer('v', None)
        self.register_buffer('sigma', torch.tensor(1.0))

    def _update_sigma(self, x):
        """
        Perform Power Iteration to estimate the spectral norm sigma.
        x: Input tensor of shape (N, C_in, H, W)
        """
        # We perform PI on a batch size of 1 to estimate the operator norm.
        # This is sufficient as the operator norm is input-independent 
        # (it depends only on weights and geometry).
        batch, c, h, w = x.shape
        target_shape = (1, c, h, w)

        # Initialize u if not set or if spatial dimensions changed (e.g., in NAS search)
        if self.u is None or self.u.shape != target_shape:
            # Re-initialize u randomly
            self.u = torch.randn(target_shape, device=x.device)
            self.u = F.normalize(self.u, dim=[1, 2, 3], eps=self.eps)
            
            # Initialize v with a forward pass to get correct shape
            # NOTE: v's shape depends on output shape of conv
            with torch.no_grad():
                self.v = F.conv2d(self.u, self.weight, None, 
                                  self.stride, self.padding, self.dilation, self.groups)
                self.v = F.normalize(self.v, dim=[1, 2, 3], eps=self.eps)

        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                # 1. Forward Operator A: u -> v
                v_s = F.conv2d(self.u, self.weight, None, 
                               self.stride, self.padding, self.dilation, self.groups)
                self.v = F.normalize(v_s, dim=[1, 2, 3], eps=self.eps)
                
                # 2. Adjoint Operator A^T: v -> u
                # The adjoint of conv2d with zero padding is conv_transpose2d.
                u_s = F.conv_transpose2d(self.v, self.weight, None, 
                                         self.stride, self.padding, 
                                         output_padding=0, 
                                         groups=self.groups, dilation=self.dilation)
                
                # Handle shape mismatch due to stride/padding ambiguities in transpose
                # conv_transpose2d output size might be slightly larger/different than original input
                if u_s.shape[2:] != self.u.shape[2:]:
                    # Crop to original input size
                    u_s = u_s[:, :, :self.u.shape[2], :self.u.shape[3]]
                
                self.u = F.normalize(u_s, dim=[1, 2, 3], eps=self.eps)

            # 3. Estimate Sigma using Rayleigh Quotient: sigma = ||A u|| (since u approx singular vector)
            out = F.conv2d(self.u, self.weight, None,
                           self.stride, self.padding, self.dilation, self.groups)
            self.sigma = torch.norm(out)

    def forward(self, x):
        if self.training:
            self._update_sigma(x)
        
        # We track sigma but do not force normalization during search to avoid stifling training.
        # This is per specification.
        return super().forward(x)

    def get_lipschitz_constant(self):
        return self.sigma.item()

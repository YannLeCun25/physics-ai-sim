import torch
import numpy as np

def generate_boundary_conditions(n_samples: int = 1000):
    """Generates synthetic boundary data for PINN training."""
    x = torch.linspace(0, 1, n_samples).view(-1, 1)
    t = torch.zeros_like(x)
    u = torch.sin(np.pi * x) # Initial condition u(x,0) = sin(pi*x)
    return x, t, u

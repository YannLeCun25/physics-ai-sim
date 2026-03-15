import torch

def laplacian(u, x, y):
    """Calculates the Laplacian of u with respect to x and y."""
    u_xx = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_yy = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    return u_xx + u_yy

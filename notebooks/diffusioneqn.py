import numpy as np
import torch

# @torch.compile()
def diffusioneqn_batch(k: torch.Tensor, F: float, source: torch.Tensor, rightbc: float):
    """
    Solve 1-D diffusion equation with given diffusivity field k
    and left-hand flux F.
    
    ARGUMENTS: 
        xgrid = vector with equidistant grid points
            F = flux at left-hand boundary, k*du/dx = -F 
       source = source term, either a vector of values at points in xgrid
                or a constant
      rightbc = Dirichlet BC on right-hand boundary
    Domain is given by xgrid (should be [0,1])
    """
    device = k.device
    N, M = k.shape
    h = 1 / (N - 1)
    
    # Prepare source term
    if isinstance(source, (int, float)):
        f = -source * torch.ones((N-1, M), device=device)
    else:
        f = -source[:N-1, :]

    # Build matrix A in batched form
    A = -2 * k[:-1, :] - k[1:, :] - torch.cat((k[:1, :], k[:-2, :]), dim=0)
    A_diag = A / (2 * h * h)

    A_offdiag = (k[:-2, :] + k[1:-1, :]) / (2 * h * h)
    A_matrix = torch.zeros((N-1, N-1, M), device=device)
    
    A_matrix[range(N-1), range(N-1), :] = A_diag
    A_matrix[range(N-2), range(1, N-1), :] = A_offdiag
    A_matrix[range(1, N-1), range(N-2), :] = A_offdiag

    # Adjustments for boundary conditions
    A_matrix[0, 1, :] += k[0, :] / h**2
    f[0, :] -= 2 * F / h
    f[-1, :] -= rightbc * (k[-1, :] + k[-2, :]) / (2 * h**2)

    # Solve for u_internal
    u_internal = torch.linalg.solve(A_matrix.permute(2, 0, 1), f.T).T

    # Concatenate solutions with the boundary condition
    u_solution = torch.cat((u_internal, rightbc * torch.ones((1, M), device=device)), dim=0)

    return u_solution
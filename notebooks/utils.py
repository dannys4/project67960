import torch
import torch.nn as nn
import numpy as np

def clenshaw_curtis(n: int):
    """
    CLENSHAW_CURTIS computes a Clenshaw Curtis quadrature rule.
    
    Parameters:
        Input, integer N, the order.
        Output, real X(N), the abscissas.
        Output, real W(N), the weights.
    
    Author:
        John Burkardt
    
    Licensing:
        This code is distributed under the GNU LGPL license.
    """
    if (n == 1):
        x = np.zeros(n)
        w = np.zeros(n)
        w[0] = 2.0
    else:
        theta = np.zeros(n)
        for i in range(0, n):
            theta[i] = float(n - 1 - i) * np.pi / float(n - 1)

        x = np.cos(theta)
        w = np.zeros(n)

        for i in range(0, n):
            w[i] = 1.0
            jhi = ((n - 1) // 2)
            for j in range(0, jhi):
                if (2 * (j + 1) == (n - 1)):
                    b = 1.0
                else:
                    b = 2.0
                w[i] = w[i] - b * np.cos(2.0 * float(j + 1) * theta[i]) \
                    / float(4 * j * (j + 2) + 3)

        w[0] = w[0] / float(n - 1)
        for i in range(1, n - 1):
            w[i] = 2.0 * w[i] / float(n - 1)
        w[n-1] = w[n-1] / float(n - 1)

    return x, w

def clenshaw_curtis_ab(n: int, a: float = 0, b: float = 1):
    """
    clenshaw_curtis_ab(n, a=0, b=1)
    
    Returns Clenshaw Curtis points shifted and scaled for interval [a,b]
    """
    pts, wts = clenshaw_curtis(n)
    pts = (pts + 1)*(b-a)/2 + a
    wts = wts*(b-a)/2
    return pts, wts


def torch_find_bracket_inplace(z: torch.Tensor, fcn, y_lo: torch.Tensor, y_hi: torch.Tensor, maxiter: int = 50):
  """
  torch_find_bracket_inplace(z, fcn, y_lo, y_hi, maxiter=50)
  
  For fcn(y) = z, store lower and upper possible bounds for y in y_lo and y_hi given batch z
  """
  for _ in range(maxiter):
    z_lo = fcn(y_lo)
    which_hi = z_lo > z
    if not which_hi.any():
      break
    # print(f"\n{y_hi.shape}, {y_lo.shape}, {which_hi.shape}")
    y_hi[which_hi] =   y_lo[which_hi]
    y_lo[which_hi] = 2*y_lo[which_hi] # Starts negative

  for _ in range(maxiter):
    z_hi = fcn(y_hi)
    which_lo = z_hi < z
    if not which_lo.any():
      break
    y_lo[which_lo] =   y_hi[which_lo]
    y_hi[which_lo] = 2*y_hi[which_lo] # Starts positive unless negative from previous loop

def torch_bisect(z: torch.Tensor, fcn, y_lo: torch.Tensor, y_hi: torch.Tensor, maxiter: int = 50, atol: float = 1e-6):
  """
  torch_bisect(z, fcn, y_lo, y_hi, maxiter=50, atol=1e-6)

  Perform bisection search to invert fcn for output batch z using preallocated y_lo, y_hi. Stop iterating when maximum iterations maxiter or absolute tolerance atol is reached on |fcn(y)-z|.
  """
  y_mid = (y_lo + y_hi)/2
  converged = False
  for iter in range(maxiter):
    z_mid = fcn(y_mid)
    z_diff = z_mid - z
    # print((z_diff.abs() < atol).sum())
    if torch.all(z_diff.abs() < atol):
      # print(f"converged in {iter} iterations")
      converged = True
      break
    which_lo = z_diff < 0
    which_hi = torch.logical_not(which_lo)
    y_hi[which_hi] = y_mid[which_hi]
    y_lo[which_lo] = y_mid[which_lo]
    y_mid = (y_lo + y_hi)/2
  return y_mid, converged

def create_hidden_sequence(input_size: int, num_hidden: int, hidden_layers: int, output_size: int, nonlinearity, normalization_layer, device: torch.device):
  """
  create_hidden_sequence(input_size, num_hidden, hidden_layers, output_size, nonlinearity, normalization_layer, device)

  Create an MLP with specific count of hidden layers, each with identical size, and a particular output size of network. Uses a nonlinearity and allows for normalization layer.
  """
  in_layer = (nn.Linear(input_size, num_hidden, device=device),)
  hidden_layers = sum([(nonlinearity(), nn.Linear(num_hidden, num_hidden, device=device)) for _ in range(hidden_layers - 2)], ())
  if normalization_layer is not None:
    hidden_layers = hidden_layers + (normalization_layer(num_hidden),)
  return in_layer + hidden_layers + (nonlinearity(), nn.Linear(num_hidden, output_size, device=device))

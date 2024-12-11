import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
from utils import *

class TriangularComponent(nn.Module):
  def __init__(self, input_size, quad_pts, quad_wts,
               hidden_layers = 3, num_hidden = 0,
               rectifier = nn.Softplus, int_nonlinearity = nn.GELU,
               sep_nonlinearity = nn.ReLU, normalization_layer = None,
               device=None):
    super().__init__()
    self.input_size = input_size
    self.device = device
    if num_hidden <= 0:
      num_hidden = min(4*input_size, 128)

    self.quad_pts = torch.tensor(quad_pts, dtype=torch.float, device=device).reshape((1,-1))
    self.quad_wts = torch.tensor(quad_wts, dtype=torch.float, device=device)

    int_all_layers = create_hidden_sequence(input_size, num_hidden, hidden_layers, 1, int_nonlinearity, normalization_layer, device) + (rectifier(),)
    self.int_network = nn.Sequential(*int_all_layers)
    if input_size > 1:
      sep_all_layers = create_hidden_sequence(input_size-1, num_hidden, hidden_layers, 1, sep_nonlinearity, normalization_layer, device)
      self.sep_network = nn.Sequential(*sep_all_layers)

  def forward(self, x: torch.Tensor):
    B, Q = x.shape[0], len(self.quad_wts)
    x = x[:,:self.input_size]

    # For Q quad points, need Q evaluations for each element of batch
    x_quad = x.unsqueeze(1).repeat(1,Q,1) # x_quad shape = (B x Q x D_in)
      
    # Each quadrature point is in [0,1], which we take as "proportions" of x[-1] for each input x
    x_quad[:,:,-1] *= self.quad_pts

    # Evaluate the integral network, which has output size 1
    x_eval_int = self.int_network(x_quad).reshape((B,Q)) # B x Q x 1 -> B x Q

    # y = int_0^x[-1] r(f(x[:-1], v)) dv = x[-1] * int_0^1 r(f(x[:-1],t*x[-1])) dt
    comp_eval = (x_eval_int @ self.quad_wts) * x[:,-1] # B
    comp_eval = comp_eval.reshape((-1,1)) # B x 1

    if self.input_size > 1:
      prefix_out = self.sep_network(x[:,:-1]) # B x 1
      comp_eval += prefix_out

    return comp_eval

  def inverse(self, x: torch.Tensor, z: torch.Tensor):
    B = x.shape[0]
    
    x = x[:,:self.input_size-1]
    z = z[:,:1]
    if self.input_size > 1:
      z -= self.sep_network(x)
    
    z_prime = z.flatten()
    y_lo = -torch.ones(B, device=self.device)
    y_hi =  torch.ones(B, device=self.device)
    xy_tmp = torch.hstack((x, torch.empty((B,1), device=self.device)))

    def forward_eval(y):
      xy_tmp[:,-1] = y
      return self(xy_tmp).flatten()

    torch_find_bracket_inplace(z_prime, forward_eval, y_lo, y_hi)
    y, bisect_converged = torch_bisect(z_prime, forward_eval, y_lo, y_hi)
    if not bisect_converged:
      print("WARNING: Bisect inversion did not converge")

    return y

  def logdet(self, x):
    return self.int_network(x[:,:self.input_size]).log()

class TriangularMap(nn.Module):
    def __init__(self, input_size: int, output_size: int, components: list):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.components = nn.ModuleList(components)
    
    def forward(self, x: torch.Tensor):
        return torch.cat([comp(x) for comp in self.components], dim=-1)

    def logdet(self, x: torch.Tensor):
        return torch.cat([comp.logdet(x) for comp in self.components], dim=-1).sum(axis=-1)

    def inverse(self, x: torch.Tensor, z: torch.Tensor):
        if z.shape[1] < self.output_size:
            raise ValueError(f'Expected reference to have {self.output_size} columns, got {z.shape[1]}')
        x_dim_exp = (z.shape[0], self.input_size - self.output_size)
        if not (x.shape == x_dim_exp):
            raise ValueError(f'Expected x to have shape {x_dim_exp}, got {x.shape}')
        # First prefix is x, then concat it with the outputs of inverse for each comp.
        x_output = x
        for (j, comp) in enumerate(self.components):
            z_slice = z[:,j:j+1]
            x_j = comp.inverse(x_output, z_slice)
            x_output = torch.cat((x_output, x_j.reshape((-1,1))), dim=-1)
        return x_output[:,-self.output_size:]

def TriangularMapFactory(input_size: int, output_size: int, *args, **kwargs):
    # S(y,x) requires y in dimension >= 0
    prefix_size = input_size - output_size
    if prefix_size < 0:
        raise ValueError(f"Expected input >= output sizes, got {input_size}, {output_size}")

    components = [TriangularComponent(prefix_size + j + 1, *args, **kwargs) for j in range(output_size)]
    return TriangularMap(input_size, output_size, components)

@torch.compile
def TriangularMapNormalCrossEntropy(trimap: TriangularMap, X: torch.Tensor):
    comps = trimap.components
    loss = torch.zeros((), device=X.device)
    for comp in comps:
        Z_j_norm2 = comp(X).square().sum()
        nabla_Z_j = comp.logdet(X).sum()
        loss_j = 0.5*Z_j_norm2 - nabla_Z_j
        loss += loss_j
    return loss

@torch.no_grad()
def sample_normal_ref(S_map: TriangularMap, N: int, x: torch.Tensor = None, device: torch.DeviceObjType = None):
  if x is None:
    x = torch.empty((N,0), device=device)
  elif x.ndim == 1:
    x = x.reshape((1,-1)).repeat((N,1))
  Z_cols = S_map.output_size if type(S_map) == TriangularMap else 1
  Z = torch.randn(N, Z_cols, device=device)
  return S_map.inverse(x, Z)
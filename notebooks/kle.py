# Copyright 2023 Daniel Sharp
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the “Software”), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import torch

@torch.compile()
def unit_exp(x1, x2, L=0.3, var_y=0.3):
    return var_y*torch.exp(-torch.abs(x1-x2)/L)

def form_KL_uniform(covfun, lo, hi, num_quad, device):
    pts, wts = np.polynomial.legendre.leggauss(num_quad)
    pts = (pts * (hi-lo)/2) + (hi + lo)/2
    wts *= (hi-lo)/2
    pts = torch.tensor(pts, device=device)
    wts = torch.tensor(wts, device=device)
    mesh1, mesh2 = torch.meshgrid(pts, pts)
    C_mat = covfun(mesh1, mesh2)
    W_half = torch.diag(torch.sqrt(wts))
    A = W_half @ C_mat @ W_half
    lam, phi = torch.linalg.eigh(A)
    psi = torch.diag(1 / torch.sqrt(wts)) @ phi
    lam = torch.flip(lam, (0,))
    psi = torch.flip(psi, (1,))
    psi = psi * (2*(psi[0,:] > 0) - 1)
    return lam, psi, pts, wts

class KLE:
    def __init__(self, lam, psi, pts, wts, covfun, device):
        self.lam = lam
        self.psi = psi
        self.pts = pts
        self.wts = wts
        self.covfun = covfun
        self.device = device
    
    def project_fcn(self, fcn):
        fcn_eval = fcn(self.pts)
        proj_coeffs = torch.einsum('qn,q,q->n', self.psi, self.wts, fcn_eval)
        return proj_coeffs / torch.sqrt(self.lam)

    def __call__(self, xgrid, z = None, M: int = None):
        generate_z = z is None
        if generate_z:
            z_shape = (len(self.lam),) if M is None else (len(self.lam), M)
            z = torch.randn(z_shape, device=self.device, dtype=torch.float64)
        X,Y = torch.meshgrid(xgrid, self.pts)
        covmat = self.covfun(X,Y) * self.wts
        psi_evals = covmat @ self.psi @ torch.diag(1/torch.sqrt(self.lam))
        ret = psi_evals @ z
        if generate_z:
            ret = (ret, z)
        return ret

def default_KLE(N_trunc, num_quad = 100, device=None):
    lam, psi, pts, wts = form_KL_uniform(unit_exp, 0., 1., num_quad, device)
    return KLE(lam[:N_trunc], psi[:,:N_trunc], pts, wts, unit_exp, device)

def ExampleKLE():
    xgrid = np.linspace(0,1,101)
    n_trunc = 100
    kl = default_KLE(n_trunc, max(100, round(1.5*n_trunc)))
    rng = np.random.default_rng()
    z = rng.standard_normal(n_trunc)
    kl_eval = eval_KLE(kl, xgrid, z)
    return xgrid, kl_eval, z

if __name__ == '__main__':
    xgrid, kl_eval, z = ExampleKLE()
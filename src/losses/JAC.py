from __future__ import division
import sys
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
from typing import Any, Callable, Optional, Tuple, List


class JacobianReg(nn.Module):

    def __init__(self, 
        latent_dim, 
        num_proj=1, 
        device='cpu'):
        super(JacobianReg, self).__init__()
        self.latent_dim = latent_dim
        self.num_proj = num_proj
        self.device = device

    def projection_matrix(self, batch_size):
        # Projection matrix
        proj = torch.rand(size=(self.num_proj, batch_size, self.latent_dim)) * 2.0 - 1.0
        l_norm = proj.pow(2).sum(-1, keepdim=True).pow(0.5)
        proj = proj.div(l_norm).to(self.device)
        return proj

    def _jacobian_vector_product( self, 
                                x: Tensor, 
                                y: Tensor, 
                                v: Tensor, 
                                create_graph: bool = False) -> Tensor: 
        """
        Jacobian-vector product dy/dx dot v.
        If you want to differentiate it, create_graph=True
        """
        flat_y = y.reshape(-1)
        flat_v = v.reshape(-1)

        grad_x,  = torch.autograd.grad(flat_y, x, flat_v, retain_graph=True, create_graph=create_graph)
        return grad_x

    def forward(self, input: Tensor, target: Tensor, projs: Tensor=None) -> Tensor:
        batch_size, emb_dim = target.size()
        if projs is None: 
            projs = self.projection_matrix(batch_size)
            num_proj = self.num_proj
        else:
            assert projs[0].size() == target.size(), f"Projection matrix size is has size of {projs.size()}"
            num_proj = len(projs)
        J2 = 0.0
        for v in projs:
            Jv = self._jacobian_vector_product(input, target, v, create_graph=True)
            J2 += emb_dim*torch.norm(Jv)**2 / (num_proj*batch_size)
        return J2


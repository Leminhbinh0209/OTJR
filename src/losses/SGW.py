import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class SWDLoss(nn.Module):
    """
    Sliced Wassersten distance
    """
    def __init__(self, 
                latent_dim: int,
                num_proj: int=32,
                is_weighted: bool=False,
                device: str='cpu') -> None:
        super(SWDLoss, self).__init__()
        self.latent_dim = latent_dim
        self.num_proj = num_proj
        self.is_weighted = is_weighted
        self.device = device
        self.weighted_proj = self._weights().to(self.device) if self.is_weighted else None
        print("SWD infor:\nNum projection: {}\nLatent dim: {}\nIs weighted: {}".format(self.num_proj, self.latent_dim, self.is_weighted))

    def _weights(self) -> Tensor:
        """ 
        Create weighted for projections
        """
        k =  np.arange(self.num_proj) + 1
        tau =  (np.cos(1.0*np.pi * k / (self.num_proj))+ 1) / 2.0
        return torch.unsqueeze(torch.from_numpy(tau), dim=1)

    def projection_matrix(self)-> Tensor:
        # Projection matrix
        proj = torch.rand(size=(self.latent_dim, self.num_proj)) * 2.0 - 1.0
        l_norm = proj.pow(2).sum(0, keepdim=True).pow(0.5)
        proj = proj.div(l_norm).to(self.device)
        return proj

    def _sliced_wasserstein_distance(self, source_dist, target_dist, projections=None):
        """
        source_dist: B x F, 
        target_dist: B x F, 
        projections: F x num_proj
        
        """
        if projections is None:
            projections = self.projection_matrix()
        else:
            l_norm = projections.pow(2).sum(0, keepdim=True).pow(0.5)
            projections = projections.div(l_norm).to(self.device)

        source_dist_projections = source_dist.matmul(projections).transpose(0, 1) # num_proj x B
        target_dist_projections = target_dist.matmul(projections).transpose(0, 1) # num_proj x B

        target_dist_proj_sort, _ = torch.sort(target_dist_projections, dim=1)
        rank_source_dist_proj = torch.argsort(torch.argsort(source_dist_projections, dim=1), dim=1)
        source_dist_proj_new = torch.gather(input=target_dist_proj_sort, dim=1, index=rank_source_dist_proj)
        delta = source_dist_proj_new - source_dist_projections # num_proj x B

        wasserstein_distance = torch.pow(delta, 2)

        movement = torch.zeros_like(source_dist) # B x F
        for i in range(len(delta)):
            movement += torch.outer(torch.squeeze(delta[i]), projections[:,i])
        norm_movement = movement.pow(2).sum(1, keepdim=True).pow(0.5)
        movement_norm = movement.div(norm_movement) 

        ### Include weighted for informativce projection!
        if self.is_weighted:
            batch_size = source_dist.size(0)
            wasserstein_distance = wasserstein_distance * self.weighted_proj
            return torch.sum(wasserstein_distance) / torch.sum(self.weighted_proj * batch_size), movement_norm
        else:
            return wasserstein_distance.mean(), movement_norm.clone().detach()

    def forward(self, source_dist: Tensor, 
                    target_dist: Tensor, 
                    projs: Tensor=None, 
                    inc_move: bool=False):

        dist, move =  self._sliced_wasserstein_distance(source_dist, target_dist, projs)
        return dist if not inc_move else (dist, move)

from __future__ import print_function
import sys
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SinkhornDiv(nn.Module):
    """
    Sinkhorn divergence Loss 
    reference: https://github.com/CEA-LIST/adv-sat/blob/238a294896ffd5361fbce0742d9a60c2b653dfb1/utils/losses.py#L47
        params:
            optimizer: optimizer to train model
            step_size: manitude of gradient adding to natural image in each backpropagating
            epsilon: radius of norm ball
            perturb_steps: number of step to create adversarial sample
            beta: hyper parameter in loss
            norm: L norm ball
            loss: type of loss

    """
    def __init__(self, 
                optimizer, 
                config, 
                device='cuda') -> None:
        super(SinkhornDiv, self).__init__()

        self.num_class  = config.MODEL.num_class

        self.optimizer = optimizer 
        self.step_size = config.ADVER.step_size 
        self.epsilon = config.ADVER.epsilon 
        self.perturb_steps = config.ADVER.num_steps 


        self.norm = config.ADVER.norm
        self.eps = config.LOSS.Sink.eps
        self.max_iter = config.LOSS.Sink.max_iter

        self.loss = config.LOSS.loss
        self.device = device

        if self.loss in ['sink']:
            print("Sinkhorn Divergence w/ PGD adversarial sample")
            self.adv_loss = nn.CrossEntropyLoss()
            self.criterion_loss = SinkhornDistance(eps=self.eps, max_iter=self.max_iter, device=self.device)
        else:
            raise RuntimeError(f"Undefined loss function: {self.loss}")

    def forward(self, model, x_natural: Tensor, y: Tensor, callback: dict) -> Tensor:
        model.eval()
        # generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(x_natural.device).detach()
        if self.norm == 'l_inf':
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    if self.loss in ['sink']:
                        loss_c = self.adv_loss(model(x_adv), y)

                grad = torch.autograd.grad(loss_c, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        model.train()
        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        # zero gradient
        self.optimizer.zero_grad()
        
        penultimate_ft,  logits = model(x_natural, return_feature=True)
        penultimate_ft_adv, logits_adv = model(x_adv, return_feature=True)

        if self.loss in ['sink']:

            loss_adv = self.adv_loss(logits_adv, y)
            # Compute the divergence
            dist, _, _ = self.criterion_loss(logits, logits_adv)
            dist_adv, _, _ = self.criterion_loss(logits_adv, logits_adv)
            dist_clean, _, _ = self.criterion_loss(logits, logits)
            dist = dist - 0.5*(dist_adv + dist_clean)
            loss_sink = dist

            Loss = loss_adv + loss_sink
        callback['adv'] = loss_adv.item()
        callback['sink'] = loss_sink.item()

        return Loss

class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    ref: https://github.com/dfdazac/wassdistance
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, device, p=2, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.device = device
        self.p = p

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y) # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        if self.device is not None:
            C = C.cuda(self.device, non_blocking=True)
            mu = mu.cuda(self.device, non_blocking=True)
            nu = nu.cuda(self.device, non_blocking=True)
            u = u.cuda(self.device, non_blocking=True)
            v = v.cuda(self.device, non_blocking=True)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1
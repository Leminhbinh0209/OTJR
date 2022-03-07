from __future__ import print_function

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class ALPLoss(nn.Module):
    """ALP Loss
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
                device='cuda'):
        super(ALPLoss, self).__init__()
        
        self.optimizer = optimizer 
        self.step_size = config.ADVER.step_size 
        self.epsilon = config.ADVER.epsilon 
        self.perturb_steps = config.ADVER.num_steps 

        self.beta = config.LOSS.ALP.beta
        self.norm = config.ADVER.norm

        self.loss = config.LOSS.loss        
        self.device = device

        if self.loss in ['alp']:
            print("ALP training")
            self.criterion_loss = nn.CrossEntropyLoss()
        else:
            raise RuntimeError(f"Undefined loss function: {self.loss}")

    def forward(self, model, x_natural: Tensor, y: Tensor, callback: dict) -> Tensor:
        model.eval()
        # generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        if self.norm == 'l_inf':
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    if self.loss in ['alp']:
                        loss_c = self.criterion_loss(model(x_adv), y)


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
        # calculate robust loss
        penultimate_ft, logits = model(x_natural, True)
        penultimate_ft_adv, logits_adv = model(x_adv, True)

        if self.loss in ['alp']:
            loss_robust = 0.5 * self.criterion_loss(logits, y) + 0.5 * self.criterion_loss(logits_adv, y)
            loss_alp = F.mse_loss(logits, logits_adv)
            Loss = loss_robust + self.beta * loss_alp
            callback['alp'] = loss_robust.item()
            callback['alp_mse'] = loss_alp.item()
        return Loss


from __future__ import print_function
import sys
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class TRADESLoss(nn.Module):
    """Trade Loss 
        params:
            optimizer: optimizer to train model
            step_size: manitude of gradient adding to natural image in each backpropagating
            epsilon: radius of norm ball
            perturb_steps: number of step to create adversarial sample
            beta: hyper parameter in loss
            norm: L norm ball
            loss: type of loss

    """
    def __init__(self, optimizer, config, 
                device='cuda'):
        super(TRADESLoss, self).__init__()
        
        self.optimizer = optimizer 
        self.step_size = config.ADVER.step_size 
        self.epsilon = config.ADVER.epsilon 
        self.perturb_steps = config.ADVER.num_steps 

        self.beta = config.LOSS.TRADES.beta
        self.norm = config.ADVER.norm
        self.loss = config.LOSS.loss

        self.device = device

        if self.loss in ['trades']:
            print("TRADES training")
            self.criterion_loss = nn.KLDivLoss(reduction='sum')
            self.natural_loss = nn.CrossEntropyLoss()
        else:
            raise RuntimeError(f"Undefined loss function: {self.loss}")

    def forward(self, model, x_natural: Tensor, y: Tensor, callback: dict) -> Tensor:
        model.eval()
        batch_size = len(x_natural)
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(x_natural.device).detach()
        if self.norm == 'l_inf':
            for perturb_iter in range(self.perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    if self.loss in ['trades']:
                        loss_c = self.criterion_loss(F.log_softmax(model(x_adv), dim=1),
                                            F.softmax(model(x_natural), dim=1))

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
        _,  logits = model(x_natural, return_feature=True)
        _, logits_adv = model(x_adv, return_feature=True)
        
        if self.loss in ['trades']:
            loss_natural = self.natural_loss(logits, y)
            loss_robust = (1.0 / batch_size) * self.criterion_loss(F.log_softmax(logits_adv, dim=1),
                                                        F.softmax(logits, dim=1))
            Loss = loss_natural + self.beta * loss_robust
            callback['nat'] = loss_natural.item()
            callback['trades'] = loss_robust.item()
        return Loss

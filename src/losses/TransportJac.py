from __future__ import division
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .SGW import SWDLoss
from .JAC import JacobianReg

class TransportJac(nn.Module):
    """
    Projectin pursuit for mont Map and utilize for Jacobian regularization
    """
    def __init__(self, 
                optimizer, 
                config, 
                device: str = 'cuda') -> None:
        super(TransportJac, self).__init__()

        self.num_class  = config.MODEL.num_class
        self.latent_dim  = config.MODEL.latent_dim
        self.optimizer = optimizer 
        self.step_size = config.ADVER.step_size 
        self.epsilon = config.ADVER.epsilon 
        self.perturb_steps = config.ADVER.num_steps 

        self.norm = config.ADVER.norm
        self.loss = config.LOSS.loss
        self.device = device

        self.trans_beta = config.LOSS.SWD.beta
        self.itr = config.LOSS.SWD.iter
        self._ot_logit = config.LOSS.SWD.use_logit
        self.one_side = config.LOSS.SWD.one_side
        self.is_weighted = config.LOSS.SWD.weighted

        self.num_jac_proj = config.LOSS.Jac.num_proj
        self._jac_logit = config.LOSS.Jac.use_logit
        self.jac_beta = config.LOSS.Jac.beta

        self.reg_dim = self.latent_dim if not self._ot_logit else self.num_class

        if self.one_side:
            print("One-side optimal transport")
        else:
            print("Two side optimal transport")
            
        self.adv_type = config.LOSS.SWD.adv_type
        
        print("Training with {} adversarial cross-entropy loss".format(self.adv_type))
        self.adv_loss = nn.CrossEntropyLoss()

        

        if self.loss == 'trans_swd':
            print(f"SWD with random projection  w/ {self.adv_type} adversarial sample")
            self.transport_criterion = SWDLoss(latent_dim=self.reg_dim, 
                                                num_proj=self.itr, 
                                                is_weighted=False,
                                                device=self.device)
        elif self.loss == 'trans_swdj':
            print(f"SWD with + Jacobian Regularization random projection w/ {self.adv_type} adversarial sample")
            self.transport_criterion = SWDLoss(latent_dim=self.reg_dim, 
                                                num_proj=self.itr,
                                                is_weighted=False, 
                                                device=self.device)

            self.jac_reg = JacobianReg(self.reg_dim, self.num_jac_proj, self.device)

        elif self.loss == 'sw_jac':
            print("Using only Jacobian Regularization with projection from SWD")
            self.transport_criterion = SWDLoss(latent_dim=self.reg_dim, 
                                                num_proj=self.itr,
                                                is_weighted=False, 
                                                device=self.device)
            self.jac_reg = JacobianReg(self.reg_dim, self.num_jac_proj, self.device)
        elif self.loss == 'rand_jac':
            print("Using only Jacobian Regularization with random projection")
            self.jac_reg = JacobianReg(self.reg_dim, self.num_jac_proj, self.device)
        else: 
            raise RuntimeError(f"Undefined loss function: {self.loss}")

    def ori_loss(self, 
                logits: Tensor, 
                logits_adv: Tensor, 
                y: Tensor,
                adv_type: str,
                callback: dict) -> Tensor:

        batch_size = len(logits)
        
        if adv_type == 'pgd':
            loss_ori = self.adv_loss(logits_adv, y)
            callback['pgd'] = loss_ori.item()
        else:
            raise RuntimeError(f"Undefined adversarial type function: {adv_type}")
        return loss_ori

    def __rand_jac(self, model, x_natural: Tensor, y: Tensor, callback: dict) -> Tensor:
        "Random Jacobian regularization"
        x_natural.requires_grad_()
        logit = model(x_natural)
        nat_loss = nn.CrossEntropyLoss()(logit, y)
        jac_regular = self.jac_reg(x_natural, logit, None)
        Loss = nat_loss +  self.jac_beta*jac_regular
        callback['nat'] = nat_loss.item()
        callback['jac'] = self.jac_beta*jac_regular.item()
        return Loss

    def __gen_adversarial_sample(self, model, x_natural: Tensor, y: Tensor) -> Tensor:
        """Generate adversarial sample for training"""
        model.eval()
        # generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(x_natural.device).detach()
        if (self.norm == 'l_inf'):
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    if self.adv_type == "trades": 
                        loss_c = self.adv_loss(F.log_softmax(model(x_adv), dim=1),
                                            F.softmax(model(x_natural), dim=1))
                    else:
                        loss_c = self.adv_loss(model(x_adv), y)

                grad = torch.autograd.grad(loss_c, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        return x_adv

    def forward(self, model, x_natural: Tensor, y: Tensor, callback: dict) -> Tensor:
        if self.loss == 'rand_jac':
            return self.__rand_jac(model, x_natural, y, callback)

        x_adv = self.__gen_adversarial_sample( model, x_natural, y)
        model.train()
        
        # zero gradient
        self.optimizer.zero_grad()
        if self.loss in ['trans_swdj', 'sw_jac', 'rand_jac']:
            x_natural.requires_grad_()

        penultimate_ft,  logits = model(x_natural, return_feature=True)
        penultimate_ft_adv, logits_adv = model(x_adv, return_feature=True)

        if  not self._ot_logit:
            data_src = penultimate_ft_adv
            data_des = penultimate_ft
        else:
            data_src = logits_adv
            data_des = logits

        if not self._jac_logit:
            der_target  = penultimate_ft
        else: 
            der_target = logits

        if self.loss == 'trans_swd':
            loss_ori = self.ori_loss(logits, logits_adv, y, self.adv_type, callback)
            transport_loss = self.transport_criterion(data_src, data_des.clone().detach()) if self.one_side else self.transport_criterion(data_src, data_des)
            Loss = loss_ori + self.trans_beta*transport_loss
            callback['trans'] = self.trans_beta * transport_loss.item()
         
        elif self.loss == 'trans_swdj':
            loss_ori = self.ori_loss(logits, logits_adv, y, self.adv_type, callback)
            transport_loss, jac_dir = self.transport_criterion(data_src, data_des.clone().detach(), None, True) if self.one_side else self.transport_criterion(data_src, data_des, None, True)
            jac_regular = self.jac_reg(x_natural, der_target, torch.unsqueeze(jac_dir, 0))
            
            Loss = loss_ori + self.trans_beta*transport_loss + self.jac_beta*jac_regular
            callback['trans'] = self.trans_beta*transport_loss.item()
            callback['jac'] = self.jac_beta*jac_regular.item()

        elif self.loss == 'sw_jac':
            nat_loss = nn.CrossEntropyLoss()(logits, y)
            _, jac_dir = self.transport_criterion(data_src, data_des, None, True)
            jac_regular = self.jac_reg(x_natural, der_target, torch.unsqueeze(jac_dir, 0))
            Loss = nat_loss + self.jac_beta*jac_regular 
            callback['nat'] = nat_loss.item()
            callback['jac'] = self.jac_beta*jac_regular.item()
            
        return Loss


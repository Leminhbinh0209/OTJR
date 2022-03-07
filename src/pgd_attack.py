from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from models.wideresnet import *
from helper.utils import  setup_device
from datasets import get_cifar10_dataloaders, get_cifar100_dataloaders
import easydict
import yaml

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 

parser = argparse.ArgumentParser(description='PGD Attack')
parser.add_argument('--config', '--cf', type=str,  
                    default="../configs/TEST_CIFAR10.yaml", 
                    help='config file directory')

args = parser.parse_args()
def _pgd_whitebox_multistep(model, X, y, epsilon, num_steps, step_size):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if config.TEST.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(y.device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    robust_err = dict()
    # Inintialize the dictionary 
    for st in num_steps: 
        robust_err[st] = 0.0

    for st in range(np.max(num_steps)):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
        if (st+1) in num_steps: 
            err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
            robust_err[st+1] = err_pgd.item()

    return err.item(), robust_err

def eval_adv_test_whitebox_multistep(model, device, test_loader, epsilon, step_size, num_steps: list):

    """
    evaluate model by white-box attack with mutiple steps
    params:
        epsilon: 
        step_size : 
        num_steps: list of steps
    """
    model.eval()
    robust_err_total = dict()
    # Inintialize the dictionary 
    for st in num_steps: 
        robust_err_total[st] = 0.0
    natural_err_total = 0
    
    for idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust =  _pgd_whitebox_multistep(model, X,  y, 
                                                epsilon=epsilon,  
                                                num_steps=num_steps,
                                                step_size=step_size)
        natural_err_total += err_natural
        for st in num_steps: 
            robust_err_total[st] += err_robust[st]
        sys.stdout.write("\r")
        sys.stdout.flush()
        sys.stdout.write('[{:03d}/{:03d}]'.format(idx+1, len(test_loader)))

    print('\nnatural acc: {:.2f}%'.format(100.0 - 100.0*natural_err_total/len(test_loader.dataset)))

    for st in num_steps: 
        print('robust acc {}: {:.2f}%'.format(st, 100.0 - 100.0*robust_err_total[st]/len(test_loader.dataset)))

def main(config):
    # white-box attack
    # general attack
    #######  Set up GPUs and benchmark model #######
    print("GPU:", config.SYS.GPUs)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.SYS.GPUs
    device, device_ids = setup_device(config.SYS.num_gpus)

    use_cuda = config.SYS.use_cuda and torch.cuda.is_available()
    np.random.seed(config.SYS.random_seed)
    torch.manual_seed(config.SYS.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    ####### Data loader #######
    if config.dataset == 'cifar10':
        config.MODEL.num_class = 10
        train_loader, test_loader  = get_cifar10_dataloaders(config)
    elif config.dataset == 'cifar100':
        train_loader, test_loader  = get_cifar100_dataloaders(config)
        config.MODEL.num_class = 100
    else: 
        raise RuntimeError(f"Undefined dataset: {config.dataset}")
    #######  Define training style function #######
    if config.model_name in ['wideresnet_34', 'wideresnet_28']:
        model = WideResNet(34 if config.model_name=='wideresnet_34'else 28, 
            num_classes=config.MODEL.num_class, 
            use_dense=config.LOSS.is_embedding, 
            num_dims=config.LOSS.embedding_dim).to(device)
    
        # load checkpoint
        model.load_state_dict(torch.load(config.TEST.model_path))
        if len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids)
        model.eval()

    sys.stdout.write('\n')
   
    eval_adv_test_whitebox_multistep(model, device, test_loader, config.TEST.epsilon, config.TEST.step_size, np.arange(5, 51,5))

if __name__ == '__main__':
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    config = easydict.EasyDict(config)
    main(config)

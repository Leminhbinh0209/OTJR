
import os
import sys
import time
from time import time
from datetime import timedelta

import torch
import torch.nn as nn

from datasets import get_cifar10_dataloaders, get_cifar100_dataloaders, get_intel_dataloaders
from helper.utils import  setup_device, Writer
from models.wideresnet import *
from torchattacks import *
import yaml
import easydict
import argparse

parser = argparse.ArgumentParser(description='Auto Attacks')
parser.add_argument('--config', '--cf', type=str,  
                    default="../configs/TEST_CIFAR10.yaml", 
                    help='config file directory')
        
args = parser.parse_args()
att_steps = 25
epsilons = 8.0 / 255.0
alpha = 2.0/255.0

def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.SYS.GPUs
    device, device_ids = setup_device(config.SYS.num_gpus)
    use_cuda =  torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # logger file 
    log_path =  os.path.join(config.TEST.out_folder, config.dataset, config.model_name, config.LOSS.loss)
    os.makedirs(log_path, exist_ok = True)
    logger  = Writer(log_path + "/auto_attack.log")
    # set up data loader

    ####### Data loader #######
    if config.dataset == 'cifar10':
        config.MODEL.num_class = 10
        _, test_loader  = get_cifar10_dataloaders(config)
    elif config.dataset == 'cifar100':
        _, test_loader  = get_cifar100_dataloaders(config)
        config.MODEL.num_class = 100
    elif config.dataset == 'intel':
        print("Loading the INTEL dataset .. ")
        _, test_loader  = get_intel_dataloaders(config)
        config.MODEL.num_class = 6
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
    else: 
        raise RuntimeError(f"Undefined model name: {config.model_name}")

    atks = [
        FGSM(model, eps=epsilons),
        BIM(model, eps=epsilons, alpha=alpha, steps=att_steps),
        CW(model, c=1, lr=0.01, steps=att_steps, kappa=0),
        PGD(model, eps=epsilons, alpha=alpha, steps=att_steps, random_start=True),
        PGDL2(model, eps=1, alpha=alpha, steps=att_steps),
        MIFGSM(model, eps=epsilons, alpha=alpha, steps=att_steps, decay=0.1),
        FAB(model, eps=epsilons, steps=att_steps, n_classes=config.MODEL.num_class, n_restarts=1, targeted=False),
        FAB(model, eps=epsilons, steps=att_steps, n_classes=config.MODEL.num_class, n_restarts=1, targeted=True),
        Square(model, eps=epsilons, n_queries=700, n_restarts=1, loss='ce', p_init=1.0),
        Square(model, eps=epsilons, n_queries=500, n_restarts=1, loss='ce', p_init=1.0),
        Square(model, eps=epsilons, n_queries=200, n_restarts=1, loss='ce', p_init=1.0 ),
        Square(model, eps=epsilons, n_queries=100, n_restarts=1, loss='ce', p_init=1.0 ),
    ]
    ### Targeted attack

 
    logger.log("Adversarial Image & Predicted Label")
    for atk in atks :
        logger.log("-"*70)
        logger.log(str(atk))
  
        correct = 0
        correct_nat = 0
        total = 0
        tic_e = time()
        for batch_id, (images, labels) in enumerate(test_loader):
            adv_images = atk(images, labels) 
            labels = labels.to(device)
            outputs = model(adv_images)
            _, pre = torch.max(outputs.data, 1)

            out_nat = model(images.to(device))
            _, pre_nat = torch.max(out_nat.data, 1)

            total += images.size(0)
            correct += (pre == labels).sum()
            correct_nat += (pre_nat == labels).sum()

            eta =  str(timedelta(seconds=int((time()-tic_e) / (batch_id+1) * (len(test_loader)-batch_id-1)))) 
            sys.stdout.write("\r")  
            sys.stdout.flush()
            sys.stdout.write('Bath ID: {}, ETA: {}'.format(batch_id+1, eta))

        sys.stdout.write("\n")
        logger.log('Natural accuracy: %.2f %%' % (100 * float(correct_nat) / total))
        logger.log('Robust accuracy: %.2f %%' % (100 * float(correct) / total))

if __name__ == "__main__":
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    config = easydict.EasyDict(config)
    main(config)

              
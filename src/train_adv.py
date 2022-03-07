import warnings
warnings.filterwarnings("ignore")
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from time import time
from datetime import timedelta
import easydict
import yaml
from functools import partial
from models.wideresnet import *
from models import resnet as Resnets
from losses import *
from helper.utils import adjust_lr_cifar10,  adjust_lr_intel, setup_device, Logger, bcolors
from helper.loops import train, eval_test

from datasets import get_cifar10_dataloaders, get_cifar100_dataloaders
from datasets import  get_intel_dataloaders

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
parser = argparse.ArgumentParser(description='Adversarial Training')
parser.add_argument('--config', '--cf', type=str,  
                    default="../configs/CIFAR10.yaml", 
                    help='config file directory')

args = parser.parse_args()

def eval_test_func(model, device, test_loader):
    """
    Evaluate data without adversarial samples
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def main(config):
    import wandb
    #######  Set up GPUs and benchmark model #######
    os.environ["CUDA_VISIBLE_DEVICES"] = config.SYS.GPUs
    device, device_ids = setup_device(config.SYS.num_gpus)

    use_cuda = config.SYS.use_cuda and torch.cuda.is_available()
    np.random.seed(config.SYS.random_seed)
    torch.manual_seed(config.SYS.random_seed)
    torch.cuda.manual_seed(config.SYS.random_seed)
    torch.cuda.manual_seed_all(config.SYS.random_seed) 
    os.environ['PYTHONHASHSEED'] = str(config.SYS.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # torch.set_num_threads()
    
    ####### Model checkpoints and logging #######
    model_dir = os.path.join(config.checkpoint_dir, config.dataset, config.model_name, config.LOSS.loss, config.running_name)
    logging_dir = os.path.join(config.logging_dir, config.dataset, config.model_name, config.LOSS.loss, config.running_name)
    os.makedirs(model_dir, exist_ok = True)
    os.makedirs(logging_dir, exist_ok = True)

    ####### Data loader #######
    
    if config.dataset == 'cifar10':
        config.MODEL.num_class = 10
        train_loader, test_loader  = get_cifar10_dataloaders(config)
    elif config.dataset == 'cifar100':
        train_loader, test_loader  = get_cifar100_dataloaders(config)
        config.MODEL.num_class = 100
    elif config.dataset == 'intel':
        train_loader, test_loader  = get_intel_dataloaders(config)
        config.MODEL.num_class = 6
    else: 
        raise RuntimeError(f"Undefined dataset: {config.dataset}")
    #######  Define training style function #######
    if config.model_name in ['wideresnet_34', 'wideresnet_28']:
        print("MODEL: ", config.model_name )
        model = WideResNet(eval(config.model_name.split('_')[-1]), 
            num_classes=config.MODEL.num_class, 
            use_dense=config.LOSS.is_embedding, 
            num_dims=config.LOSS.embedding_dim).to(device)   
    elif 'resnet' in config.model_name:
        # Surrogate model
        print(f"MODEL: {config.model_name}")
        model =  Resnets.__dict__[config.model_name](pretrained=True, num_classes=config.MODEL.num_class).to(device)  
        config.LOSS.is_embedding = False 
    else: 
        raise RuntimeError(f"Undefined model name: {config.model_name}")

    if config.LOSS.is_embedding:
        print("Change the latent dimension from {} to {}".format(config.MODEL.latent_dim, config.LOSS.embedding_dim))
        config.MODEL.latent_dim = config.LOSS.embedding_dim

    ####### Reload the checkpoint model #######
    if config.TRAIN.TRANSFER.is_transfer:
        print(f"{bcolors.OKCYAN} Load pretrain weights {config.TRAIN.TRANSFER.checkpoint_dir}...\n Num epoch training {config.TRAIN.epochs}  {bcolors.ENDC}")
        model.load_state_dict(torch.load(config.TRAIN.TRANSFER.checkpoint_dir))
        config.TRAIN.start_epoch = config.TRAIN.TRANSFER.start_epoch + 1

    init_logs = True
    if config.TRAIN.RESUME.is_resume: # RELOAD previous model from checkpoint
            print("Resume and start training model from the epoch {}".format(config.TRAIN.RESUME.start_epoch))
            model.load_state_dict(torch.load(config.TRAIN.RESUME.checkpoint_dir))
            config.TRAIN.start_epoch = config.TRAIN.RESUME.start_epoch + 1
            if config.TRAIN.RESUME.logging_dir:
                print("Continue to write on file {}".format(config.TRAIN.RESUME.logging_dir))
                logger = Logger(config.TRAIN.RESUME.logging_dir, resume=True)
                init_logs = False
                
    if init_logs: 
        logger = Logger(os.path.join(logging_dir, '_wideres-logging.txt'))
        logger.set_names(['epoch', 'lr', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    ### Parallel the model
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    if config.TRAIN.watchdb:
        ####### Wandb to watch the performance of the model #######
        wan_id = wandb.util.generate_id()
        wandb = wandb.init(config=config, 
                            project=f"{config.project}-{config.dataset}-{config.model_name}",
                            id=wan_id,
                            dir=logging_dir,
                            name=config.LOSS.loss.upper())
        ####### Watch out the model by wandb #######
        wandb.watch(model)
    
    optimizer = optim.SGD(model.parameters(), 
                        lr=config.TRAIN.lr,
                        momentum=config.TRAIN.momentum, 
                        weight_decay=config.TRAIN.weight_decay) 
    # learning rate schedule
    if 'cifar' in config.dataset:
        adjust_learning_rate = partial(adjust_lr_cifar10, optimizer=optimizer, config=config)  
    elif config.dataset in [ 'intel']:
        print("Step learning rate schedule")
        adjust_learning_rate = partial(adjust_lr_intel, optimizer=optimizer, config=config)   
    ########## Define loss function #######
    if 'xe' == config.LOSS.loss:
        print("Natural training without adversarial")
        criterion = nn.CrossEntropyLoss()
        criterion.loss = "xe"

    elif 'alp' in config.LOSS.loss:
        criterion = ALPLoss(optimizer=optimizer, 
                        config = config,
                        device=device)
    elif 'trades' in  config.LOSS.loss :
        criterion = TRADESLoss(optimizer=optimizer, 
                        config = config,
                        device=device)

    elif 'pgd' in config.LOSS.loss :    
        criterion = PGDLoss(optimizer=optimizer, 
                        config = config,
                        device=device)
    elif 'sink' == config.LOSS.loss :    
        criterion = SinkhornDiv(optimizer=optimizer, 
                        config = config,
                        device=device)

    elif 'trans' in config.LOSS.loss or 'jac' in config.LOSS.loss :    
        ### Load a pre-trained model 
        criterion = TransportJac(optimizer=optimizer, 
                        config = config,
                        device=device)
    else:
        raise RuntimeError(f"Undefined loss function: {config.LOSS.loss}")

    ####### Start training #######
    tic_e = time()
    
    for epoch in range(config.TRAIN.start_epoch, config.TRAIN.epochs + 1):
        ####### adjust learning rate for SGD #######
        adjust_learning_rate(epoch=epoch)
        ####### adversarial training #######
        traintime_tic = time()
        train(config , model, device, train_loader, optimizer, epoch, criterion)
        traintime_eta =  str(timedelta(seconds=int((time()-traintime_tic))))
        print(f"Time / epoch: {traintime_eta}")

        sys.stdout.write('\n')
        train_loss, train_acc = eval_test(model, device, train_loader, criterion,  infor='train')
        test_loss, test_acc = eval_test(model, device, test_loader, criterion,  infor='test')
        ####### screen printing #######
        sys.stdout.write('\nTrain loss: {:.4f}, Train acc:  {:.2f}% \nTest loss: {:.4f}, Test acc:  {:.2f}%'.format(
        train_loss, train_acc*100.0, test_loss, test_acc*100.0))
        eta =  str(timedelta(seconds=int((time()-tic_e) / (epoch+1-config.TRAIN.start_epoch) * (config.TRAIN.epochs-epoch)))) 
        print('\n============== ETA: {} =============='.format(eta))
        ####### logging data #######
        if config.TRAIN.watchdb:
            report = {'train_loss':train_loss , 'train_acc':train_acc, 'val_loss':test_loss, 'val_acc':test_acc, 'lr':optimizer.param_groups[0]['lr']}
            wandb.log(report)
        logger.append([epoch, optimizer.param_groups[0]['lr'], train_loss, train_acc*100.0, test_loss, test_acc*100.0])

        ####### save checkpoint #######
        if (epoch >= config.TRAIN.start_freq) and (epoch % config.TRAIN.save_freq == 0):
            torch.save(obj = model.module.state_dict() if len(device_ids) > 1 else  model.state_dict(),
                       f = os.path.join(model_dir, '_{}-epoch{}.pt'.format(config.model_name, epoch)))
    logger.close()
if __name__ == '__main__':
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    config = easydict.EasyDict(config)
    main(config)
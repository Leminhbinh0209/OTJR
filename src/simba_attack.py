import warnings
warnings.filterwarnings("ignore")
import os
from time import time
from datetime import timedelta
import easydict
import yaml

from helper.utils import  setup_device, Writer
from datasets import get_cifar10_dataloaders, get_cifar100_dataloaders
from helper.utils import  setup_device, Writer
from models.wideresnet import *
from SimBA.simba import SimBA
import argparse


parser = argparse.ArgumentParser(description='SimBA Attack')
parser.add_argument('--config', '--cf', type=str,  
                    default="../configs/TEST_CIFAR10.yaml", 
                    help='config file directory')
        
args = parser.parse_args()

global sim_dict
sim_dict =  easydict.EasyDict({})
sim_dict.num_runs = 1000 # Not use
sim_dict.num_iters = 500
sim_dict.log_every = 100
sim_dict.epsilon = 8/255
sim_dict.linf_bound = 0.0
sim_dict.freq_dims = 32 # window_size
sim_dict.order = 'rand'
sim_dict.stride = 7 # Not use
sim_dict.targeted = False
sim_dict.pixel_attack = False



def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.SYS.GPUs
    device, device_ids = setup_device(config.SYS.num_gpus)
    use_cuda =  torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # logger file 
    log_path =  os.path.join(config.TEST.out_folder, config.dataset, config.model_name, config.LOSS.loss)
    os.makedirs(log_path, exist_ok = True)
    logger  = Writer(log_path + f"/simba_{sim_dict.num_iters}.log")
    # set up data loader

    ####### Data loader #######
    if config.dataset == 'cifar10':
        _, test_loader  = get_cifar10_dataloaders(config)
    elif config.dataset == 'cifar100':
        _, test_loader  = get_cifar100_dataloaders(config)
    # set models
    n_classes = 10 if config.dataset=='cifar10' else 100
    if config.model_name in ['wideresnet_34', 'wideresnet_28']:
        model = WideResNet(34 if config.model_name=='wideresnet_34'else 28, 
            num_classes=n_classes, 
            use_dense=config.LOSS.is_embedding, 
            num_dims=config.LOSS.embedding_dim).to(device)
    
        # load checkpoint
        model.load_state_dict(torch.load(config.TEST.model_path))
        if len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids)
        model.eval()
    else: 
        raise RuntimeError(f"Undefined model name: {config.model_name}")
    # definbe attaker model
    attacker = SimBA(model, 'cifar10', 32)

    if sim_dict.order == 'rand':
        n_dims = 3 * sim_dict.freq_dims * sim_dict.freq_dims
    else:
        n_dims = 3 * config.MODEL.input_size * config.MODEL.input_size
    if sim_dict.num_iters > 0:
        max_iters = int(min(n_dims, sim_dict.num_iters))
    else:
        max_iters = int(n_dims)
    tic = time()
    correct, total = 0.0, 0.0
    for batch_id, (images, labels) in enumerate(test_loader):
        labels = labels
        images = images
        y_preds = attacker.simba_batch(
            images, labels, max_iters, sim_dict.freq_dims, sim_dict.stride, sim_dict.epsilon, linf_bound=sim_dict.linf_bound,
            order=sim_dict.order, targeted=sim_dict.targeted, pixel_attack=sim_dict.pixel_attack, log_every=sim_dict.log_every)

        correct += (y_preds == labels).sum()
        total += images.size(0)

        eta = str(timedelta(seconds=int((time()-tic) / (batch_id+1) * (len(test_loader) - 1 - batch_id))))
        sys.stdout.write("\r")
        sys.stdout.flush()
        sys.stdout.write('[{:03d}/{:03d}]   Eta: {}'.format( batch_id+1, len(test_loader), eta))
        correct += (y_preds == labels).sum()
        total += images.size(0)

    logger.log('Robust accuracy: %.2f %%' % (100 * float(correct) / total))
if __name__ == "__main__":
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    config = easydict.EasyDict(config)   
    main(config)
                   





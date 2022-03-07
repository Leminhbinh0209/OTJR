from __future__ import print_function, division
import sys
sys.path.append('../')
from time import time
from datetime import timedelta
import torch
import torch.nn.functional as F

def train(config, model, device, train_loader, optimizer, epoch, criterion) -> None:
    model.train()
    tic = time()
    for batch_idx, data in enumerate(train_loader):
        input, target = data
        input, target  = input.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        # calculate robust loss
        callback = dict()
        if 'xe' ==  config.LOSS.loss:
            loss = criterion(model(input), target)
        else:
            loss = criterion(model, input, target, callback)  
        # exclude loss for free attack
        if loss is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        call = ',  '.join(["{}: {:.5f}".format(k, v) for k,v in callback.items()])
        eta = str(timedelta(seconds=int((time()-tic) / (batch_idx+1) * (len(train_loader) - 1 - batch_idx))))
        sys.stdout.write("\r")
        sys.stdout.flush()
        sys.stdout.write('Train Epoch: {:03d} [{:03d}/{:03d} ({:2.0f}%)]   Loss: {:.6f},  Eta: {}.   {}'.format(
            epoch, batch_idx+1, len(train_loader),
                   100. * (batch_idx+1) / len(train_loader), loss.item() if loss is not None else 0, eta, call))
        
def eval_test(model, device, val_loader, criterion=None, infor='') : 
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            input, target = data
            input, target = input.to(device), target.to(device)
            
            # Use max-mahalanobis to predict
            if criterion is not None and 'mvw' in criterion.loss:
                penultimate_ft,  logits = model(input, return_feature=True)
                loss_, x_probs = criterion.vmf_pred(penultimate_ft, logits, labels=target, idx=None, return_prob=True, use_memory=criterion.use_vmf)
                test_loss += loss_.item() * target.shape[0]
                pred = x_probs.max(1, keepdim=True)[1] # closest centers
                correct += pred.eq(target.view_as(pred)).sum().item()
            else:
                output = model(input)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
            sys.stdout.write("\r")
            sys.stdout.flush()
            sys.stdout.write('Eval {}: [{:03d}/{:03d} ({:2.0f}%)]'.format(infor, idx+1, len(val_loader),  100. * (idx+1) / len(val_loader)))
    test_loss /= len(val_loader.dataset)
    test_accuracy = correct / len(val_loader.dataset)
    return test_loss, test_accuracy
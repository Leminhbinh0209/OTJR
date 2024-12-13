# Bridging Optimal Transport and Jacobian Regularization by Optimal Trajectory for Enhanced Adversarial Defense
Binh M. Le, Shahroz Tariq, and Simon S. Woo, "Bridging Optimal Transport and Jacobian Regularization by Optimal Trajectory for Enhanced Adversarial Defense" (**Oral**) <br /> 
17th Asian Conference on Computer Vision (ACCV) 2024 <br /> 


## 1. Installation
- Ubuntu 18.04.5 LTS
- CUDA 11.1
- Python 3.6.13
- Python packages are detailed separately in ```requirements.txt```.

## 2.Datasets
Download the following datasets, then extract to folder ```./data```:
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
- [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)
- [INTEL](https://www.kaggle.com/puneet6060/intel-image-classification) 

## 3. Train model
In the configuration file ```./configs/CIFAR10.yaml```, modify:
+ dataset type: ```dataset``` can be in [```cifar10, cifar100```]
+ model type: ```model_name``` can be in [```wideresnet_28, wideresnet_34```]
+ loss function: ```loss (line 55)``` can be in:
  + ```xe```: cross-entropy loss
  + ```trades```: TRADES
  + ```pgd```: PGD-AT
  + ```alp```: ALP
  + ```rand_jac```: Random Jacobian regularization
  + ```sw_jac```: Optimal Jacobian regularization
  + ```trans_swdj```: OTJR
  
Start training the defense model:
```
$ python src/train_adv.py --config ../configs/CIFAR10.yaml
```
## 4. Attack model
In the configuration file ```./configs/TEST_CIFAR10.yaml```, modify:
+ model's checkpoint: ```model_path```


Run white-box and square attack:
```
$ python src/auto_attack.py --config ../configs/TEST_CIFAR10.yaml
```


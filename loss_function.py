import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import math


def log_sm_loss(input, target):
    eout = torch.exp(input)
    seout = torch.sum(eout,dim=1)

    loss = torch.log(seout)
    tmp = torch.full_like(input, fill_value=0.0)
    tmp.scatter_(dim=1, index=target.unsqueeze(1), value=1.0)
    
    tmp *= input
    loss -= torch.sum(tmp,dim=1)
    
    return loss.sum() / target.shape[0]


def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)


def std_loss(input, target):
    tmp = torch.full_like(input, fill_value = -0.5)
    tmp.scatter_(dim = 1, index = target.unsqueeze(1), value = 0.5)
    dif = input - tmp
    dif2 = dif * dif
    adif2 = dif2.sum() / (input.shape[0] * input.shape[1] - 1.0)
    std = torch.sqrt(adif2)
    
    return std

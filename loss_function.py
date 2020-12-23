import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import math


# loss, which is minus log of main (target) probability
# pytorch calls it cross entropy loss
# gives same results as their's - tested
def log_sm_loss(input, target):
    eout = torch.exp(input)
    seout = torch.sum(eout,dim=1)

    loss = torch.log(seout)
    tmp = torch.zeros(input.shape, requires_grad=False)
    for i in range(input.shape[0]):
        tmp[i,target[i]] = 1.0
    
    tmp = tmp.to('cuda')
    tmp *= input
    loss -= torch.sum(tmp,dim=1)
    
    return loss.sum() / target.shape[0]


# standard deviation loss
# works pretty well, one run gave me 95.35 % accuracy [95.2 - 95.8 % for log_sm_loss]
# second run 95.34 % accuracy
# third run 95.19 % accuracy
# optimizing power std^0.5 - one run of 95.50 % (makes it quite competitive with cross entropy)
# optimizing power std^0.1 - one run of 95.18 %
def std_loss(input, target):
    # here is bit of cheating - I ask for outputs to be -0.5 for incorrect and 0.5 for correct cases
    tmp = torch.ones(input.shape, requires_grad=False) * (-0.5)
    for i in range(input.shape[0]):
        tmp[i,target[i]] = 0.5
    
    tmp = tmp.to('cuda')
    tmp = input - tmp
    tmp2 = tmp * tmp
    tmp3 = tmp2.sum() / (input.shape[0] * input.shape[1] - 1.0)
    tmp4 = torch.sqrt(tmp3)
    
    return tmp4


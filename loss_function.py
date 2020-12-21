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
def std_loss(input, target):
    # here is bit of cheating - I ask for outputs to be -1 for incorrect and 9 for correct cases
    # could be verified if other values work as well or
    # seems like these two values can be arbitrary, so picking nicer numbers
    tmp = torch.ones(input.shape, requires_grad=False) * (-0.5)
    for i in range(input.shape[0]):
        tmp[i,target[i]] = 0.5
    
    tmp = tmp.to('cuda')
    tmp = input - tmp
    tmp2 = tmp * tmp
    tmp3 = tmp2.sum() / (input.shape[0] * input.shape[1] - 1.0)
    tmp4 = torch.sqrt(tmp3)
    
    return tmp4

########################################################################################################
# all these might need better with some other distribution than normal?                                #
########################################################################################################

# this is actually working
# accuracy ~ 93.89 % - needs more work [like p(x>y) = p(x-y>0) instead of interscetion?]
# second run gave around ~ 93.85 %
def bell_curves_intersection_loss(input, target):
    tmp = torch.ones(input.shape, requires_grad=False) * (-1.0)
    for i in range(input.shape[0]):
        tmp[i,target[i]] = 9.0
    
    tmp = tmp.to('cuda')
    tmp = input - tmp
    tmp2 = tmp * tmp
    tmp3 = tmp2.sum() / (input.shape[0] * input.shape[1] - 1.0)
    tmp4 = torch.sqrt(tmp3)
    
    tmp5 = torch.tensor(range(10001), requires_grad=False).to('cuda') * 24.0 / 10000.0 - 8.0
    
    tmp6 = (tmp5 + 1.0) / tmp4
    tmp7 = torch.exp(-0.5 * tmp6 * tmp6) / (tmp4 * math.sqrt(2.0 * math.pi))
    tmp7 = tmp7.view(-1,1)

    tmp8 = (tmp5 - 9.0) / tmp4
    tmp9 = torch.exp(-0.5 * tmp8 * tmp8) / (tmp4 * math.sqrt(2.0 * math.pi))
    tmp9 = tmp9.view(-1,1)
    
    tmp10 = torch.minimum(tmp7,tmp9).sum()
    tmp11 = torch.maximum(tmp7,tmp9).sum()
    
    return tmp10 / tmp11

# this works as well, which is less surprising, though results are bit worse
# ~ 92.5 %
def diff_probs_loss(input, target):
    tmp = torch.ones(input.shape, requires_grad=False) * (-1.0)
    for i in range(input.shape[0]):
        tmp[i,target[i]] = 9.0
    
    tmp = tmp.to('cuda')
    tmp = input - tmp
    tmp2 = tmp * tmp
    tmp3 = tmp2.sum() / (input.shape[0] * input.shape[1] - 1.0)
    tmp4 = torch.sqrt(tmp3) * math.sqrt(2)
    
    tmp5 = torch.tensor(range(10001), requires_grad=False).to('cuda') * 20.0 / 10000.0
    
    tmp6 = (tmp5 - 10.0) / tmp4
    tmp7 = torch.exp(-0.5 * tmp6 * tmp6) / (tmp4 * math.sqrt(2.0 * math.pi))
    tmp7 = tmp7.view(-1,1)

    tmp10 = tmp7.sum() * 20.0 / 10000.0
    
    #print(tmp10)
    return 1.0 - tmp10

# ~ 92.58 % - might need to run longer than 200 epochs?
def log_diff_probs_loss(input, target):
    tmp = torch.ones(input.shape, requires_grad=False) * (-1.0)
    for i in range(input.shape[0]):
        tmp[i,target[i]] = 9.0
    
    tmp = tmp.to('cuda')
    tmp = input - tmp
    tmp2 = tmp * tmp
    tmp3 = tmp2.sum() / (input.shape[0] * input.shape[1] - 1.0)
    tmp4 = torch.sqrt(tmp3) * math.sqrt(2)
    
    tmp5 = torch.tensor(range(10001), requires_grad=False).to('cuda') * 20.0 / 10000.0
    
    tmp6 = (tmp5 - 10.0) / tmp4
    tmp7 = torch.exp(-0.5 * tmp6 * tmp6) / (tmp4 * math.sqrt(2.0 * math.pi))
    tmp7 = tmp7.view(-1,1)

    tmp10 = tmp7.sum() * 20.0 / 10000.0
    
    #print(tmp10)
    return - torch.log(tmp10)

########################################################################################################
# all these might need better with some other distribution than normal?                                #
########################################################################################################

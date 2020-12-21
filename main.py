'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import math

from models.densenet import *
from models.resnet import *
from models.wrn import *

import os
import argparse

from utils import progress_bar
from disp_results import display_results

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device', device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=12)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=12)

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
# works pretty well, one run gave me 95.35 % accuracy [95.3 - 95.8 % for log_sm_loss]
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
# all these might need better than normal distribution?                                                #
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
# all these might need better than normal distribution?                                                #
########################################################################################################

# Model
print('==> Building model..')
#net = DenseNet121()
net = ResNet18()
#net = WideResNet(16, 8, 0.0, in_channels=3, labels=10)

args.lr = 0.1
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    H = []
    Y = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        Y.append(targets)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        #loss = std_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        H.append(outputs.to('cpu'))

    H = torch.cat(H,0).detach().numpy()
    Y = torch.cat(Y,0).numpy()

    return train_loss / len(trainloader) / trainloader.batch_size, 100.*correct/total, H, Y


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    H = []
    Y = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            Y.append(targets)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            #loss = std_loss(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            H.append(outputs.to('cpu'))
            
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    
    H = torch.cat(H,0).detach().numpy()
    Y = torch.cat(Y,0).numpy()

    return test_loss / len(testloader) / trainloader.batch_size, 100.*correct/total, H, Y

tr_ls = []
tr_as = []
te_ls = []
te_as = []

for epoch in range(start_epoch, start_epoch+225):
    tr_l, tr_a, tr_H, tr_Y = train(epoch)
    te_l, te_a, te_H, te_Y = test(epoch)
    
    tr_ls.append(tr_l)
    tr_as.append(tr_a)
    te_ls.append(te_l)
    te_as.append(te_a)
    display_results(tr_as,te_as,tr_ls,te_ls, False, tr_H, tr_Y, te_H, te_Y, epoch)
    print(max(tr_as), "% ", max(te_as), "%")
    
    file1 = open("debug.txt", "a")  # append mode 
    file1.write(f"{epoch},{tr_l},{tr_a},{te_l},{te_a}\n") 
    file1.close()
    
    scheduler.step()


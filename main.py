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

from loss_function import *

import os
import argparse
import shutil
import glob

from utils.utils import progress_bar
from utils.disp_results import display_results

dirnum = 0
while os.path.isdir("./experiment_" + str(dirnum)):
    dirnum += 1
dirname = "./experiment_" + str(dirnum) + "/"
print(dirname)
os.mkdir(dirname)
os.mkdir(dirname + 'sc')
for file in glob.glob('./*.py'):
    shutil.copy(file, dirname + 'sc')
for file in glob.glob('./models/*.py'):
    shutil.copy(file, dirname + 'sc')

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

    return train_loss / len(trainloader), 100.*correct/total, H, Y


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
        if not os.path.isdir(dirname + 'checkpoint'):
            os.mkdir(dirname + 'checkpoint')
        torch.save(state, dirname + 'checkpoint/ckpt.pth')
        best_acc = acc
    
    H = torch.cat(H,0).detach().numpy()
    Y = torch.cat(Y,0).numpy()

    return test_loss / len(testloader), 100.*correct/total, H, Y

tr_ls = []
tr_as = []
te_ls = []
te_as = []

num_epochs = 225
for epoch in range(start_epoch, start_epoch+num_epochs):
    tr_l, tr_a, tr_H, tr_Y = train(epoch)
    te_l, te_a, te_H, te_Y = test(epoch)
    
    tr_ls.append(tr_l)
    tr_as.append(tr_a)
    te_ls.append(te_l)
    te_as.append(te_a)
    display_results(tr_as,te_as,tr_ls,te_ls, False, tr_H, tr_Y, te_H, te_Y, epoch, dirname)
    print(max(tr_as), "% ", max(te_as), "%")
    
    file1 = open(dirname + "debug.txt", "a")  # append mode 
    file1.write(f"{epoch},{tr_l},{tr_a},{te_l},{te_a}\n") 
    file1.close()
    
    scheduler.step()

os.system(f"ffmpeg -r 10 -f image2 -s 2500x1200 -start_number {start_epoch} -i {dirname}tmp/epoch_%04d.png -vframes {num_epochs} -vcodec libx264 -crf 25  -pix_fmt yuv420p {dirname}test.mp4")


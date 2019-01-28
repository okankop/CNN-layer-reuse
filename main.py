'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10, cifar100, fashion-mnist]')
parser.add_argument('--model', default='lrunet', type=str, help='model = [lrunet, shufflenet, shufflenetv2, mobilenet, mobilenetv2]')
parser.add_argument('--layer_reuse', default=8, type=int, help='layer reuse')
parser.add_argument('--width_mult', default=1.0, type=float, help='width multiplier')
parser.add_argument('--drop', default=0.5, type=float, help='applied dropout')
parser.add_argument('--groups', default=3, type=int, help='The number of groups at group convolution at ShuffleNet')
parser.add_argument('--resume_path', default='', type=str, help='Save data (.pth) of previous training')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

if(args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset    = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset     = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 10
    init_ch     = 3 # Number of channels for the first conv layer
elif(args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset    = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset     = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 100
    init_ch     = 3 # Number of channels for the first conv layer
elif(args.dataset == 'fashionmnist'):
    print("| Preparing FashionMNIST dataset...")
    sys.stdout.write("| ")
    trainset    = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    testset     = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
    num_classes = 10
    init_ch     = 1 # Number of channels for the first conv layer

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
if args.model == 'lrunet':
    net = LruNet(num_classes, args.width_mult, args.layer_reuse, args.drop, init_ch)
elif args.model == 'mobilenet':
    net = MobileNet(num_classes, args.width_mult, init_ch)
elif args.model == 'mobilenetv2':
    net = MobileNetV2(num_classes, args.width_mult, init_ch)
elif args.model == 'shufflenet':
    net = ShuffleNet(num_classes, args.width_mult, args.groups, init_ch)
elif args.model == 'shufflenetv2':
    net = ShuffleNetV2(num_classes, args.width_mult, init_ch)
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
print(net)

conv_params = 0
for key in net.modules():
    if (isinstance(key, nn.Conv2d) | isinstance(key, nn.Linear)):
        #print(key)
        conv_params += sum(p.numel() for p in key.parameters() if p.requires_grad)
print("Total number of convolution parameters: ", conv_params)

pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Total number of trainable parameters: ", pytorch_total_params)

if args.resume_path:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.resume_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


# Training
def train(epoch):
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

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
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+275):
    train(epoch)
    test(epoch)

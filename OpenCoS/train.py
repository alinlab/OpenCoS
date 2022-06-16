#!/usr/bin/env python3 -u

from __future__ import print_function

import argparse
import csv
import os, logging
import copy
import random

import numpy as np
import torch
from torch.autograd import Variable, grad
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

import models
from utils import progress_bar, set_logging_defaults
from datasets import load_dataset
from collections import OrderedDict

# torch_version = [int(v) for v in torch.__version__.split('.')]
tensorboardX_compat = True #(torch_version[0] >= 1) and (torch_version[1] >= 1) # PyTorch >= 1.1
try:
    from tensorboardX import SummaryWriter
except ImportError:
    print ('No tensorboardX package is found. Start training without tensorboardX')
    tensorboardX_compat = False
    #raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")

parser = argparse.ArgumentParser(description='SimCLR linear evaluation or fine-tuning Training')
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--model', default="wide_resnet", type=str,
                    help='model type (default: wide_resnet)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--batch-size', default=64, type=int, help='batch size')
parser.add_argument('--num_iters', default=50000, type=int, help='total epochs to run')
parser.add_argument('--decay', default=0, type=float, help='weight decay')
parser.add_argument('--ngpu', default=1, type=int, help='number of gpu')
parser.add_argument('--sgpu', default=0, type=int, help='gpu index (start)')
parser.add_argument('--dataset', default='cifar10', type=str, help='the name for dataset')
parser.add_argument('--udata', default='svhn', type=str, help='type of unlabel data')
parser.add_argument('--tinyroot', default='/data/tinyimagenet/tiny-imagenet-200/', type=str, help='TinyImageNet directory')
parser.add_argument('--imgroot', default='/data/ILSVRC/Data/CLS-LOC/', type=str, help='unlabel data directory')
parser.add_argument('--dataroot', default='/data/', type=str, help='data directory')
parser.add_argument('--saveroot', default='./results', type=str, help='data directory')
parser.add_argument('--finetune', '-ft', action='store_true', help='finetuning')
parser.add_argument('--pc', default=25, type=int, help='number of samples per class')
parser.add_argument('--nworkers', default=4, type=int, help='num_workers')

parser.add_argument('--multinomial', action='store_true', help='linear evaluation')
parser.add_argument('--stop_iters', default=None, type=int, help='early stopping')
parser.add_argument('--model_path', default=None, type=str, help='model path')
parser.add_argument('--ood_samples', default=0, type=int, help='number of ood samples in [0,10000,20000,30000,40000]')
parser.add_argument('--fix_optim', action='store_true', help='using optimizer of FixMatch')
parser.add_argument('--simclr_optim', action='store_true', help='using optimizer of SimCLR semi finetune')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()

best_val = 0  # best validation accuracy
start_iters = 0  # start from epoch 0 or last checkpoint epoch
current_val = 0

cudnn.benchmark = True

# Data
_labeled_trainset, _unlabeled_trainset, _labeled_testset = load_dataset(args.dataset, args.dataroot, batch_size=args.batch_size, pc=str(args.pc), method='default', uroot=args.udata, tinyroot=args.tinyroot, imgroot=args.imgroot, ood_samples=args.ood_samples, use_jitter=False)
_labeled_num_class = _labeled_trainset.num_classes
print('Numclass: ', _labeled_num_class)
print('==> Preparing dataset: {}'.format(args.dataset))
print('Number of label dataset: ' ,len(_labeled_trainset))
print('Number of unlabel dataset: ',len(_unlabeled_trainset))
print('Number of test dataset: ',len(_labeled_testset))


logdir = os.path.join(args.saveroot, args.dataset, args.model, args.name)
set_logging_defaults(logdir, args)
logger = logging.getLogger('main')
logname = os.path.join(logdir, 'log.csv')
if args.multinomial:
    tensorboardX_compat = False
if tensorboardX_compat:
    writer = SummaryWriter(logdir=logdir)

if use_cuda:
    torch.cuda.set_device(args.sgpu)
    print(torch.cuda.device_count())
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss()

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def train():
    # Model
    print('==> Building model: {}'.format(args.model))
    net = models.load_model(args.model, _labeled_num_class)
    if args.finetune:
        model_dict = net.state_dict()
        if (args.model in ['resnet50']):
            try:
                pretrained_dict = torch.load(args.model_path, map_location='cpu')['model']
            except KeyError:
                pretrained_dict = torch.load(args.model_path, map_location='cpu')['net']
            classifier = ['fc.weight', 'fc.bias']
            imagesize = 224
        elif (args.model in ['wide_resnet', 'CIFAR_ResNet50']):
            try:
                pretrained_dict = torch.load(args.model_path, map_location='cpu')['model']
            except KeyError:
                pretrained_dict = torch.load(args.model_path, map_location='cpu')['net']
            classifier = ['linear.weight', 'linear.bias']
            imagesize = 32
        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            if k[:6]=='module':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and k not in classifier}
        model_dict.update(new_state_dict)
        net.load_state_dict(model_dict)

    net.cuda()
    print('    Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))
    # print(net)
    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.sgpu, args.sgpu + args.ngpu)))

    if args.simclr_optim:
        assert (not args.fix_optim)
        args.lr = 0.05 * float(args.batch_size) / 256
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0, nesterov=True)
    elif args.fix_optim:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay, nesterov=True)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.lr) 

    net.train()
    if len(_labeled_trainset) < args.batch_size:
        rand_sampler = torch.utils.data.RandomSampler(_labeled_trainset, num_samples=args.batch_size, replacement=True)
        _labeled_trainloader = torch.utils.data.DataLoader(_labeled_trainset, batch_size=args.batch_size, sampler=rand_sampler, num_workers=0)
    else:
        _labeled_trainloader = torch.utils.data.DataLoader(_labeled_trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    _labeled_testloader = torch.utils.data.DataLoader(_labeled_testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    _labeled_train_iter = iter(cycle(_labeled_trainloader))
    train_loss = 0
    correct = 0
    total = 0

    run_iters = args.num_iters if args.stop_iters is None else args.stop_iters
    for batch_idx in range(start_iters, run_iters + 1):
        (inputs, inputs_aug), targets = next(_labeled_train_iter)

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        logits = net(inputs)

        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.fix_optim:
            adjust_learning_rate(optimizer, batch_idx+1)

        if batch_idx % 1000 == 0:
            if batch_idx // 1000 > (run_iters // 1000) - 5:
                median = True
            else:
                median = False
            logger = logging.getLogger('train')
            logger.info('[Iters {}] [Loss {:.3f}]'.format(
                batch_idx,
                train_loss/1000))
            print('[Iters {}] [Loss {:.3f}]'.format(
                batch_idx,
                train_loss/1000))
            if tensorboardX_compat:
                writer.add_scalar("training/loss", train_loss/1000, batch_idx+1)

            train_loss = 0
            save = val(net, batch_idx, _labeled_testloader, median=median)
            if save:
                checkpoint(net, optimizer, best_val, batch_idx)
            net.train()
        else:
            progress_bar(batch_idx % 1000, 1000, 'working...')

    checkpoint(net, optimizer, current_val, args.num_iters, last=True)


class MergeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2):
        assert len(dataset1)==len(dataset2)
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, i):
        return (self.dataset1[i][0],)+ self.dataset2[i]

    def __len__(self):
        return len(self.dataset1)


def multinomial():
    # Model
    print('==> Building model: {}'.format(args.model))
    net = models.load_model(args.model, _labeled_num_class)
    if args.finetune:
        model_dict = net.state_dict()
        if (args.model in ['resnet50']):
            try:
                pretrained_dict = torch.load(args.model_path, map_location='cpu')['model']
            except KeyError:
                pretrained_dict = torch.load(args.model_path, map_location='cpu')['net']
            classifier = ['fc.weight', 'fc.bias']
            imagesize = 224
        elif (args.model in ['wide_resnet', 'CIFAR_ResNet50']):
            try:
                pretrained_dict = torch.load(args.model_path, map_location='cpu')['model']
            except KeyError:
                pretrained_dict = torch.load(args.model_path, map_location='cpu')['net']
            classifier = ['linear.weight', 'linear.bias']
            imagesize = 32
        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            if k[:6]=='module':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and k not in classifier}
        model_dict.update(new_state_dict)
        net.load_state_dict(model_dict)

    net.cuda()
    print('    Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))
    # print(net)
    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.sgpu, args.sgpu + args.ngpu)))

    if (args.model in ['resnet50']):
        optimizer = optim.LBFGS(net.fc.parameters(), lr=1, max_iter=5000)
        transform_test = transforms.Compose([
            transforms.Resize(224), # for linear eval
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    elif (args.model in ['wide_resnet', 'CIFAR_ResNet50']):
        optimizer = optim.LBFGS(net.linear.parameters(), lr=1, max_iter=5000)
        transform_test = transforms.Compose([
            transforms.Resize(32), # for linear eval
            transforms.ToTensor(),
        ])

    net.eval()

    labeled_trainset = copy.deepcopy(_labeled_trainset.base_dataset) # full dataset
    labeled_trainset.transform = transform_test
    labeled_trainset = torch.utils.data.Subset(labeled_trainset, _labeled_trainset.indices) # slicing

    labeled_trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=min(args.batch_size, len(labeled_trainset)), shuffle=False, num_workers=4)
    labeled_testset = copy.deepcopy(_labeled_testset) # full dataset
    labeled_testset.transform = transform_test
    labeled_testloader = torch.utils.data.DataLoader(labeled_testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    train_loss = 0
    correct = 0
    total = 0

    feats = []
    labels = []
    for batch_idx, (inputs, targets) in enumerate(labeled_trainloader):
        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        with torch.no_grad():
            feats.append(net.feature(inputs))
            labels.append(targets)
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)


    def closure1():
        optimizer.zero_grad()
        outputs = net.fc(feats)
        loss = criterion(outputs, labels)
        for param in net.fc.parameters():
            loss += 0.5 * param.pow(2).sum() * 1e-4
        print('loss:', loss.item())
        loss.backward()
        return loss

    def closure2():
        optimizer.zero_grad()
        outputs = net.linear(feats)
        loss = criterion(outputs, labels)
        for param in net.linear.parameters():
            loss += 0.5 * param.pow(2).sum() * 1e-4
        print('loss:', loss.item())
        loss.backward()
        return loss

    if (args.model in ['resnet50']):
        optimizer.step(closure1)
    elif (args.model in ['wide_resnet', 'CIFAR_ResNet50']):
        optimizer.step(closure2)

    save = val(net, 100, labeled_testloader)
    checkpoint(net, optimizer, best_val, 100)

median_acc = []

def val(net, iters, testloader, median=False):
    global best_val
    global median_acc
    global current_val
    net.eval()
    val_loss = 0.0
    correct = 0.0
    total = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)
            loss = torch.mean(criterion(outputs, targets))
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().float()
            progress_bar(batch_idx, len(testloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))

    logger = logging.getLogger('test')
    logger.info('[Loss {:.3f}] [Acc {:.3f}]'.format(
        val_loss/(batch_idx+1), 100.*correct/total))

    acc = 100.*correct/total

    if median:
        median_acc.append(acc.item())
    if tensorboardX_compat:
        writer.add_scalar("validation/loss", val_loss/(batch_idx+1), iters+1)
        writer.add_scalar("validation/top1_acc", acc, iters+1)
    current_val = acc
    if acc > best_val:
        best_val = acc
        return True
    else:
        return False

def checkpoint(net, optimizer, acc, iters, last=False):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'acc': acc,
        'iters': iters,
        'rng_state': torch.get_rng_state()
    }
    torch.save(state, os.path.join(logdir, 'ckpt.t7' if (not last) else 'last_ckpt.t7'))


def adjust_learning_rate(optimizer, iters):
    """decrease the learning rate"""
    lr = args.lr * np.cos(iters/(args.num_iters+1) * (7 * np.pi) / (2 * 8))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if args.multinomial:
    multinomial()
else:
    train()

print("Best Accuracy : {}".format(best_val))
print("Median Accuracy : {}".format(np.median(median_acc)))
logger = logging.getLogger('best')
if args.multinomial:
    logger.info('[Acc {:.3f}]'.format(best_val))
else:
    logger.info('[Acc {:.3f}] [MEDIAN Acc {:.3f}]'.format(best_val, np.median(median_acc)))
if tensorboardX_compat:
    writer.close()

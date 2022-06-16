#!/usr/bin/env python3 -u

from __future__ import print_function

import argparse
import csv
import os
import json
import copy

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import data_loader
import model_loader
import models
from models.projector import Projector

from argument import parser, print_args
from utils import progress_bar, checkpoint

from loss import pairwise_similarity,NT_xent

# Download packages from following git #
# "pip install torchlars" or git from https://github.com/kakaobrain/torchlars, version 0.1.2
# git from https://github.com/ildoonet/pytorch-gradual-warmup-lr #
from torchlars import LARS
from warmup_scheduler import GradualWarmupScheduler

args = parser()
if args.local_rank == 0:
    print_args(args)

start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)

world_size = args.ngpu
torch.distributed.init_process_group(
    'nccl',
    init_method='env://',
    world_size=world_size,
    rank=args.local_rank,
)

# Data
if args.local_rank == 0:
    print('==> Preparing data..')
trainloader, traindst, testloader, testdst ,train_sampler = data_loader.get_dataset(args)
if args.local_rank == 0:
    print('Number of training data: ', len(traindst))

# Model
if args.local_rank == 0:
    print('==> Building model..')
torch.cuda.set_device(args.local_rank)
model = model_loader.get_model(args)
if args.model == 'wide_resnet':
    projector = Projector(expansion=0)
else:
    projector = Projector(expansion=4)

# Log and saving checkpoint information #
if not os.path.isdir('results') and args.local_rank % ngpus_per_node == 0:
    os.mkdir('results')
args.name += (args.train_type + '_' +args.model + '_' + args.dataset)
loginfo = 'results/log_' + args.name + '_' + str(args.seed)
logname = (loginfo+ '.csv')

if args.local_rank == 0:
    print ('Training info...')
    print (loginfo)

# Model upload to GPU # 
model.cuda()
projector.cuda()
model       = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model       = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
)
projector   = torch.nn.parallel.DistributedDataParallel(
                projector,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
)

ngpus_per_node = torch.cuda.device_count()
print(torch.cuda.device_count())
cudnn.benchmark = True
print('Using CUDA..')

# Aggregating model parameter & projection parameter #
model_params = []
model_params += model.parameters()
model_params += projector.parameters()

# LARS optimizer from KAKAO-BRAIN github
# "pip install torchlars" or git from https://github.com/kakaobrain/torchlars
base_optimizer  = optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=args.decay)
optimizer       = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)

# Cosine learning rate annealing (SGDR) & Learning rate warmup #
# git from https://github.com/ildoonet/pytorch-gradual-warmup-lr #
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=args.lr_multiplier, total_epoch=10, after_scheduler=scheduler_cosine)

def train(epoch):
    print('\nEpoch: %d' % epoch)

    scheduler_warmup.step()
    model.train()
    projector.train()
    train_sampler.set_epoch(epoch)

    train_loss = 0
    reg_loss = 0
    
    for batch_idx, ((inputs_1, inputs_2), targets) in enumerate(trainloader):
        inputs_1, inputs_2 = inputs_1.cuda() ,inputs_2.cuda()
        inputs  = torch.cat((inputs_1,inputs_2))

        outputs = projector(model(inputs))

        similarity  = pairwise_similarity(outputs,temperature=args.temperature) 
        loss        = NT_xent(similarity)

        train_loss += loss.data


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Reg: %.5f'
                     % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1)))

    return (train_loss/batch_idx, reg_loss/batch_idx)


def test(epoch):
    model.eval()
    projector.eval()

    test_loss = 0

    # Save at the last epoch #       
    if epoch == start_epoch + args.epoch - 1 and args.local_rank % ngpus_per_node == 0:
        checkpoint(model, test_loss, epoch, args, optimizer)
        checkpoint(projector, test_loss, epoch, args, optimizer, save_name_add='_projector')
    # Save at every 100 epoch #
    elif epoch > 1 and epoch %100 == 0 and args.local_rank % ngpus_per_node == 0:
        checkpoint(model, test_loss, epoch, args, optimizer, save_name_add='_epoch_'+str(epoch))
        checkpoint(projector, test_loss, epoch, args, optimizer, save_name_add=('_projector_epoch_' + str(epoch)))

    return (test_loss)


##### Log file #####
if args.local_rank % ngpus_per_node == 0:
    if os.path.exists(logname):
        os.remove(logname) 
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'reg loss'])


##### Training #####
for epoch in range(start_epoch, args.epoch):
    train_loss, reg_loss = train(epoch)
    _ = test(epoch)

    if args.local_rank % ngpus_per_node == 0:
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss.item(), reg_loss])



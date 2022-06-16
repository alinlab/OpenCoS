#!/usr/bin/env python3 -u

from __future__ import print_function

import argparse
import csv
import os, logging
import copy
from collections import OrderedDict

import numpy as np
import torch
from torch.autograd import Variable, grad
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from imbalanced import ImbalancedDatasetSampler
import models
from utils import progress_bar, set_logging_defaults
from datasets import load_dataset

# torch_version = [int(v) for v in torch.__version__.split('.')]
tensorboardX_compat = True #(torch_version[0] >= 1) and (torch_version[1] >= 1) # PyTorch >= 1.1
try:
    from tensorboardX import SummaryWriter
except ImportError:
    print ('No tensorboardX package is found. Start training without tensorboardX')
    tensorboardX_compat = False
    #raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")


parser = argparse.ArgumentParser(description='FixMatch + OpenCoS Training')
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
parser.add_argument('--dataroot', default='/data/', type=str, help='data directory')
parser.add_argument('--udata', default='svhn', type=str, help='type of unlabel data')
parser.add_argument('--tinyroot', default='/data/tinyimagenet/tiny-imagenet-200/', type=str, help='TinyImageNet directory')
parser.add_argument('--imgroot', default='/data/ILSVRC/Data/CLS-LOC/', type=str, help='unlabel data directory')
parser.add_argument('--saveroot', default='./results', type=str, help='data directory')
parser.add_argument('--finetune', '-ft', action='store_true', help='finetuning')
parser.add_argument('--pc', default=25, type=int, help='number of samples per class')
parser.add_argument('--ema', action='store_true', help='EMA training')
parser.add_argument('--nworkers', default=4, type=int, help='num_workers')
parser.add_argument('--mu', default=7, type=int, help='unlabeled batch / labeled batch')

parser.add_argument('--sim_path', default=None, type=str, help='saved similarity')
parser.add_argument('--model_path', default=None, type=str, help='(unsupervised) pretrained model path')
parser.add_argument('--ood_samples', default=0, type=int, help='number of ood samples in [0,10000,20000,30000,40000]')
parser.add_argument('--fix_optim', action='store_true', help='using optimizer of FixMatch')

parser.add_argument('--lmd_u', default=1., type=float, help='Lu loss weight')
parser.add_argument('--lmd_unif', default=1., type=float, help='smoothing loss weight')
parser.add_argument('--aux_divide', action='store_true', help='divide bn parameters')
parser.add_argument('--ths', default=1., type=float, help='parameter for threshold')
parser.add_argument('--ths_pred', default=0.95, type=float, help='parameter for threshold')
parser.add_argument('--rampup', action='store_true', help='using rampup for ood loss')
parser.add_argument('--temp_s2', default=1, type=float, help='temperature scaling')
parser.add_argument('--total_unlabel', action='store_true', help='using total unlabel data')
parser.add_argument('--stop_iters', default=None, type=int, help='early stopping')
parser.add_argument('--use_jitter', action='store_true', help='using jitter augmentation for unlabeled data')
parser.add_argument('--simclr_optim', action='store_true', help='using optimizer of SimCLR semi finetune')
parser.add_argument('--no_head', action='store_true', help='not using the mlp head of simclr')

parser.add_argument('--le_path', default=None, type=str, help='simclr-le path')
parser.add_argument('--top_ratio', default=0.1, type=float, help='top in-class samples')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()

best_val = 0  # best validation accuracy
best_val_ema = 0  # best validation accuracy
start_iters = 0  # start from epoch 0 or last checkpoint epoch
current_val = 0
current_val_ema = 0

cudnn.benchmark = True

# Data
_labeled_trainset, _unlabeled_trainset, _labeled_testset = load_dataset(args.dataset, args.dataroot, batch_size=args.batch_size, pc=str(args.pc), method='default', uroot=args.udata, tinyroot=args.tinyroot, imgroot=args.imgroot, ood_samples=args.ood_samples, use_jitter=args.use_jitter)
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

if tensorboardX_compat:
    writer = SummaryWriter(logdir=logdir)

if use_cuda:
    torch.cuda.set_device(args.sgpu)
    print(torch.cuda.device_count())
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss()
criterion_none = nn.CrossEntropyLoss(reduction='none')

def get_logit(net, dataloader):
    net.eval()
    emb = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            emb.append(outputs.cpu())
    emb = torch.cat(emb, dim=0)
    return emb

def get_embed(net, head, dataloader):
    net.eval()
    emb = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            if head is None:
                outputs = net(inputs, feature=True)[1]
            else:
                outputs = head(net(inputs, feature=True)[1])
            emb.append(outputs.cpu())
    emb = torch.cat(emb, dim=0)
    return emb

def get_embed_center(net, head, dataloader):
    net.eval()
    emb = []
    centers = [0 for c in range(_labeled_num_class)]
    cnt = [0 for c in range(_labeled_num_class)]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            if head is None:
                outputs = net(inputs, feature=True)[1]
            else:
                outputs = head(net(inputs, feature=True)[1])
            emb.append(outputs.cpu())
            for ii in range(targets.size(0)):
                cnt[targets[ii].item()] = cnt[targets[ii].item()] + 1
                centers[targets[ii].item()] = centers[targets[ii].item()] + outputs[ii].cpu()
    for c in range(_labeled_num_class):
        centers[c] = (centers[c] / cnt[c]).unsqueeze(0)
    centers = torch.cat(centers, dim=0)
    emb = torch.cat(emb, dim=0)
    return emb, centers

def cosine_similarity(x1, x2, eps=1e-12):
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

class MergeDataset1(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2):
        assert len(dataset1)==len(dataset2)
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, i):
        return self.dataset1[i][:1] + (self.dataset2[i][0].item(), )

    def __len__(self):
        return len(self.dataset1)

class MergeDataset2(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2):
        assert len(dataset1)==len(dataset2)
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, i):
        return self.dataset1[i][:2] + self.dataset2[i]

    def __len__(self):
        return len(self.dataset1)

class Projector(nn.Module):
    def __init__(self, dimsize=2048):
        super(Projector, self).__init__()

        self.linear_1 = nn.Linear(dimsize, dimsize)
        self.linear_2 = nn.Linear(dimsize, dimsize)

    def forward(self, x):
        output = self.linear_1(x)
        output = F.relu(output)
        output = self.linear_2(output)

        return output

def callback_get_label(dataset, idx):
    return dataset[idx][1]

def load_dataset_softlabel(_labeled_num_class, _labeled_trainset, _unlabeled_trainset, sim_path=None):
    print('Identifying out-of-class samples...')
    # Load checkpoint.
    if (args.model in ['resnet50', 'resnet50_auxbn']):
        net_t = models.load_model("resnet50", _labeled_num_class)
        transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        try:
            pretrained_dict = torch.load(args.model_path, map_location='cpu')['model']
        except KeyError:
            pretrained_dict = torch.load(args.model_path, map_location='cpu')['net']
        classifier = ['fc.weight', 'fc.bias', 'linear_rot.weight', 'linear_rot.bias']
        dimsize = 2048
    elif (args.model in ['CIFAR_ResNet50', 'CIFAR_ResNet50_AuxBN']):
        net_t = models.load_model('CIFAR_ResNet50', _labeled_num_class)
        transform_test = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
            ])
        try:
            pretrained_dict = torch.load(args.model_path, map_location='cpu')['model']
        except KeyError:
            pretrained_dict = torch.load(args.model_path, map_location='cpu')['net']
        classifier = ['linear.weight', 'linear.bias', 'linear_rot.weight', 'linear_rot.bias']
        dimsize = 2048
    elif (args.model in ['wide_resnet', 'wide_resnet_auxbn']):
        net_t = models.load_model('wide_resnet', _labeled_num_class)
        transform_test = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
            ])
        try:
            pretrained_dict = torch.load(args.model_path, map_location='cpu')['model']
        except KeyError:
            pretrained_dict = torch.load(args.model_path, map_location='cpu')['net']
        classifier = ['linear.weight', 'linear.bias', 'linear_rot.weight', 'linear_rot.bias']
        dimsize = 128
    model_dict = net_t.state_dict()
    new_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        if k[:6]=='module':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and k not in classifier}

    model_dict.update(new_state_dict)
    net_t.load_state_dict(model_dict)
    net_t.cuda()

    if args.no_head:
        head_t = None
    else:
        head_t = Projector(dimsize = dimsize)
        head_dict = head_t.state_dict()
        pretrained_dict = torch.load(args.model_path+'_projector', map_location='cpu')['model']
        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            if k[:6]=='module':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        new_state_dict = {k: v for k, v in new_state_dict.items() if k in head_dict}
        head_dict.update(new_state_dict)
        head_t.load_state_dict(head_dict)
        head_t.cuda()

    labeled_trainset = copy.deepcopy(_labeled_trainset.base_dataset) # full dataset
    labeled_trainset.transform = transform_test
    labeled_trainset = torch.utils.data.Subset(labeled_trainset, _labeled_trainset.indices) # slicing
    labeled_trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=256, shuffle=False, num_workers=4)
    unlabeled_trainset = copy.deepcopy(_unlabeled_trainset)
    for u_dataset in unlabeled_trainset.datasets:
        if type(u_dataset)==torch.utils.data.Subset:
            u_dataset.dataset.transform = transform_test
        else:
            u_dataset.transform = transform_test
    unlabeled_trainloader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=256, shuffle=False, num_workers=4)

    if sim_path is None:
        emb_l, center_l = get_embed_center(net_t, head_t, labeled_trainloader)
        emb_u = get_embed(net_t, head_t, unlabeled_trainloader)
        sim_l = cosine_similarity(emb_l, center_l) # N_l x C
        sim_u = cosine_similarity(emb_u, center_l) # N_u x C
        torch.save(sim_l, os.path.join(logdir, 'similarity_l.pt'))
        torch.save(sim_u, os.path.join(logdir, 'similarity_u.pt'))
    else:
        sim_l = torch.load(os.path.join(sim_path, 'similarity_l.pt'))
        sim_u = torch.load(os.path.join(sim_path, 'similarity_u.pt'))


    if args.le_path:
        del net_t, head_t
        # Load checkpoint.
        if (args.model in ['resnet50', 'resnet50_auxbn']):
            net_t = models.load_model("resnet50", _labeled_num_class)
        elif (args.model in ['CIFAR_ResNet50', 'CIFAR_ResNet50_AuxBN']):
            net_t = models.load_model('CIFAR_ResNet50', _labeled_num_class)
        elif (args.model in ['wide_resnet', 'wide_resnet_auxbn']):
            net_t = models.load_model('wide_resnet', _labeled_num_class)
        pretrained_dict = torch.load(args.le_path, map_location='cpu')
        model_dict = net_t.state_dict()
        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            if k[:6]=='module':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        classifier = []
        new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and k not in classifier}

        model_dict.update(new_state_dict)
        net_t.load_state_dict(model_dict)
        net_t.cuda()
    else:
        if (args.model in ['resnet50', 'resnet50_auxbn']):
            optimizer = optim.LBFGS(net_t.fc.parameters(), lr=1, max_iter=5000)
        elif (args.model in ['wide_resnet', 'CIFAR_ResNet50', 'wide_resnet_auxbn', 'CIFAR_ResNet50_AuxBN']):
            optimizer = optim.LBFGS(net_t.linear.parameters(), lr=1, max_iter=5000)
        else:
            raise NotImplementedError

        feats = []
        labels = []
        for batch_idx, (inputs, targets) in enumerate(labeled_trainloader):
            inputs = inputs.cuda()
            targets = targets.cuda()
            with torch.no_grad():
                feats.append(net_t.feature(inputs))
                labels.append(targets)
        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0)

        def closure1():
            optimizer.zero_grad()
            outputs = net_t.fc(feats)
            loss = criterion(outputs, labels)
            for param in net_t.fc.parameters():
                loss += 0.5 * param.pow(2).sum() * 1e-4
            print('loss:', loss.item())
            loss.backward()
            return loss

        def closure2():
            optimizer.zero_grad()
            outputs = net_t.linear(feats)
            loss = criterion(outputs, labels)
            for param in net_t.linear.parameters():
                loss += 0.5 * param.pow(2).sum() * 1e-4
            print('loss:', loss.item())
            loss.backward()
            return loss

        if args.top_ratio > 0:
            if (args.model in ['resnet50', 'resnet50_auxbn']):
                optimizer.step(closure1)
            elif (args.model in ['wide_resnet', 'CIFAR_ResNet50', 'wide_resnet_auxbn', 'CIFAR_ResNet50_AuxBN']):
                optimizer.step(closure2)
            else:
                raise NotImplementedError
            try:
                os.makedirs('./simclr_le_'+str(args.pc)+'pc')
            except FileExistsError:
                # directory already exists
                pass
            try:
                os.makedirs(os.path.join('./simclr_le_'+str(args.pc)+'pc', args.dataset))
            except FileExistsError:
                # directory already exists
                pass
            le_name = args.model_path.split('/')[-1]
            torch.save(net_t.state_dict(), os.path.join('./simclr_le_'+str(args.pc)+'pc', args.dataset, le_name+'_le'))

    if args.top_ratio > 0:
        scores_l = torch.max(sim_l, dim=1)[0]
        mean = torch.mean(scores_l)
        std = torch.std(scores_l)
        sample = (torch.max(sim_u, dim=1)[0] > mean - args.ths * std)

        id_sample_idx = torch.tensor(list(range(len(_unlabeled_trainset))))[sample]
        unlabeled_trainloader_subset = torch.utils.data.DataLoader(torch.utils.data.Subset(unlabeled_trainset, id_sample_idx), batch_size=256, shuffle=False, num_workers=4)

        logit_u = get_logit(net_t, unlabeled_trainloader_subset)
    else:
        logit_u = None
    del net_t

    return sim_l, sim_u, logit_u

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def ema_train():
    # Model
    print('==> Building model: {}'.format(args.model))
    net = models.load_model(args.model, _labeled_num_class, divide=args.aux_divide)

    if args.finetune:
        model_dict = net.state_dict()
        if (args.model in ['resnet50', 'resnet50_auxbn']):
            try:
                pretrained_dict = torch.load(args.model_path, map_location='cpu')['model']
            except KeyError:
                pretrained_dict = torch.load(args.model_path, map_location='cpu')['net']
            classifier = ['fc.weight', 'fc.bias', 'linear_rot.weight', 'linear_rot.bias']
            imagesize = 224
        elif (args.model in ['CIFAR_ResNet50', 'CIFAR_ResNet50_AuxBN', 'wide_resnet', 'wide_resnet_auxbn']):
            try:
                pretrained_dict = torch.load(args.model_path, map_location='cpu')['model']
            except KeyError:
                pretrained_dict = torch.load(args.model_path, map_location='cpu')['net']
            classifier = ['linear.weight', 'linear.bias', 'linear_rot.weight', 'linear_rot.bias']
            imagesize = 32

        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            if k[:6]=='module':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        new_state_dict = {k: v for k, v in new_state_dict.items() if k not in classifier}

        if (args.model in ['resnet50_auxbn']):
            tmp_state_dict = copy.deepcopy(new_state_dict)
            new_state_dict = OrderedDict()
            for k in model_dict.keys():
                pos = k.find('downsample') 
                if pos!=-1: 
                    if k[pos+13:pos+19]=='bn_aux': 
                        v = tmp_state_dict[k[:pos+12]+k[pos+19:]] 
                    elif k[pos+13:pos+15]=='bn': 
                        v = tmp_state_dict[k[:pos+12]+k[pos+15:]] 
                    else: 
                        v = tmp_state_dict[k] 
                else: 
                    pos = k.find('bn') 
                    if pos == -1:
                        if k in classifier:
                            continue
                        else:
                            v = tmp_state_dict[k] 
                    elif k[pos+4:pos+10]=='bn_aux': 
                        v = tmp_state_dict[k[:pos+3]+k[pos+10:]] 
                    elif k[pos+4:pos+6]=='bn': 
                        v = tmp_state_dict[k[:pos+3]+k[pos+6:]] 
                    else: 
                        print (pos) 
                        print (k) 
                        raise KeyError 
                new_state_dict[k] = v
        elif (args.model in ['CIFAR_ResNet50_AuxBN', 'wide_resnet_auxbn']):
            tmp_state_dict = copy.deepcopy(new_state_dict)
            new_state_dict = OrderedDict()
            for k in model_dict.keys():
                pos = k.find('shortcut') 
                if pos!=-1: 
                    if k[pos+11:pos+17]=='bn_aux': 
                        v = tmp_state_dict[k[:pos+10]+k[pos+17:]] 
                    elif k[pos+11:pos+13]=='bn': 
                        v = tmp_state_dict[k[:pos+10]+k[pos+13:]] 
                    else: 
                        v = tmp_state_dict[k] 
                else: 
                    pos = k.find('bn') 
                    if pos == -1:
                        if k in classifier:
                            continue
                        else:
                            v = tmp_state_dict[k] 
                    elif k[pos+4:pos+10]=='bn_aux': 
                        v = tmp_state_dict[k[:pos+3]+k[pos+10:]] 
                    elif k[pos+4:pos+6]=='bn': 
                        v = tmp_state_dict[k[:pos+3]+k[pos+6:]] 
                    else: 
                        print (pos) 
                        print (k) 
                        raise KeyError 
                new_state_dict[k] = v

        model_dict.update(new_state_dict)
        net.load_state_dict(model_dict)

    net_ema = copy.deepcopy(net)
    for param in net_ema.parameters():
        param.detach_()

    net.cuda()
    net_ema.cuda()
    print('    Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))
    # print(net)
    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.sgpu, args.sgpu + args.ngpu)))
        net_ema = torch.nn.DataParallel(net_ema, device_ids=list(range(args.sgpu, args.sgpu + args.ngpu)))

    ## fixmatch lr linear scaling
    if args.simclr_optim:
        assert (not args.fix_optim)
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0, nesterov=True)
    elif args.fix_optim:
        args.lr = args.lr * args.mu / 7.
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay, nesterov=True)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.lr) # weight decay in ema_optimizer

    ema_optimizer = WeightEMA(net, net_ema, alpha=0.999, wd=(not args.fix_optim and not args.simclr_optim))

    net.train()
    net_ema.train()

    sim_l, sim_u, logit_u = load_dataset_softlabel(_labeled_num_class, _labeled_trainset, _unlabeled_trainset, args.sim_path)
    scores_l = torch.max(sim_l, dim=1)[0]
    mean = torch.mean(scores_l)
    std = torch.std(scores_l)
    sample = (torch.max(sim_u, dim=1)[0] > mean - args.ths * std)

    id_sample_idx = torch.tensor(list(range(len(_unlabeled_trainset))))[sample]
    id_trainset = torch.utils.data.Subset(_unlabeled_trainset, id_sample_idx)
    id_trainloader = torch.utils.data.DataLoader(id_trainset, batch_size=args.batch_size * args.mu, shuffle=True, num_workers=args.nworkers, drop_last=True)
    id_train_iter = iter(cycle(id_trainloader))
    print('Number of (detected) ID unlabel dataset: ',len(id_sample_idx))

    ood_sample_idx = torch.tensor(list(range(len(_unlabeled_trainset))))[~sample]
    unlabeled_trainset = copy.deepcopy(_unlabeled_trainset)
    unlabeled_trainset = MergeDataset2(unlabeled_trainset, torch.utils.data.TensorDataset(sim_u))
    ood_trainset = torch.utils.data.Subset(unlabeled_trainset, ood_sample_idx)
    ood_trainloader = torch.utils.data.DataLoader(ood_trainset, batch_size=args.batch_size * args.mu, shuffle=True, num_workers=args.nworkers, drop_last=True)
    ood_train_iter = iter(cycle(ood_trainloader))
    print('Number of OOD unlabel dataset: ',len(ood_sample_idx))

    if args.top_ratio > 0:
        pred = torch.softmax(logit_u, dim=1)
        conf = torch.max(pred, dim=1)[0]
        topk_idx = conf.sort(descending=True)[1][:int(args.top_ratio * sample.long().sum().item())]
        topk_classes = torch.max(pred[topk_idx],dim=1)[1].reshape(-1).long()
        topk_idx = id_sample_idx[topk_idx]
        print('Number of top-k unlabel dataset: ',len(topk_idx))
        unlabeled_trainset = copy.deepcopy(_unlabeled_trainset)
        for u_dataset in unlabeled_trainset.datasets:
            if type(u_dataset)==torch.utils.data.Subset:
                u_dataset.dataset.transform = _labeled_trainset.base_dataset.transform
            else:
                u_dataset.transform = _labeled_trainset.base_dataset.transform

        topk_trainset = torch.utils.data.Subset(unlabeled_trainset, topk_idx)
        topk_trainset = MergeDataset1(topk_trainset , torch.utils.data.TensorDataset(topk_classes)) ## label replacement
        new_labeled_trainset = torch.utils.data.ConcatDataset([_labeled_trainset, topk_trainset])
    else:
        new_labeled_trainset = _labeled_trainset

    if len(new_labeled_trainset) < args.batch_size:
        rand_sampler = torch.utils.data.RandomSampler(new_labeled_trainset, num_samples=args.batch_size, eplacement=True)
        _labeled_trainloader = torch.utils.data.DataLoader(new_labeled_trainset, batch_size=args.batch_size, sampler=rand_sampler, num_workers=0)
    else:
        imbalance_sampler = ImbalancedDatasetSampler(new_labeled_trainset, callback_get_label=callback_get_label)
        _labeled_trainloader = torch.utils.data.DataLoader(new_labeled_trainset, batch_size=args.batch_size, sampler=imbalance_sampler, num_workers=args.nworkers, drop_last=True)
    _labeled_testloader = torch.utils.data.DataLoader(_labeled_testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    _labeled_train_iter = iter(cycle(_labeled_trainloader))


    train_loss = 0
    correct = 0
    total = 0

    run_iters = args.num_iters if args.stop_iters is None else args.stop_iters
    for batch_idx in range(start_iters, run_iters + 1):
        (inputs, inputs_strong), targets = next(_labeled_train_iter)
        (inputs_o, inputs_o_strong), _ = next(id_train_iter)
        if args.lmd_unif > 0:
            (inputs_s, inputs_s_strong), _, sim_ood = next(ood_train_iter)

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
            inputs_o = inputs_o.cuda()
            inputs_o_strong = inputs_o_strong.cuda()
            if args.lmd_unif > 0:
                inputs_s = inputs_s.cuda()
                sim_ood = sim_ood.cuda()
                targets_s = torch.softmax(sim_ood / args.temp_s2, dim=1)

        inputs_total = torch.cat([inputs, inputs_o, inputs_o_strong], dim=0)
        inputs_total = list(torch.split(inputs_total, args.batch_size))
        inputs_total = interleave(inputs_total, args.batch_size)

        logits = [net(inputs_total[0])]
        for input in inputs_total[1:]:
           logits.append(net(input))

        # put interleaved samples back
        logits = interleave(logits, args.batch_size)

        outputs = logits[0]
        u_chunk = (len(logits)-1)//2
        outputs_o = torch.cat(logits[1:u_chunk+1], dim=0)
        outputs_o2 = torch.cat(logits[u_chunk+1:], dim=0)

        with torch.no_grad():
            outputs_o.detach_()
            pred_o = torch.softmax(outputs_o, dim=1)
            pred_max = pred_o.max(1)
            mask = (pred_max[0] >= args.ths_pred)
            targets_pseudo = pred_max[1]

        if args.lmd_unif > 0:
            logits_s = net(inputs_s, aux=True)
            loss_unif = - torch.mean(torch.sum(F.log_softmax(logits_s, dim=1) * targets_s, dim=1)) * args.lmd_unif
            if args.rampup:
                loss_unif = loss_unif * np.clip(batch_idx/args.num_iters, 0.0, 1.0)
        else:
            loss_unif = 0

        loss = criterion(outputs, targets) +  args.lmd_u * torch.mean(criterion_none(outputs_o2, targets_pseudo.detach()) * mask) + loss_unif

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()
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
            ema_optimizer.step(bn=True)
            save = val(net, batch_idx, _labeled_testloader, median=median)
            if save:
               checkpoint(net, optimizer, best_val, batch_idx)
            save = val(net_ema, batch_idx, _labeled_testloader, ema=True, median=median)
            if save:
                checkpoint(net_ema, optimizer, best_val_ema, batch_idx, ema=True)
            net.train()
            net_ema.train()
        else:
            progress_bar(batch_idx % 1000, 1000, 'working...')

    checkpoint(net, optimizer, current_val, args.num_iters, last=True)
    checkpoint(net_ema, optimizer, current_val_ema, args.num_iters, ema=True, last=True)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

median_acc = []
median_acc_ema = []

def val(net, iters, testloader, ema=False, median=False):
    global best_val
    global best_val_ema
    global median_acc
    global median_acc_ema
    global current_val
    global current_val_ema
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

    if ema:
        if median:
            median_acc_ema.append(acc.item())
        if tensorboardX_compat:
            writer.add_scalar("validation/ema_loss", val_loss/(batch_idx+1), iters+1)
            writer.add_scalar("validation/ema_top1_acc", acc, iters+1)
        current_val_ema = acc
        if acc > best_val_ema:
            best_val_ema = acc
            return True
        else:
            return False
    else:
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

def checkpoint(net, optimizer, acc, iters, ema=False, last=False):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'acc': acc,
        'iters': iters,
        'rng_state': torch.get_rng_state()
    }
    if ema:
        torch.save(state, os.path.join(logdir, 'ema_ckpt.t7' if (not last) else 'last_ema_ckpt.t7'))
    else:
        torch.save(state, os.path.join(logdir, 'ckpt.t7' if (not last) else 'last_ckpt.t7'))

def adjust_learning_rate(optimizer, iters):
    """decrease the learning rate"""
    lr = args.lr * np.cos(iters/(args.num_iters+1) * (7 * np.pi) / (2 * 8))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999, wd=False):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.tmp_model = models.load_model(args.model, _labeled_num_class, divide=args.aux_divide)
        self.wd = 0.02 * args.lr if wd else 0

        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.data.copy_(param.data)

    def step(self, bn=False):
        if bn:
            # copy batchnorm stats to ema model
            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                tmp_param.data.copy_(ema_param.data.detach())

            self.ema_model.load_state_dict(self.model.state_dict())

            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                ema_param.data.copy_(tmp_param.data.detach())
        else:
            one_minus_alpha = 1.0 - self.alpha
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(self.alpha)
                ema_param.data.add_(param.data.detach() * one_minus_alpha)
                # customized weight decay
                param.data.mul_(1 - self.wd)

if args.ema:
    ema_train()

    print("Best Accuracy : {}".format(best_val))
    print("Best Accuracy EMA : {}".format(best_val_ema))
    print("Median Accuracy : {}".format(np.median(median_acc)))
    print("Median Accuracy EMA : {}".format(np.median(median_acc_ema)))
    logger = logging.getLogger('best')
    logger.info('[Acc {:.3f}] [EMA Acc {:.3f}] [MEDIAN Acc {:.3f}] [MEDIAN EMA Acc {:.3f}]'.format(best_val, best_val_ema, np.median(median_acc), np.median(median_acc_ema)))
else:
    raise NotImplementedError

if tensorboardX_compat:
    writer.close()

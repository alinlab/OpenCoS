import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data.distributed import DistributedSampler
from collections import defaultdict
import math 
import random
import numpy as np

# Setup Augmentations

def get_dataset(args):
 
    ### color augmentation ###    
    color_jitter = transforms.ColorJitter(0.8*args.color_jitter_strength, 0.8*args.color_jitter_strength, 0.8*args.color_jitter_strength, 0.2*args.color_jitter_strength)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
   
    if 'contrastive' in args.train_type:
        contrastive_learning = True
    else:
        contrastive_learning = False

    if 'linear_eval' in args.train_type or 'multinomial' in args.train_type:
        linear_eval = True
    else:
        linear_eval = False

    if contrastive_learning:
        transform_train = transforms.Compose([
            rnd_color_jitter,
            rnd_gray,
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(32),
            transforms.ToTensor(),
        ])

        transform_test = transform_train

    elif linear_eval:
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])

        transform_test = transform_train

    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])

    class TransformTwice:
        def __init__(self, transform):
            self.transform = transform

        def __call__(self, inp):
            out1 = self.transform(inp)
            out2 = self.transform(inp)
            return out1, out2

    if args.dataset == 'cifar-10':

        if contrastive_learning:
            train_dst   = datasets.CIFAR10(root=args.dataroot, train=True, download=True,transform=TransformTwice(transform_train))
        else:
            train_dst   = datasets.CIFAR10(root=args.dataroot, train=True, download=True,transform=(transform_train))
        val_dst     = datasets.CIFAR10(root=args.dataroot, train=False, download=True,transform=transform_test)

        if args.ooc_data == 'svhn':
            ooc_dst   = datasets.SVHN(root=args.dataroot, split='train', download=True, transform=TransformTwice(transform_train))
            ooc_index = np.load(os.path.join('../OpenCoS/splits', 'svhn_unlabel_train_idx.npy')).astype(np.int64)
        elif args.ooc_data == 'tiny':
            ooc_dst   = datasets.ImageFolder(root=args.tinyroot, transform=TransformTwice(transform_train))
            ooc_index = np.load(os.path.join('../OpenCoS/splits', 'tiny_unlabel_train_idx.npy')).astype(np.int64)

        if args.ooc_data in ['svhn', 'tiny']:
            cifar_label = np.load(os.path.join('../OpenCoS/splits', 'cifar10_400pc_label_idx.npy')).astype(np.int64)
            cifar_unlabel = np.load(os.path.join('../OpenCoS/splits', 'cifar10_unlabel_train_idx.npy')).astype(np.int64)
            cifar_unlabel = cifar_unlabel[:50000 - len(cifar_label) - 40000]

            cifar_index = np.concatenate((cifar_label, cifar_unlabel))
            train_dst = torch.utils.data.Subset(train_dst, cifar_index)

            ooc_index = ooc_index[:40000]
            ooc_dst = torch.utils.data.Subset(ooc_dst, ooc_index)

            train_dst = torch.utils.data.ConcatDataset([train_dst, ooc_dst])

    if args.dataset == 'cifar-100':

        if contrastive_learning:
            train_dst   = datasets.CIFAR100(root=args.dataroot, train=True, download=True,transform=TransformTwice(transform_train))
        else:
            train_dst   = datasets.CIFAR100(root=args.dataroot, train=True, download=True,transform=(transform_train))
        val_dst     = datasets.CIFAR100(root=args.dataroot, train=False, download=True,transform=transform_test)

        if args.ooc_data == 'svhn':
            ooc_dst   = datasets.SVHN(root=args.dataroot, split='train', download=True, transform=TransformTwice(transform_train))
            ooc_index = np.load(os.path.join('../OpenCoS/splits', 'svhn_unlabel_train_idx.npy')).astype(np.int64)
        elif args.ooc_data == 'tiny':
            ooc_dst   = datasets.ImageFolder(root=args.tinyroot, transform=TransformTwice(transform_train))
            ooc_index = np.load(os.path.join('../OpenCoS/splits', 'tiny_unlabel_train_idx.npy')).astype(np.int64)

        if args.ooc_data in ['svhn', 'tiny']:
            cifar_label = np.load(os.path.join('../OpenCoS/splits', 'cifar100_100pc_label_idx.npy')).astype(np.int64)
            cifar_unlabel = np.load(os.path.join('../OpenCoS/splits', 'cifar100_unlabel_train_idx.npy')).astype(np.int64)
            cifar_unlabel = cifar_unlabel[:50000 - len(cifar_label) - 40000]

            cifar_index = np.concatenate((cifar_label, cifar_unlabel))
            train_dst = torch.utils.data.Subset(train_dst, cifar_index)

            ooc_index = ooc_index[:40000]
            ooc_dst = torch.utils.data.Subset(ooc_dst, ooc_index)

            train_dst = torch.utils.data.ConcatDataset([train_dst, ooc_dst])

    if contrastive_learning:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dst,
            num_replicas=args.ngpu,
            rank=args.local_rank,
            )
        train_loader = torch.utils.data.DataLoader(train_dst,batch_size=args.batch_size,num_workers=4,
                pin_memory=True,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
            )

        val_loader = torch.utils.data.DataLoader(val_dst,batch_size=100,num_workers=4,
                pin_memory=True,
                shuffle=False,
            )
        
        return train_loader, train_dst, val_loader, val_dst, train_sampler
    else:
        train_loader = torch.utils.data.DataLoader(train_dst,
                                              batch_size=args.batch_size,
                                              shuffle=True, num_workers=4)

        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=100,
                                             shuffle=False, num_workers=4)

        return train_loader, train_dst, val_loader, val_dst

import csv, torchvision, numpy as np, random, os
from PIL import Image

from torch.utils.data import Sampler, Dataset, DataLoader, BatchSampler, SequentialSampler, RandomSampler, Subset, ConcatDataset
from torchvision import transforms, datasets
from collections import defaultdict
from randaugment import RandAugmentMC
import bisect
import warnings
import numpy as np
import torch




class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

class TransformDouble:
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform1(inp)
        out2 = self.transform2(inp)
        return out1, out2

class TransformList:
    def __init__(self, transform1, transform2, K):
        self.transform1 = transform1
        self.transform2 = transform2
        self.K = K

    def __call__(self, inp):
        return self.transform1(inp), [self.transform2(inp) for _ in range(self.K)]

class DatasetWrapper(Dataset):
    # Additinoal attributes
    # - indices
    # - classwise_indices
    # - num_classes
    # - get_class

    def __init__(self, dataset, indices=None):
        self.base_dataset = dataset
        if indices is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = indices

        # torchvision 0.2.0 compatibility
        if torchvision.__version__.startswith('0.2'):
            if isinstance(self.base_dataset, datasets.ImageFolder):
                self.base_dataset.targets = [s[1] for s in self.base_dataset.imgs]
            else:
                if self.base_dataset.train:
                    self.base_dataset.targets = self.base_dataset.train_labels
                else:
                    self.base_dataset.targets = self.base_dataset.test_labels

        self.classwise_indices = defaultdict(list)
        for i in range(len(self)):
            y = self.base_dataset.targets[int(self.indices[i])]
            self.classwise_indices[y].append(i)
        self.num_classes = len(self.classwise_indices.keys())

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)

    def get_class(self, i):
        return self.base_dataset.targets[self.indices[i]]

    def reset(self):
        self.__init__(self.base_dataset, self.indices)


def load_dataset(name, root, sample='default', **kwargs):

    if 'imagenet' in kwargs['uroot']:
        imagesize = 224
    else:
        imagesize = 32

    if imagesize==32:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(imagesize),
            transforms.ToTensor(),
        ])

        if kwargs['use_jitter']:
            ### color augmentation ###
            color_jitter_strength = 0.5
            color_jitter = transforms.ColorJitter(0.8*color_jitter_strength, 0.8*color_jitter_strength, 0.8*color_jitter_strength, 0.2*color_jitter_strength)
            rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
            rnd_gray = transforms.RandomGrayscale(p=0.2)
            transform_aug = transforms.Compose([
                rnd_color_jitter,
                rnd_gray,
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            ### RandAugment ###
            transform_aug = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                RandAugmentMC(n=2, m=10),
                transforms.ToTensor(),
            ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        if kwargs['use_jitter']:
            ### color augmentation ###
            color_jitter_strength = 1
            color_jitter = transforms.ColorJitter(0.8*color_jitter_strength, 0.8*color_jitter_strength, 0.8*color_jitter_strength, 0.2*color_jitter_strength)
            rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
            rnd_gray = transforms.RandomGrayscale(p=0.2)
            transform_aug = transforms.Compose([
                rnd_color_jitter,
                rnd_gray,
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            ### RandAugment ###
            transform_aug = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                RandAugmentMC(n=2, m=10),
                transforms.ToTensor(),
            ])

    if name == 'cifar10':
        Label = datasets.CIFAR10
        if int(kwargs['pc']) == 5000:
            trainidx = None
        else:
            trainidx = np.load(os.path.join('splits', name + '_' + kwargs['pc'] + 'pc_label_idx.npy')).astype(np.int64)
            unlabel_idx = [i for i in range(50000) if i not in trainidx]
        trainset = DatasetWrapper(Label(root, train=True,  download=True, transform=TransformDouble(transform_train, transform_aug)), trainidx)
        testset  = Label(root, train=False, download=True, transform=transform_test)

    elif name == 'cifar100':
        Label = datasets.CIFAR100
        if int(kwargs['pc']) == 500:
            trainidx = None
        else:
            trainidx = np.load(os.path.join('splits', name + '_' + kwargs['pc'] + 'pc_label_idx.npy')).astype(np.int64)
            unlabel_idx = [i for i in range(50000) if i not in trainidx]
        trainset = DatasetWrapper(Label(root, train=True,  download=True, transform=TransformDouble(transform_train, transform_aug)), trainidx)
        testset  = Label(root, train=False, download=True, transform=transform_test)

    elif name == 'animal':
        animal_class = [2,3,4,5,6,7]
        def target_transformA(target):
            return target - 2

        Label = datasets.CIFAR10
        trainidx = np.load(os.path.join('splits', 'cifar10_animal_' + kwargs['pc'] + 'pc_label_idx.npy')).astype(np.int64)
        testidx = np.load(os.path.join('splits', 'cifar10_animal_test_idx.npy')).astype(np.int64)
        trainset = DatasetWrapper(Label(root, train=True,  download=True, transform=TransformDouble(transform_train, transform_aug), target_transform=target_transformA), trainidx)
        testset  = DatasetWrapper(Label(root, train=False, download=True, transform=transform_test, target_transform=target_transformA),testidx)
        notanimal_idx = np.load(os.path.join('splits', 'cifar10_notanimal_unlabel_idx.npy')).astype(np.int64)
        unlabel_idx = [i for i in range(50000) if i not in trainidx and i not in notanimal_idx]

    elif name in ['dog_cls', 'bird_cls', 'primate_cls', 'insect_cls', 'reptile_cls', 'aquatic_animal_cls', 'food_cls', 'produce_cls', 'scenery_cls']:
        train_val_dataset_dir = os.path.join(kwargs['imgroot'], "train")
        test_dataset_dir = os.path.join(kwargs['imgroot'], "val")

        superclasses = np.load(os.path.join('splits_img', name + '_idx.npy')).astype(np.int64)
        def target_transformS(target):
            return int(np.where(superclasses == target)[0])

        trainidx = np.load(os.path.join('splits_img', name + '_'+ kwargs['pc'] +'pc_train.npy')).astype(np.int64)
        testidx  = np.load(os.path.join('splits_img', name + '_test.npy')).astype(np.int64)

        trainset = DatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=TransformDouble(transform_train, transform_aug), target_transform=target_transformS), trainidx)
        testset  = Subset(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test, target_transform=target_transformS), testidx)

    else:
        raise Exception('Unknown dataset: {}'.format(name))

    if kwargs['method']=='default':
        unlabel_transform = TransformDouble(transform_train, transform_aug)
    elif kwargs['method']=='mixmatch':
        unlabel_transform = TransformTwice(transform_train)
    elif kwargs['method']=='remixmatch':
        unlabel_transform = TransformList(transform_train, transform_aug, kwargs['naug'])
    else:
        raise Exception('Unknown methods: {}'.format(kwargs['method']))

    Unlabel = []
    if 'imagenet' in kwargs['uroot']:
        unlabel_dataset_dir = os.path.join(kwargs['imgroot'], "train")
        if name in ['dog_cls', 'bird_cls', 'primate_cls', 'insect_cls' , 'reptile_cls', 'aquatic_animal_cls', 'food_cls', 'produce_cls', 'scenery_cls']:
            unlabel_idx = np.arange(1281167)
            unlabel_idx = list(set(unlabel_idx) - set(trainidx))
            Unlabel.append(Subset(datasets.ImageFolder(root=unlabel_dataset_dir, transform=unlabel_transform), unlabel_idx))
        else:
            Unlabel.append(datasets.ImageFolder(root=unlabel_dataset_dir, transform=unlabel_transform))

    if 'tiny' in kwargs['uroot']:
        unlabel_dataset_dir = os.path.join(kwargs['tinyroot'], "train")
        if kwargs['ood_samples']>0:
            tiny_index = np.load(os.path.join('splits', 'tiny_unlabel_train_idx.npy')).astype(np.int64)
            tiny_index = tiny_index[:kwargs['ood_samples']]
            Unlabel.append(Subset(datasets.ImageFolder(root=unlabel_dataset_dir, transform=unlabel_transform), tiny_index))
            if name in ['cifar10']:
                cifar_unlabel = np.load(os.path.join('splits', 'cifar10_unlabel_train_idx.npy')).astype(np.int64)
                cifar_unlabel = cifar_unlabel[:50000 - len(trainidx) - kwargs['ood_samples']]
                Unlabel.append(Subset(datasets.CIFAR10(root, train=True,  download=True, transform=unlabel_transform), cifar_unlabel))
            elif name in ['cifar100']:
                cifar_unlabel = np.load(os.path.join('splits', 'cifar100_unlabel_train_idx.npy')).astype(np.int64)
                cifar_unlabel = cifar_unlabel[:50000 - len(trainidx) - kwargs['ood_samples']]
                Unlabel.append(Subset(datasets.CIFAR100(root, train=True,  download=True, transform=unlabel_transform), cifar_unlabel))
            else:
                raise Exception('Unknown labeled dataset: {}'.format(name))

    if 'cten' in kwargs['uroot']:
        if name in ['cifar10', 'animal']:
            if name == 'animal':
                Unlabel.append(Subset(datasets.CIFAR10(root, train=True,  download=True, transform=unlabel_transform), notanimal_idx))
            Unlabel.append(Subset(datasets.CIFAR10(root, train=True,  download=True, transform=unlabel_transform), unlabel_idx))
        else:
            Unlabel.append(datasets.CIFAR10(root, train=True,  download=True, transform=unlabel_transform))
    if 'chund' in kwargs['uroot']:
        if name in ['cifar100']:
            Unlabel.append(Subset(datasets.CIFAR100(root, train=True,  download=True, transform=unlabel_transform), unlabel_idx))
        else:
            Unlabel.append(datasets.CIFAR100(root, train=True,  download=True, transform=unlabel_transform))
    if 'svhn' in kwargs['uroot']:
        if kwargs['ood_samples']>0:
            svhn_index = np.load(os.path.join('splits', 'svhn_unlabel_train_idx.npy')).astype(np.int64)
            svhn_index = svhn_index[:kwargs['ood_samples']]
            Unlabel.append(Subset(datasets.SVHN(root, split='train',  download=True, transform=unlabel_transform), svhn_index))
            if name in ['cifar10']:
                cifar_unlabel = np.load(os.path.join('splits', 'cifar10_unlabel_train_idx.npy')).astype(np.int64)
                cifar_unlabel = cifar_unlabel[:50000 - len(trainidx) - kwargs['ood_samples']]
                Unlabel.append(Subset(datasets.CIFAR10(root, train=True,  download=True, transform=unlabel_transform), cifar_unlabel))
            elif name in ['cifar100']:
                cifar_unlabel = np.load(os.path.join('splits', 'cifar100_unlabel_train_idx.npy')).astype(np.int64)
                cifar_unlabel = cifar_unlabel[:50000 - len(trainidx) - kwargs['ood_samples']]
                Unlabel.append(Subset(datasets.CIFAR100(root, train=True,  download=True, transform=unlabel_transform), cifar_unlabel))
            else:
                assert(False)

    unlabeled_trainset = ConcatDataset(Unlabel)

    return trainset, unlabeled_trainset, testset


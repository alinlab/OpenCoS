import argparse

def parser():

    parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning of Visual Representation')
    parser.add_argument('--train_type', default='contrastive_learning', type=str, help='standard')
    parser.add_argument('--lr', default=1.5, type=float, help='learning rate, LearningRate = 0.3 × BatchSize/256 for ImageNet, 0.5,1.0,1.5 for CIFAR')
    parser.add_argument('--lr_multiplier', default=1.0, type=float, help='learning rate multiplier, 5,10,15 -> 0.5,1.0,1.5 for CIFAR')
    parser.add_argument('--dataset', default='cifar-10', type=str, help='cifar-10/cifar-100/lsun/imagenet-resize/svhn')
    parser.add_argument('--dataroot', default='/data', type=str, help='PATH TO dataset cifar-10, cifar-100, svhn')
    parser.add_argument('--tinyroot', default='/data/tinyimagenet/tiny-imagenet-200/train/', type=str, help='PATH TO tinyimagenet dataset')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--model', default="ResNet50", type=str,
                        help='model type (default: ResNet50)')
    parser.add_argument('--name', default='', type=str, help='name of run')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size / multi-gpu setting: batch per gpu')
    parser.add_argument('--epoch', default=1000, type=int, 
                        help='total epochs to run')
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                        help='use standard augmentation (default: True)')
    parser.add_argument('--decay', default=1e-6, type=float, help='weight decay')

    ##### arguments for data augmentation #####
    parser.add_argument('--color_jitter_strength', default=0.5, type=float, help='0.5 for CIFAR, 1.0 for ImageNet')
    parser.add_argument('--temperature', default=0.5, type=float, help='temperature for pairwise-similarity')
 
    ##### arguments for linear evaluation #####
    parser.add_argument('--multinomial_l2_regul', default=0.5, type=float, help='regularization for multinomial logistic regression')

    ##### arguments for distributted parallel ####
    parser.add_argument('--local_rank', type=int, default=0)   
    parser.add_argument('--ngpu', type=int, default=4)

    parser.add_argument('--ooc_data', type=str, default=None)   

    args = parser.parse_args()

    return args

def print_args(args):
    for k, v in vars(args).items():
        print('{:<16} : {}'.format(k, v))


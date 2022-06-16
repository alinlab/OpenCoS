import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from .aux_batchnorm import BatchNorm2d

__all__ = ['CIFAR_ResNet_AuxBN', 'CIFAR_ResNet18_AuxBN', 'CIFAR_ResNet34_AuxBN', 'CIFAR_ResNet10_AuxBN', 'CIFAR_ResNet50_AuxBN']

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(module, nn.Conv2d):
                if type(inputs) == tuple:
                    inputs = (module(inputs[0]),)+ inputs[1:]
                else:
                    inputs = module(inputs)
            else:
                if type(inputs) == tuple:
                    inputs = (module(*inputs),)+ inputs[1:]
                else:
                    inputs = module(inputs)

        if type(inputs) == tuple:
            return inputs[0]
        else:
            return inputs


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, divide=False):
        super(PreActBlock, self).__init__()
        self.bn1 = BatchNorm2d(in_planes, divide=divide)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = BatchNorm2d(planes, divide=divide)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x, aux=False):
        out = F.relu(self.bn1(x,aux))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out,aux)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, divide=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, divide=divide)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, divide=divide)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(self.expansion*planes, divide=divide)

        self.shortcut = mySequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = mySequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(self.expansion*planes, divide=divide)
            )

    def forward(self, x, aux=False):
        out = F.relu(self.bn1(self.conv1(x),aux))
        out = F.relu(self.bn2(self.conv2(out),aux))
        out = self.bn3(self.conv3(out),aux)
        out += self.shortcut(x,aux)
        out = F.relu(out)
        return out




class CIFAR_ResNet_AuxBN(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, bias=True, divide=False):
        super(CIFAR_ResNet_AuxBN, self).__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(3,64)
        self.bn1 = BatchNorm2d(64, divide=divide)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, divide=divide)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, divide=divide)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, divide=divide)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, divide=divide)
        self.linear = nn.Linear(512*block.expansion, num_classes, bias=bias)
        self.linear_rot = nn.Linear(512*block.expansion, 4, bias=bias)


    def _make_layer(self, block, planes, num_blocks, stride, divide):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, divide))
            self.in_planes = planes * block.expansion
        return mySequential(*layers)

    def forward(self, x, feature=False, aux=False):
        out = x
        out = self.conv1(out)
        out = self.bn1(out, aux)
        out = F.relu(out)
        out1 = self.layer1(out, aux)
        out2 = self.layer2(out1, aux)
        out3 = self.layer3(out2, aux)
        out = self.layer4(out3, aux)
        out = F.avg_pool2d(out, 4)
        out4 = out.view(out.size(0), -1)
        out = self.linear(out4)
        if feature:
            return out, out4
        else:
            return out

    def feature(self, x, aux=False):
        out = x
        out = self.conv1(out)
        out = self.bn1(out, aux)
        out = F.relu(out)
        out1 = self.layer1(out, aux)
        out2 = self.layer2(out1, aux)
        out3 = self.layer3(out2, aux)
        out = self.layer4(out3, aux)
        out = F.avg_pool2d(out, 4)
        out4 = out.view(out.size(0), -1)
        return out4

    def rot(self, x, aux=False):
        out = x
        out = self.conv1(out)
        out = self.bn1(out, aux)
        out = F.relu(out)
        out1 = self.layer1(out, aux)
        out2 = self.layer2(out1, aux)
        out3 = self.layer3(out2, aux)
        out = self.layer4(out3, aux)
        out = F.avg_pool2d(out, 4)
        out4 = out.view(out.size(0), -1)
        out_rot = self.linear_rot(out4)
        return out_rot


    def forward_rot(self, x, aux=False):
        out = x
        out = self.conv1(out)
        out = self.bn1(out, aux)
        out = F.relu(out)
        out1 = self.layer1(out, aux)
        out2 = self.layer2(out1, aux)
        out3 = self.layer3(out2, aux)
        out = self.layer4(out3, aux)
        out = F.avg_pool2d(out, 4)
        out4 = out.view(out.size(0), -1)
        out = self.linear(out4)
        out_rot = self.linear_rot(out4)
        return out, out_rot


def CIFAR_ResNet10_AuxBN(pretrained=False, **kwargs):
    return CIFAR_ResNet_AuxBN(PreActBlock, [1,1,1,1], **kwargs)

def CIFAR_ResNet18_AuxBN(pretrained=False, **kwargs):
    return CIFAR_ResNet_AuxBN(PreActBlock, [2,2,2,2], **kwargs)

def CIFAR_ResNet34_AuxBN(pretrained=False, **kwargs):
    return CIFAR_ResNet_AuxBN(PreActBlock, [3,4,6,3], **kwargs)

def CIFAR_ResNet50_AuxBN(pretrained=False, **kwargs):
    return CIFAR_ResNet_AuxBN(Bottleneck, [3,4,6,3], **kwargs)

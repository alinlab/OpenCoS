import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from .aux_batchnorm import BatchNorm2d
from .cifar_resnet_auxbn import mySequential
from .resnet import conv3x3, conv1x1

__all__ = ['resnet18_auxbn', 'resnet34_auxbn', 'resnet50_auxbn', 'resnet101_auxbn', 'resnet152_auxbn']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, aux=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out,aux)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out,aux)

        if self.downsample is not None:
            identity = self.downsample(x,aux)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, divide=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width, divide=divide)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width, divide=divide)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, divide=divide)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, aux=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out,aux)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out,aux)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out,aux)

        if self.downsample is not None:
            identity = self.downsample(x,aux)

        out += identity
        out = self.relu(out)

        return out

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = BatchNorm2d(planes)
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

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None, bias=True, divide=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2d

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes, divide=divide)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, divide=divide)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer, divide=divide)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer, divide=divide)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer, divide=divide)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes, bias=bias)
        self.linear_rot = nn.Linear(512*block.expansion, 4, bias=bias)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.bn.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.bn.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None, divide=False):
        if norm_layer is None:
            norm_layer = BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = mySequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, divide=divide),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer, divide=divide))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer, divide=divide))

        return mySequential(*layers)

    def forward(self, x, feature=False, aux=False):
        x = self.conv1(x)
        x = self.bn1(x,aux)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x,aux)
        x = self.layer2(x,aux)
        x = self.layer3(x,aux)
        x = self.layer4(x,aux)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        o = self.fc(x)

        if feature:
            return o, x
        else:
            return o

    def feature(self, x, aux=False):
        x = self.conv1(x)
        x = self.bn1(x,aux)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x,aux)
        x = self.layer2(x,aux)
        x = self.layer3(x,aux)
        x = self.layer4(x,aux)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


    def rot(self, x, aux=False):
        x = self.conv1(x)
        x = self.bn1(x,aux)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x,aux)
        x = self.layer2(x,aux)
        x = self.layer3(x,aux)
        x = self.layer4(x,aux)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        o_rot = self.linear_rot(x)

        return o_rot

    def forward_rot(self, x, aux=False):
        x = self.conv1(x)
        x = self.bn1(x,aux)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x,aux)
        x = self.layer2(x,aux)
        x = self.layer3(x,aux)
        x = self.layer4(x,aux)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        o = self.fc(x)
        o_rot = self.linear_rot(x)

        return o, o_rot


def resnet10_auxbn(pretrained=False, **kwargs):
    """Constructs a ResNet-10 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

def resnet18_auxbn(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34_auxbn(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50_auxbn(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101_auxbn(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152_auxbn(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

'''Wide-ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] Sergey Zagoruyko, Nikos Komodakis
    Wide Residual Networks. arXiv:1605.07146
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .aux_batchnorm import BatchNorm2d
from .resnet_auxbn import mySequential

__all__ = ['WideResNet_AuxBN', 'wide_resnet_auxbn']

class WideBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, divide=False):
        super(WideBasicBlock, self).__init__()
        self.bn1 = BatchNorm2d(in_planes, divide=divide)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, divide=divide)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x, aux=False):
        o1 = F.leaky_relu(self.bn1(x,aux), 0.1)
        y = self.conv1(o1)
        o2 = F.leaky_relu(self.bn2(y,aux), 0.1)
        z = self.conv2(o2)
        if len(self.shortcut)==0:
            return z + x
        else:
            return z + self.shortcut(o1)



class WideResNet_AuxBN(nn.Module):
    """ WRN28-width with leaky relu (negative slope is 0.1)"""
    def __init__(self, block, depth, width, num_classes, divide=False):
        super(WideResNet_AuxBN, self).__init__()
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        widths = [int(v * width) for v in (16, 32, 64)]

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, widths[0], n, stride=1, divide=divide)
        self.layer2 = self._make_layer(block, widths[1], n, stride=2, divide=divide)
        self.layer3 = self._make_layer(block, widths[2], n, stride=2, divide=divide)
        self.bn1 = BatchNorm2d(widths[2], divide=divide)
        self.linear = nn.Linear(widths[2]*block.expansion, num_classes)
        self.linear_rot = nn.Linear(widths[2]*block.expansion, 4)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, BatchNorm2d):
                nn.init.uniform_(m.bn.weight)
                nn.init.constant_(m.bn.bias, 0)
                nn.init.constant_(m.bn.running_mean, 0)
                nn.init.constant_(m.bn.running_var, 1)
                nn.init.constant_(m.bn_aux.running_mean, 0)
                nn.init.constant_(m.bn_aux.running_var, 1)

    def _make_layer(self, block, planes, num_blocks, stride, divide):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, divide))
            self.in_planes = planes * block.expansion
        return mySequential(*layers)

    def forward(self, x, feature=False, aux=False):
        f0 = self.conv1(x)
        f1 = self.layer1(f0,aux)
        f2 = self.layer2(f1,aux)
        f3 = self.layer3(f2,aux)
        out = F.leaky_relu(self.bn1(f3,aux), 0.1)
        out = F.avg_pool2d(out, 8)
        out4 = out.view(out.size(0), -1)
        out = self.linear(out4)

        if feature:
            return out, out4
        else:
            return out

    def feature(self, x, aux=False):
        f0 = self.conv1(x)
        f1 = self.layer1(f0,aux)
        f2 = self.layer2(f1,aux)
        f3 = self.layer3(f2,aux)
        out = F.leaky_relu(self.bn1(f3,aux), 0.1)
        out = F.avg_pool2d(out, 8)
        out4 = out.view(out.size(0), -1)
        return out4


    def rot(self, x, aux=False):
        f0 = self.conv1(x)
        f1 = self.layer1(f0,aux)
        f2 = self.layer2(f1,aux)
        f3 = self.layer3(f2,aux)
        out = F.leaky_relu(self.bn1(f3,aux), 0.1)
        out = F.avg_pool2d(out, 8)
        out4 = out.view(out.size(0), -1)
        out_rot = self.linear_rot(out4)

        return out_rot


    def forward_rot(self, x, aux=False):
        f0 = self.conv1(x)
        f1 = self.layer1(f0,aux)
        f2 = self.layer2(f1,aux)
        f3 = self.layer3(f2,aux)
        out = F.leaky_relu(self.bn1(f3,aux), 0.1)
        out = F.avg_pool2d(out, 8)
        out4 = out.view(out.size(0), -1)
        out = self.linear(out4)
        out_rot = self.linear_rot(out4)

        return out, out_rot


def wide_resnet_auxbn(depth, width, num_classes=10, divide=False):
    return WideResNet_AuxBN(WideBasicBlock, depth, width, num_classes, divide=divide)


#if __name__ == "__main__":
#    net = wide_resnet(28, 2)
#    net.cuda()
#    x = torch.randn(2,3,32,32).cuda()
#    print (net)
#    print (net(x).size())

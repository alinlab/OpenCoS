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


__all__ = ['WideResNet', 'wide_resnet']


class WideBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(WideBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        o1 = F.leaky_relu(self.bn1(x), 0.1)
        y = self.conv1(o1)
        o2 = F.leaky_relu(self.bn2(y), 0.1)
        z = self.conv2(o2)
        if len(self.shortcut)==0:
            return z + x
        else:
            return z + self.shortcut(o1)



class WideResNet(nn.Module):
    """ WRN28-width with leaky relu (negative slope is 0.1)"""
    def __init__(self, block, depth, width, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        widths = [int(v * width) for v in (16, 32, 64)]

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, widths[0], n, stride=1)
        self.layer2 = self._make_layer(block, widths[1], n, stride=2)
        self.layer3 = self._make_layer(block, widths[2], n, stride=2)
        self.bn1 = nn.BatchNorm2d(widths[2])
        self.linear = nn.Linear(widths[2]*block.expansion, num_classes)
        self.linear_rot = nn.Linear(widths[2]*block.expansion, 4)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_mean, 0)
                nn.init.constant_(m.running_var, 1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, feature=False, blocks=False, aux=False):
        f0 = self.conv1(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        out = F.leaky_relu(self.bn1(f3), 0.1)
        out = F.avg_pool2d(out, 8)
        out4 = out.view(out.size(0), -1)
        out = self.linear(out4)
        if blocks:
            return out, [f1,f2,f3]
        elif feature:
            return out, out4
        else:
            return out

    def feature(self, x, aux=False):
        f0 = self.conv1(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        out = F.leaky_relu(self.bn1(f3), 0.1)
        out = F.avg_pool2d(out, 8)
        out4 = out.view(out.size(0), -1)
        return out4

    def blocks(self, x, aux=False):
        f0 = self.conv1(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return [f1,f2,f3]

    def rot(self, x, aux=False):
        f0 = self.conv1(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        out = F.leaky_relu(self.bn1(f3), 0.1)
        out = F.avg_pool2d(out, 8)
        out4 = out.view(out.size(0), -1)
        out_rot = self.linear_rot(out4)
        return out_rot

    def forward_rot(self, x, aux=False):
        f0 = self.conv1(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        out = F.leaky_relu(self.bn1(f3), 0.1)
        out = F.avg_pool2d(out, 8)
        out4 = out.view(out.size(0), -1)
        out = self.linear(out4)
        out_rot = self.linear_rot(out4)
        return out, out_rot


def wide_resnet(depth, width, num_classes=10):
    return WideResNet(WideBasicBlock, depth, width, num_classes)


#if __name__ == "__main__":
#    net = wide_resnet(28, 2)
#    net.cuda()
#    x = torch.randn(2,3,32,32).cuda()
#    print (net)
#    print (net(x).size())

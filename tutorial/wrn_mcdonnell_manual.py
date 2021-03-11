"""source https://raw.githubusercontent.com/szagoruyko/binary-wide-resnet/master/wrn_mcdonnell.py"""
from collections import OrderedDict
import math, re, torch
import torch.nn.functional as F
from torch import nn
from Dropit import Dropit

# import settings


def init_weight(*args):
    return nn.Parameter(
        nn.init.kaiming_normal_(torch.zeros(*args), mode="fan_out", nonlinearity="relu")
    )


class ForwardSign(torch.autograd.Function):
    """Fake sign op for 1-bit weights.

    See eq. (1) in https://arxiv.org/abs/1802.08530

    Does He-init like forward, and nothing on backward.
    """

    @staticmethod
    def forward(ctx, x):
        return math.sqrt(2.0 / (x.shape[1] * x.shape[2] * x.shape[3])) * x.sign()

    @staticmethod
    def backward(ctx, g):
        return g


class RoundedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad):
        return grad.clone()


rounded = RoundedFunction.apply


class Block(nn.Module):
    """Pre-activated ResNet block."""

    def __init__(self, width, dropit):
        super().__init__()
        self.dropit = dropit
        self.bn0 = nn.BatchNorm2d(width, affine=False)
        self.conv0 = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False)
        # self.dropit0 = Dropit(precision=actprec)
        assert init_weight(width, width, 3, 3).size() == self.conv0.weight.size()
        self.bn1 = nn.BatchNorm2d(width, affine=False)
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False)
        # self.dropit1 = Dropit(precision=actprec)
        # self.register_parameter('conv1', init_weight(width, width, 3, 3))

    def forward(self, x):
        if self.dropit is not None:
            h = F.relu6(self.bn0(x))
            h = self.conv0(torch.utils.checkpoint.checkpoint(self.dropit, h))
            h = F.relu6(self.bn1(h))
            h = self.conv1(torch.utils.checkpoint.checkpoint(self.dropit, h))
        else:
            h = self.conv0(rounded(F.relu6(self.bn0(x))))
            h = self.conv1(rounded(F.relu6(self.bn1(h))))
        return x + h


class DownsampleBlock(nn.Module):
    """Downsample block.

    Does F.avg_pool2d + torch.cat instead of strided conv.
    """

    def __init__(self, width, dropit):
        super().__init__()
        self.dropit = dropit
        self.bn0 = nn.BatchNorm2d(width // 2, affine=False)
        self.conv0 = nn.Conv2d(
            width // 2, width, kernel_size=3, padding=1, bias=False, stride=2
        )
        # self.dropit0 = Dropit(precision=actprec)
        assert init_weight(width, width // 2, 3, 3).size() == self.conv0.weight.size()

        self.bn1 = nn.BatchNorm2d(width, affine=False)
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False)
        # self.dropit1 = Dropit(precision=actprec)
        # self.register_parameter('conv1', init_weight(width, width, 3, 3))

    def forward(self, x):
        if self.dropit is not None:
            h = F.relu6(self.bn0(x))
            h = self.conv0(torch.utils.checkpoint.checkpoint(self.dropit, h))
            h = F.relu6(self.bn1(h))
            h = self.conv1(torch.utils.checkpoint.checkpoint(self.dropit, h))
        else:
            h = self.conv0(rounded(F.relu6(self.bn0(x))))
            h = self.conv1(rounded(F.relu6(self.bn1(h))))
        x_d = F.avg_pool2d(x, kernel_size=3, padding=1, stride=2)
        x_d = torch.cat([x_d, torch.zeros_like(x_d)], dim=1)
        return x_d + h


class WRN_McDonnell(nn.Module):
    """Implementation of modified Wide Residual Network.

    Differences with pre-activated ResNet and Wide ResNet:
        * BatchNorm has no affine weight and bias parameters
        * First layer has 16 * width channels
        * Last fc layer is removed in favor of 1x1 conv + F.avg_pool2d
        * Downsample is done by F.avg_pool2d + torch.cat instead of strided conv

    First and last convolutional layers are kept in float32.
    """

    def __init__(self, depth, num_classes, width, actprec=3, dropit=True):
        super().__init__()
        widths = [int(v * width) for v in (16, 32, 64)]
        n = (depth - 2) // 6
        self.dropit = Dropit(precision=actprec) if dropit else None

        self.conv0 = nn.Conv2d(3, widths[0], kernel_size=3, padding=1, bias=False)
        assert init_weight(widths[0], 3, 3, 3).size() == self.conv0.weight.size()

        self.group0 = self._make_block(widths[0], n)
        self.group1 = self._make_block(widths[1], n, downsample=True)
        self.group2 = self._make_block(widths[2], n, downsample=True)

        self.bn = nn.BatchNorm2d(widths[2], affine=False)
        # self.register_parameter('conv_last', init_weight(num_classes, widths[2], 1, 1))

        self.conv_last = nn.Conv2d(widths[2], num_classes, kernel_size=1, bias=False)
        self.bn_last = nn.BatchNorm2d(num_classes)

    def _make_block(self, width, n, downsample=False):
        def select_block(j):
            if downsample and j == 0:
                return DownsampleBlock(width, dropit=self.dropit)
            return Block(width, dropit=self.dropit)

        return nn.Sequential(
            OrderedDict(("block%d" % i, select_block(i)) for i in range(n))
        )

    def forward(self, x):
        h = self.conv0(x)
        h = self.group0(h)
        h = self.group1(h)
        h = self.group2(h)
        if self.dropit is not None:
            h = self.dropit(F.relu6(self.bn(h)))
        else:
            h = rounded(F.relu6(self.bn(h)))
        h = self.conv_last(h)
        h = self.bn_last(h)
        return F.avg_pool2d(h, kernel_size=h.shape[-2:]).view(h.shape[0], -1)


if __name__ == "__main__":
    model = WRN_McDonnell(28, 10, 10, True).to(torch.float64)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # model = Block(3, True).to(torch.float64)
    torch.autograd.gradcheck(
        model, torch.rand(1, 3, 28, 28, dtype=torch.float64, requires_grad=True)
    )
    print("Autograd went well, ok boomer")

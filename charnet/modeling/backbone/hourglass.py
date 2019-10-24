# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

from torch import nn
import torch.nn.functional as F


_norm_func = lambda num_features: nn.BatchNorm2d(num_features, eps=1e-5)


def _make_layer(in_channels, out_channels, num_blocks):
    """
    Creates a sequence of Residual blocks.

    :param in_channels: input channels
    :param out_channels: output channels
    :param num_blocks: number of blocks to create
    :param kwargs: passed to the Residual
    :return: sequence of residual blocks
    """
    blocks = []
    blocks.append(Residual(in_channels, out_channels))
    for _ in range(1, num_blocks):
        blocks.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blocks)


def _make_layer_revr(in_channels, out_channels, num_blocks):
    """
    Creates a sequence of Residual modules in reverse.

    :param in_channels:
    :param out_channels:
    :param num_blocks:
    :param kwargs:
    :return:
    """
    blocks = []
    for _ in range(num_blocks - 1):
        blocks.append(Residual(in_channels, in_channels))
    blocks.append(Residual(in_channels, out_channels))
    return nn.Sequential(*blocks)


class Residual(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        """
        Creates a Residual Module. It contains two convolutions, with a residual connection.
        The residual connection applies convolutions if the size of the output differs from the size
        of the input (or if the image is down samples by stride).

        :param in_channels: input channels
        :param out_channels: output channels
        :param stride: stride
        """
        super(Residual, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            _norm_func(out_channels),
            nn.ReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            _norm_func(out_channels)
        )
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
                _norm_func(out_channels)
            )
        else:
            self.skip = None
        self.out_relu = nn.ReLU()

    def forward(self, x):
        b1 = self.conv_2(self.conv_1(x))
        if self.skip is None:
            return self.out_relu(b1 + x)
        else:
            return self.out_relu(b1 + self.skip(x))


class HourGlassBlock(nn.Module):
    def __init__(self, n, channels, blocks):
        """
        Creates a HourGlassBlock.

        :param n: number of downsampling steps.
        :param channels: list of size n containing the number of channels each block creates
        :param blocks:
        """
        super(HourGlassBlock, self).__init__()

        self.up_1 = _make_layer(channels[0], channels[0], blocks[0])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.low_1 = _make_layer(channels[0], channels[1], blocks[0])
        if n <= 1:
            self.low_2 = _make_layer(channels[1], channels[1], blocks[1])
        else:
            self.low_2 = HourGlassBlock(n - 1, channels[1:], blocks[1:])
        self.low_3 = _make_layer_revr(channels[1], channels[0], blocks[0])
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        up_1 = self.up_1(x)
        low = self.low_3(self.low_2(self.low_1(self.pool(x))))
        return self.upsample(low) + up_1


class HourGlassNet(nn.Module):
    def __init__(self, n, channels, blocks):
        super(HourGlassNet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3, bias=False),
            _norm_func(128),
            nn.ReLU(),
            Residual(128, 256, stride=2)
        )
        hourglass_blocks = []
        for _ in range(2):
            hourglass_blocks.append(
                HourGlassBlock(n, channels, blocks)
            )
        self.hourglass_blocks = nn.Sequential(*hourglass_blocks)

    def forward(self, x):
        return self.hourglass_blocks(self.pre(x))


def hourglass88():
    return HourGlassNet(3, [256, 256, 256, 512], [2, 2, 2, 2])

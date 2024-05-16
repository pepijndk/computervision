# import torch
# import torch.nn.functional as F
import tensorlayerx as tlx
import tensorlayerx.nn as nn
from base import BaseModel

W_init = tlx.initializers.TruncatedNormal(stddev=0.02)
G_init = tlx.initializers.TruncatedNormal(mean=1.0, stddev=0.02)


class ResidualBlock(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(out_channels=64, kernel_size=(3, 3), stride=(1, 1), act=None, padding='SAME',
                               W_init=W_init, data_format='channels_first', b_init=None)
        self.bn1 = nn.BatchNorm2d(num_features=64, act=tlx.ReLU, gamma_init=G_init, data_format='channels_first')
        self.conv2 = nn.Conv2d(out_channels=64, kernel_size=(3, 3), stride=(1, 1), act=None, padding='SAME',
                               W_init=W_init, data_format='channels_first', b_init=None)
        self.bn2 = nn.BatchNorm2d(num_features=64, act=None, gamma_init=G_init, data_format='channels_first')

    def forward(self, x):
        input_x = x.copy()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = input_x + x
        return x


class Generator(BaseModel):
    def __init__(self, number_of_blocks=16):
        super().__init__()
        self.conv1 = nn.Conv2d(
            out_channels=64, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME', W_init=W_init,
            data_format='channels_first')
        self.residual_block = self.make_blocks(number_of_blocks)
        self.conv2 = nn.Conv2d(
            out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='SAME', W_init=W_init,
            data_format='channels_first', b_init=None)
        self.bn1 = nn.BatchNorm2d(num_features=64, act=None, gamma_init=G_init, data_format='channels_first')
        self.conv3 = nn.Conv2d(out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding='SAME', W_init=W_init,
                               data_format='channels_first')
        self.subpixelconv1 = nn.SubpixelConv2d(data_format='channels_first', scale=2, act=tlx.ReLU)
        self.conv4 = nn.Conv2d(out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding='SAME', W_init=W_init,
                               data_format='channels_first')
        self.subpixelconv2 = nn.SubpixelConv2d(data_format='channels_first', scale=2, act=tlx.ReLU)
        self.conv5 = nn.Conv2d(3, kernel_size=(1, 1), stride=(1, 1), act=tlx.Tanh, padding='SAME', W_init=W_init,
                               data_format='channels_first')

    def forward(self, x):
        x = self.conv1(x)
        temp = x
        x = self.residual_block(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = x + temp
        x = self.conv3(x)
        x = self.subpixelconv1(x)
        x = self.conv4(x)
        x = self.subpixelconv2(x)
        x = self.conv5(x)
        return x


def make_blocks(n):
    blocks = [ResidualBlock() for _ in n]
    blocks = nn.Sequential(blocks)
    return blocks

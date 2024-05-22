# import torch
# import torch.nn.functional as F
import numpy as np
import tensorlayerx as tlx
import tensorlayerx.nn as nn
import torch

from base import BaseModel

W_init = tlx.initializers.TruncatedNormal(stddev=0.02)
G_init = tlx.initializers.TruncatedNormal(mean=1.0, stddev=0.02)


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
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
        self.residual_block = make_blocks(number_of_blocks)
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


class Discriminator(BaseModel):
    def __init__(self, dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(
            out_channels=dim, kernel_size=(4, 4), stride=(2, 2), act=tlx.LeakyReLU, padding='SAME', W_init=W_init,
            data_format='channels_first')
        self.conv2 = nn.Conv2d(
            out_channels=dim * 2, kernel_size=(4, 4), stride=(2, 2), act=None, padding='SAME', W_init=W_init,
            data_format='channels_first', b_init=None)
        self.bn1 = nn.BatchNorm2d(num_features=dim * 2, act=tlx.LeakyReLU, gamma_init=G_init,
                                  data_format='channels_first')
        self.conv3 = nn.Conv2d(
            out_channels=dim * 4, kernel_size=(4, 4), stride=(2, 2), act=None, padding='SAME', W_init=W_init,
            data_format='channels_first', b_init=None)
        self.bn2 = nn.BatchNorm2d(num_features=dim * 4, act=tlx.LeakyReLU, gamma_init=G_init,
                                  data_format='channels_first')
        self.conv4 = nn.Conv2d(
            out_channels=dim * 8, kernel_size=(4, 4), stride=(2, 2), act=None, padding='SAME', W_init=W_init,
            data_format='channels_first', b_init=None)
        self.bn3 = nn.BatchNorm2d(num_features=dim * 8, act=tlx.LeakyReLU, gamma_init=G_init,
                                  data_format='channels_first')
        self.conv5 = nn.Conv2d(
            out_channels=dim * 16, kernel_size=(4, 4), stride=(2, 2), act=None, padding='SAME', W_init=W_init,
            data_format='channels_first', b_init=None)
        self.bn4 = nn.BatchNorm2d(num_features=dim * 16, act=tlx.LeakyReLU, gamma_init=G_init,
                                  data_format='channels_first')
        self.conv6 = nn.Conv2d(
            out_channels=dim * 32, kernel_size=(4, 4), stride=(2, 2), act=None, padding='SAME', W_init=W_init,
            data_format='channels_first', b_init=None)
        self.bn5 = nn.BatchNorm2d(num_features=dim * 32, act=tlx.LeakyReLU, gamma_init=G_init,
                                  data_format='channels_first')
        self.conv7 = nn.Conv2d(
            out_channels=dim * 16, kernel_size=(1, 1), stride=(1, 1), act=None, padding='SAME', W_init=W_init,
            data_format='channels_first', b_init=None)
        self.bn6 = nn.BatchNorm2d(num_features=dim * 16, act=tlx.LeakyReLU, gamma_init=G_init,
                                  data_format='channels_first')
        self.conv8 = nn.Conv2d(
            out_channels=dim * 8, kernel_size=(1, 1), stride=(1, 1), act=None, padding='SAME', W_init=W_init,
            data_format='channels_first', b_init=None)
        self.bn7 = nn.BatchNorm2d(num_features=dim * 8, act=None, gamma_init=G_init, data_format='channels_first')
        self.conv9 = nn.Conv2d(
            out_channels=dim * 2, kernel_size=(1, 1), stride=(1, 1), act=None, padding='SAME', W_init=W_init,
            data_format='channels_first', b_init=None)
        self.bn8 = nn.BatchNorm2d(num_features=dim * 2, act=tlx.LeakyReLU, gamma_init=G_init,
                                  data_format='channels_first')
        self.conv10 = nn.Conv2d(
            out_channels=dim * 2, kernel_size=(3, 3), stride=(1, 1), act=None, padding='SAME', W_init=W_init,
            data_format='channels_first', b_init=None)
        self.bn9 = nn.BatchNorm2d(num_features=dim * 2, act=tlx.LeakyReLU, gamma_init=G_init,
                                  data_format='channels_first')
        self.conv11 = nn.Conv2d(
            out_channels=dim * 8, kernel_size=(3, 3), stride=(1, 1), act=None, padding='SAME', W_init=W_init,
            data_format='channels_first', b_init=None)
        self.bn10 = nn.BatchNorm2d(num_features=dim * 8, gamma_init=G_init, data_format='channels_first')
        self.add = nn.Elementwise(combine_fn=tlx.add, act=tlx.LeakyReLU)
        self.flat = nn.Flatten()
        self.dense = nn.Linear(out_features=1, W_init=W_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.conv5(x)
        x = self.bn4(x)
        x = self.conv6(x)
        x = self.bn5(x)
        x = self.conv7(x)
        x = self.bn6(x)
        x = self.conv8(x)
        x = self.bn7(x)
        temp = x
        x = self.conv9(x)
        x = self.bn8(x)
        x = self.conv10(x)
        x = self.bn9(x)
        x = self.conv11(x)
        x = self.bn10(x)
        x = self.add([temp, x])
        x = self.flat(x)
        x = self.dense(x)

        return x


def make_blocks(n):
    blocks = [ResidualBlock() for _ in range(n)]
    blocks = nn.Sequential(blocks)
    return blocks


if __name__ == "__main__":
    gen = Generator()
    dummy_input = tlx.convert_to_tensor(np.ones((3, 300, 300)), dtype=tlx.float32)
    dummy_output = gen(dummy_input)
    print(dummy_output)

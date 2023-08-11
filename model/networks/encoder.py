import re
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
# from .external_function import SpectralNorm
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
import math

class AddCoords(nn.Module):
    """
    Add Coords to a tensor
    """
    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()
        self.with_r = with_r

    def forward(self, x):
        """
        :param x: shape (batch, channel, x_dim, y_dim)
        :return: shape (batch, channel+2, x_dim, y_dim)
        """
        B, _, x_dim, y_dim = x.size()

        # coord calculate
        xx_channel = torch.arange(x_dim).repeat(B, 1, y_dim, 1).permute(0, 1, 3, 2).type_as(x)
        yy_cahnnel = torch.arange(y_dim).repeat(B, 1, x_dim, 1).type_as(x)
        # normalization
        xx_channel = xx_channel.float() / (x_dim-1)
        yy_cahnnel = yy_cahnnel.float() / (y_dim-1)
        xx_channel = xx_channel * 2 - 1
        yy_cahnnel = yy_cahnnel * 2 - 1
        ret = torch.cat([x, xx_channel, yy_cahnnel], dim=1)

        if self.with_r:
            rr = torch.sqrt(xx_channel ** 2 + yy_cahnnel ** 2)
            ret = torch.cat([ret, rr], dim=1)

        return ret

class CoordConv(nn.Module):
    """
    CoordConv operation
    """
    def __init__(self, input_nc, output_dim, kernel_size, stride, use_bias, with_r=False, groups=1, padding=0):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(with_r=with_r)
        input_nc = input_nc + 2
        if with_r:
            input_nc = input_nc + 1
        self.conv = nn.Conv2d(input_nc, output_dim, kernel_size, stride, bias=use_bias, groups=groups, padding=padding)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=None, norm='none', activation='lrelu', pad_type='zero', coord=False, group=1, transpose=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        if not padding:
            padding = 0

        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transpose:
            # self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias, groups=group)
            self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride=stride, groups=group)
        else:
            if norm == 'sn':
                if coord:
                    self.conv = SpectralNorm(CoordConv(input_dim, output_dim, kernel_size, stride, use_bias=self.use_bias, groups=group))
                else:
                    self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias, groups=group))
            else:
                if coord:
                    self.conv = CoordConv(input_dim, output_dim, kernel_size, stride, use_bias=self.use_bias, groups=group)
                else:
                    self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias, groups=group)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class StyleCNNEncoder(nn.Module):
    def __init__(self, input_dim, dim, style_dim, layers, norm, activ, pad_type):
        super(StyleCNNEncoder, self).__init__()

        self.num_dowm = layers
        self.head_0 = Conv2dBlock(input_dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)

        down_layers = []
        for i in range(self.num_dowm):
            r = min(2 ** i, 8)
            indim = dim * r
            outdim = min(indim * 2, style_dim)
            down_layers.append(
                Conv2dBlock(indim, outdim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
            )

        self.down_layers = nn.ModuleList(down_layers)

    def forward(self, x):
        feature_list = []
        x = self.head_0(x)
        feature_list.append(x)
        for dl in self.down_layers:
            x = dl(x)
            feature_list.append(x)
        feature_list = list(reversed(feature_list))
        return feature_list


class VerticeNet(nn.Module):
    def __init__(self, input_dim, norm='none', activ='lrelu', pad_type='zero'):
        super(VerticeNet, self).__init__()

        # self.conv1 = Conv2dBlock(input_dim, 64, (4, 1), (2, 1), padding=(1, 1), norm=norm, activation=activ, pad_type=pad_type)
        # self.conv2 = Conv2dBlock(64, 128, (3, 1), (2, 1), padding=(1, 0), norm=norm, activation=activ, pad_type=pad_type)
        # self.conv3 = Conv2dBlock(128, 256, (3, 1), (2, 1), padding=(1, 0), norm=norm, activation=activ, pad_type=pad_type)
        #
        # self.upconv1 = Conv2dBlock(256, 128, (3, 1), (2, 1), padding=(1, 0), norm=norm, activation=activ, pad_type=pad_type, transpose=True)
        # self.upconv2 = Conv2dBlock(128, 64, (3, 1), (2, 1), padding=(1, 0), norm=norm, activation=activ, pad_type=pad_type, transpose=True)
        # self.upconv3 = Conv2dBlock(64, input_dim, (4, 1), (2, 1), padding=(1, 0), norm=norm, activation=activ, pad_type=pad_type, transpose=True)

        self.conv1 = nn.Conv2d(6, 64, (4,1), padding=(1,0), stride=(2,1))
        self.conv2 = nn.Conv2d(64, 128, (3,1), padding=(1,0), stride=(2,1))
        self.conv3 = nn.Conv2d(128, 256, (3,1), padding=(1,0), stride=(2,1))
        self.upconv1 = nn.ConvTranspose2d(256, 128, (3,1), padding=(1,0), stride=(2,1))
        self.upconv2 = nn.ConvTranspose2d(128, 64, (3,1), padding=(1,0), stride=(2,1))
        self.upconv3 = nn.ConvTranspose2d(64, 6, (4,1), padding=(1,0), stride=(2,1))

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.upbn1 = nn.BatchNorm2d(128)
        self.upbn2 = nn.BatchNorm2d(64)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.relu(self.upbn1(self.upconv1(x)), inplace=True)
        x = F.relu(self.upbn2(self.upconv2(x)), inplace=True)
        x = self.upconv3(x)
        return x


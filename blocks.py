import torch
import torch.nn as nn

# Implementation of SRGAN as described in https://arxiv.org/pdf/1609.04802.pdf
# The following classes define the building blocks for the generator and discriminator networks


class ConvBlock(nn.Module):
    # Convolutional layer with optional batch normalization and/or activation function
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size // 2),
            bias=(not batch_norm)
        ))

        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        if activation is not None:
            if activation == 'leakyrelu':
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif activation == 'prelu':
                layers.append(nn.PReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise RuntimeError(
                    'Unsupported activation function:', activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    # Two convolutional layers with a skip connection
    def __init__(self, num_channels, kernel_size, stride):
        super().__init__()
        self.conv_block1 = ConvBlock(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=kernel_size,
            stride=stride,
            batch_norm=True,
            activation='prelu'
        )
        self.conv_block2 = ConvBlock(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=kernel_size,
            stride=stride,
            batch_norm=True,
            activation=None
        )

    def forward(self, x):
        y = self.conv_block1(x)
        y = self.conv_block2(y)
        return x + y # Skip connection


class UpsampleBlock(nn.Module):
    # Uses convolution reshaping to perform upsampling
    def __init__(self, in_channels, kernel_size, upscale_factor):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=(in_channels * upscale_factor ** 2),
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size // 2)
        )
        self.ps = nn.PixelShuffle(upscale_factor=upscale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.ps(x)
        x = self.prelu(x)
        return x

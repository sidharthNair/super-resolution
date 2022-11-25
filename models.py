from math import log2
import torch.nn as nn

from blocks import ConvBlock, ResidualBlock, UpsampleBlock

# Implementation of SRGAN as described in https://arxiv.org/pdf/1609.04802.pdf


class Generator(nn.Module):
    def __init__(self, in_channels=3, num_residual_blocks=16, scaling_factor=4):
        super().__init__()

        # Convolution (k9n64s1) followed by PReLU
        self.conv_block1 = ConvBlock(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=9,
            stride=1,
            batch_norm=False,
            activation='prelu'
        )

        # B residual blocks (k3n64s1)
        residual_block_list = [ResidualBlock(num_channels=64, kernel_size=3, stride=1)
                               for i in range(num_residual_blocks)]
        self.residual_blocks = nn.Sequential(*residual_block_list)

        # Convolution (k3n64s1) followed by batch normalization
        self.conv_block2 = ConvBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            batch_norm=True,
            activation=None
        )

        # Upsampling layers (k3n256s1)
        upsample_block_list = [UpsampleBlock(in_channels=64, kernel_size=3, upscale_factor=2)
                               for i in range(int(log2(scaling_factor)))]
        self.upsample_blocks = nn.Sequential(*upsample_block_list)

        # Final convolution (k9n3s1)
        self.conv_block3 = ConvBlock(
            in_channels=64,
            out_channels=in_channels,
            kernel_size=9,
            stride=1,
            batch_norm=False,
            activation='tanh'
        )

    def forward(self, input):
        initial = self.conv_block1(input)
        output = self.residual_blocks(initial)
        output = self.conv_block2(output)
        output = initial + output  # Skip connection
        output = self.upsample_blocks(output)
        output = self.conv_block3(output)
        return output


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        # 8 convolutional blocks
        conv_block_list = []

        # Number of output channels for each convolutional block
        out_channels = [64, 64, 128, 128, 256, 256, 512, 512]

        for i in range(len(out_channels)):
            # First block does not have batch normalization
            # Stride alternates between 1 and 2 for each block
            conv_block_list.append(ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels[i],
                kernel_size=3,
                stride=(1 if i % 2 == 0 else 2),
                batch_norm=(i != 0),
                activation='leakyrelu'
            ))
            in_channels = out_channels[i]
        self.conv_blocks = nn.Sequential(*conv_block_list)

        # Classifies whether the input is LR or HR
        # Two dense layers with a final sigmoid activation for classification
        # Sigmoid is not present here because we will use BCEWithLogitsLoss during training
        self.classifier_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(6, 6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.classifier_block(x)
        return x

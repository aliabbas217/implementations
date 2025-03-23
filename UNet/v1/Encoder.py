import torch
import torch.nn as nn
import torch.nn.functional as F
from ConvBlock import ConvBlock

class Encoder(nn.Module):
  def __init__(
    self, in_channels = 3, levels=4, init_filters = 32,
    layers_per_block = 4, max_pool2d_kernel_size = 2, conv_block_kernel_size = 2
  ):
    super(Encoder, self).__init__()
    self.levels = levels
    self.init_filters = init_filters
    self.layers_per_block = layers_per_block
    self.in_channels = in_channels
    self.max_pool2d_kernel_size = max_pool2d_kernel_size
    self.conv_block_kernel_size = conv_block_kernel_size
    self.first_conv_block = ConvBlock(
        in_channels=self.in_channels, filters=self.init_filters,
        layers_per_block=self.layers_per_block,
        kernel_size=self.conv_block_kernel_size
    )
    self.max_pool2d = nn.MaxPool2d(kernel_size= max_pool2d_kernel_size)
    self.seq = nn.ModuleList([
        ConvBlock(
            in_channels= self.init_filters * (2**i), 
            filters = self.init_filters * (2**(i+1)),
            layers_per_block= self.layers_per_block,
            kernel_size= self.conv_block_kernel_size
        )
        for i in range(self.levels - 1)
    ])

  def forward(self, x):
    x = self.first_conv_block(x)
    x = nn.ParameterList([x])
    for conv_block in self.seq:
      y = self.max_pool2d(x[-1])
      x.append(conv_block(y))
    return x

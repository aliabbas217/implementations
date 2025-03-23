import torch
import torch.nn as nn
import torch.nn.functional as F
from ConvBlock import ConvBlock

class Decoder(nn.Module):
  def __init__(self, levels_in_encoder=4, init_filters = 32, layers_per_block = 4, conv_block_kernel_size = 2):
    super(Decoder, self).__init__()
    self.levels_in_encoder = levels_in_encoder
    self.levels_in_decoder = self.levels_in_encoder - 1
    self.init_filters = init_filters
    self.conv_block_kernel_size = conv_block_kernel_size
    self.layers_per_block = layers_per_block
    self.transpose_convs = nn.ModuleList([
        nn.ConvTranspose2d(
            kernel_size=2, stride=2,
            in_channels= self.init_filters*(2**(self.levels_in_encoder-1-i)),
            out_channels= self.init_filters*(2**(self.levels_in_encoder-1-i))
        )
        for i in range(self.levels_in_decoder)
    ])
    self.conv_blocks = nn.ModuleList([
        ConvBlock(
            in_channels= (self.init_filters * (2**(self.levels_in_decoder-1-i)))+(self.init_filters * (2**(self.levels_in_decoder-i))), 
            filters= self.init_filters * (2**(self.levels_in_decoder-1-i)),
            layers_per_block= self.layers_per_block,
            kernel_size= self.conv_block_kernel_size
        )
        for i in range(self.levels_in_decoder)
    ])

  def forward(self, encoder_outputs):
    output = self.conv_blocks[0](torch.cat((self.transpose_convs[0](encoder_outputs[-1]), encoder_outputs[-2]), dim= 1))
    for i in range(self.levels_in_decoder-1):
      output = self.conv_blocks[i+1](torch.cat((self.transpose_convs[i+1](output), encoder_outputs[-3-i]), dim= 1))
    return output
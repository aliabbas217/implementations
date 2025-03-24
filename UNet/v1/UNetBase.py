import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder import Encoder
from Decoder import Decoder


class UNetBase(nn.Module):
  def __init__(self, in_channels=3, init_filters = 32, layers_per_block = 4, levels = 4, kernel_size= 2):
    super(UNetBase, self).__init__()
    self.in_channels = in_channels
    self.init_filters = init_filters
    self.layers_per_block = layers_per_block
    self.kernel_size = kernel_size
    self.levels = levels
    self.encoder = Encoder(
        in_channels=in_channels, init_filters=init_filters, layers_per_block=layers_per_block,
        levels=self.levels, conv_block_kernel_size=kernel_size
    )
    self.decoder = Decoder(
        levels_in_encoder=self.levels, init_filters=init_filters, layers_per_block=layers_per_block,
        conv_block_kernel_size=kernel_size
    )

  def forward(self, x):
    encoder_outputs = self.encoder(x)
    decoder_outputs = self.decoder(encoder_outputs)

    return decoder_outputs
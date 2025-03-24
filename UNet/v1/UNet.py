import torch
import torch.nn as nn
import torch.nn.functional as F
from UNetBase import UNetBase


class UNet(nn.Module):
  def __init__(self, in_channels=3, init_filters = 32, layers_per_block = 4, num_classes = 3, levels = 4, kernel_size= 2):
    super(UNet, self).__init__()
    self.in_channels = in_channels
    self.init_filters = init_filters
    self.layers_per_block = layers_per_block
    self.num_classes = num_classes
    self.kernel_size = kernel_size
    self.levels = levels
    self.u_net_base = UNetBase(
        in_channels=self.in_channels, init_filters=self.init_filters,
        layers_per_block=self.layers_per_block, levels=self.levels, kernel_size=self.kernel_size
    )
    self.final_conv = nn.Conv2d(in_channels=init_filters, out_channels=num_classes, kernel_size=1)


  def forward(self, x):
    base_output = self.u_net_base(x)
    output = self.final_conv(base_output)
    return output
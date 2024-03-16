"""
 > Network architecture of Sea-pix-GAN model
   * Original paper: https://doi.org/10.1016/j.jvcir.2023.104021
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..pix2pix import GeneratorUNet


class DiscriminatorSeaPixGan(nn.Module):
    def __init__(self, in_channels=3):
        super(DiscriminatorSeaPixGan, self).__init__()

        ##########################
        # TODO: impl discriminator
        pass
        ##########################

    def forward(self, img):
        return self.model(img)

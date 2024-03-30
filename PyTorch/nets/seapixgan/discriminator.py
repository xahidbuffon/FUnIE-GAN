"""
 > Network architecture of Sea-pix-GAN model
   * Original paper: https://doi.org/10.1016/j.jvcir.2023.104021
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscriminatorSeaPixGan(nn.Module):
    def __init__(self, in_channels=3):
        super(DiscriminatorSeaPixGan, self).__init__()

        def down_layer(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *down_layer(2*in_channels, 64),
            *down_layer(64, 128),
            *down_layer(128, 256),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, 4, padding=0, bias=False),
            nn.BatchNorm2d(512, momentum=0.8),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 1, 4, padding=0, bias=False)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


"""
 > Network architecture of Sea-pix-GAN model
   * Original paper: https://doi.org/10.1016/j.jvcir.2023.104021
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneratorSeaPixGan(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorSeaPixGan, self).__init__()
    
        # Encoder: a series of Convolution-BatchNorm-ReLU*
        # TODO: check if any parameters are needed (ex. bn, normalize, dropout?)
        self.e1 = _EncodeLayer(in_channels, 64)
        self.e2 = _EncodeLayer(64, 128)
        self.e3 = _EncodeLayer(128, 256)
        self.e4 = _EncodeLayer(256, 512)
        self.e5 = _EncodeLayer(512, 512)
        self.e6 = _EncodeLayer(512, 512)
        self.e7 = _EncodeLayer(512, 512)
        self.e8 = _EncodeLayer(512, 512)
        
        # Decoder: a series of Convolution-BatchNormDropout-ReLU*
        # TODO: check if any parameters are needed
        self.d1 = _DecodeLayer(512, 512)
        self.d2 = _DecodeLayer(512, 512)
        self.d3 = _DecodeLayer(512, 512)
        self.d4 = _DecodeLayer(512, 512)
        self.d5 = _DecodeLayer(512, 256)
        self.d6 = _DecodeLayer(256, 128)
        self.d7 = _DecodeLayer(128, 64)

        self.conv2d = _DecodeLayer(64, 3) # TODO

        self.final = nn.Sequential( # TODO
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)

        # skip connections between mirrored layers i and n-i
        d1 = self.d1(e8)
        d2 = self.d2(d1, e7)
        d3 = self.d3(d2, e6)
        d4 = self.d4(d3, e5)
        d5 = self.d5(d4, e4)
        d6 = self.d6(d5, e3)
        d7 = self.d7(d6, e2)
        cov = self.conv2d(d7, e1)

        return self.final(cov)


class _EncodeLayer(nn.Module):
    """ Decoder: a series of Convolution-BatchNormDropout-ReLU*
    """
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(_EncodeLayer, self).__init__()
        # TODO: impl encode layer

    def forward(self, x):
        return self.model(x)


class _DecodeLayer(nn.Module):
    """ Encoder: a series of Convolution-BatchNorm-ReLU*
    """
    def __init__(self, in_size, out_size, dropout=0.0):
        super(_DecodeLayer, self).__init__()
        # TODO: impl decode layer

    def forward(self, x, skip_input=None):
        x = self.model(x)
        if skip_input is not None: # skip connection is not a must
            x = torch.cat((x, skip_input), 1)
        return x

"""
 > Network architecture of Sea-pix-GAN model
   * Original paper: https://doi.org/10.1016/j.jvcir.2023.104021
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneratorSeaPixGan(nn.Module):
    def __init__(self):
        super(GeneratorSeaPixGan, self).__init__()
    
        self.e1 = _EncodeLayer(3, 64, batch_normalize=False)
        self.e2 = _EncodeLayer(64, 128)
        self.e3 = _EncodeLayer(128, 256)
        self.e4 = _EncodeLayer(256, 512)
        self.e5 = _EncodeLayer(512, 512)
        self.e6 = _EncodeLayer(512, 512)
        self.e7 = _EncodeLayer(512, 512)
        self.e8 = _EncodeLayer(512, 512)
        
        self.d1 = _DecodeLayer(512, 512, dropout=True)
        self.d2 = _DecodeLayer(1024, 512, dropout=True)
        self.d3 = _DecodeLayer(1024, 512, dropout=True)
        self.d4 = _DecodeLayer(1024, 512)
        self.d5 = _DecodeLayer(1024, 256)
        self.d6 = _DecodeLayer(512, 128)
        self.d7 = _DecodeLayer(256, 64)

        self.deconv = nn.ConvTranspose2d(
            in_channels=128, out_channels=3, 
            kernel_size=4, padding=1, bias=False
        )

    def forward(self, x):
        # x: (256×256×3)
        e1 = self.e1(x)   # (128×128×64)
        e2 = self.e2(e1)  # (64×64×128)
        e3 = self.e3(e2)  # (32×32×256)
        e4 = self.e4(e3)  # (16×16×512)
        e5 = self.e5(e4)  # (8×8×512)
        e6 = self.e6(e5)  # (4×4×512)
        e7 = self.e7(e6)  # (2×2×512)
        e8 = self.e8(e7)  # (1×1×512)

        d1 = self.d1(e8, e7)  # (2×2×(512+512))
        d2 = self.d2(d1, e6)  # (4×4×(512+512))
        d3 = self.d3(d2, e5)  # (8×8×(512+512))
        d4 = self.d4(d3, e4)  # (16×16×(512+512))
        d5 = self.d5(d4, e3)  # (32×32×(256+256))
        d6 = self.d6(d5, e2)  # (64×64×(128+128))
        d7 = self.d7(d6, e1)  # (128×128×(64+64))

        final = self.deconv(d7) # (256×256×3)

        return final


class _EncodeLayer(nn.Module):
    """ Encoder: a series of Convolution-BatchNorm-ReLU*
    """
    def __init__(self, in_size, out_size, batch_normalize=True):
        super(_EncodeLayer, self).__init__()
        layers = [nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=4, stride=2, padding=1, bias=False)]
        if batch_normalize: 
            layers.append(nn.BatchNorm2d(num_features=out_size))
        layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class _DecodeLayer(nn.Module):
    """ Decoder: a series of Convolution-BatchNormDropout-ReLU*
    """
    def __init__(self, in_size, out_size, dropout=False):
        super(_DecodeLayer, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels=in_size, out_channels=out_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

"""
 > Network architecture of Sea-pix-GAN model
   * Original paper: https://doi.org/10.1016/j.jvcir.2023.104021
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .discriminator import DiscriminatorSeaPixGan
from .generator import GeneratorSeaPixGan

class SeaPixGAN_Nets:
    def __init__(self, base_model='pix2pix'):
        if base_model=='pix2pix': # default
            self.netG = GeneratorSeaPixGan() 
            self.netD = DiscriminatorSeaPixGan()
        elif base_model=='resnet':
            #TODO: add ResNet support
            pass
        else: 
            pass


""" 
# UGAN have this function for training, not sure if we need it as well

class Gradient_Difference_Loss(nn.Module):
    def __init__(self, alpha=1, chans=3, cuda=True):
        super(Gradient_Difference_Loss, self).__init__()
        self.alpha = alpha
        self.chans = chans
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        SobelX = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        SobelY = [[1, 2, -1], [0, 0, 0], [1, 2, -1]]
        self.Kx = Tensor(SobelX).expand(self.chans, 1, 3, 3)
        self.Ky = Tensor(SobelY).expand(self.chans, 1, 3, 3)

    def get_gradients(self, im):
        gx = F.conv2d(im, self.Kx, stride=1, padding=1, groups=self.chans)
        gy = F.conv2d(im, self.Ky, stride=1, padding=1, groups=self.chans)
        return gx, gy

    def forward(self, pred, true):
        # get graduent of pred and true
        gradX_true, gradY_true = self.get_gradients(true)
        grad_true = torch.abs(gradX_true) + torch.abs(gradY_true)
        gradX_pred, gradY_pred = self.get_gradients(pred)
        grad_pred_a = torch.abs(gradX_pred)**self.alpha + torch.abs(gradY_pred)**self.alpha
        # compute and return GDL
        return 0.5 * torch.mean(grad_true - grad_pred_a)
"""


"""
 > Training pipeline for Sea-pix-GAN models
   * Original paper: https://doi.org/10.1016/j.jvcir.2023.104021
"""

# TODO: add training script
# py libs
import os
import sys
import yaml
import argparse
import numpy as np
from PIL import Image
# pytorch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
# local libs
from nets.seapixgan import SeaPixGan_Nets, Gradient_Difference_Loss
from nets.commons import Weights_Normal, Gradient_Penalty
from utils.data_utils import GetTrainingPairs, GetValImage

## get configs and training options
parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", type=str, default="configs/train_euvp.yaml")
parser.add_argument("--epoch", type=int, default=0, help="which epoch to start from")
parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--l1_weight", type=float, default=100, help="Weight for L1 loss")
parser.add_argument("--ig_weight", type=float, default=1, help="0 for UGAN / 1 for UGAN-P")
parser.add_argument("--gp_weight", type=float, default=10, help="Weight for gradient penalty (D)")
parser.add_argument("--n_critic", type=int, default=5, help="training steps for D per iter w.r.t G")
args = parser.parse_args()

## training params
epoch = args.epoch
num_epochs = args.num_epochs
batch_size =  args.batch_size
lr_rate = args.lr
num_critic = args.n_critic
lambda_gp = args.gp_weight # 10 (default)  
lambda_1 = args.l1_weight  # 100 (default) 
lambda_2 = args.ig_weight  # UGAN-P (default)
model_v = "UGAN_P" if lambda_2 else "UGAN" 
# load the data config file
with open(args.cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
# get info from config file
dataset_name = cfg["dataset_name"] 
dataset_path = cfg["dataset_path"]
channels = cfg["chans"]
img_width = cfg["im_width"]
img_height = cfg["im_height"] 
val_interval = cfg["val_interval"]
ckpt_interval = cfg["ckpt_interval"]

## create dir for model and validation data
samples_dir = "samples/%s/%s" % (model_v, dataset_name)
checkpoint_dir = "checkpoints/%s/%s/" % (model_v, dataset_name)
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)


""" UGAN specifics: loss functions and patch-size
-------------------------------------------------"""
L1_G  = torch.nn.L1Loss() # l1 loss term
L_bce = torch.nn.BCELoss() # Binary cross entropy
#L1_gp = Gradient_Penalty() # wgan_gp loss term
#L_gdl = Gradient_Difference_Loss() # GDL loss term


# Initialize generator and discriminator
seapixgan_ = SeaPixGan_Nets(base_model='pix2pix')
generator = seapixgan_.netG
discriminator = seapixgan_.netD

# see if cuda is available
if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    #L1_gp.cuda()
    L1_G = L1_G.cuda()
    #L_gdl = L_gdl.cuda()
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

# Initialize weights or load pretrained models
if args.epoch == 0:
    generator.apply(Weights_Normal)
    discriminator.apply(Weights_Normal)
else:
    generator.load_state_dict(torch.load("checkpoints/%s/%s/generator_%d.pth" % (model_v, dataset_name, args.epoch)))
    discriminator.load_state_dict(torch.load("checkpoints/%s/%s/discriminator_%d.pth" % (model_v, dataset_name, epoch)))
    print ("Loaded model from epoch %d" %(epoch))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_rate)


## Data pipeline
transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    GetTrainingPairs(dataset_path, dataset_name, transforms_=transforms_),
    batch_size = batch_size,
    shuffle = True,
    num_workers = 8,
)

val_dataloader = DataLoader(
    GetValImage(dataset_path, dataset_name, transforms_=transforms_, sub_dir='validation'),
    batch_size=4,
    shuffle=True,
    num_workers=1,
)


## Training pipeline
for epoch in range(epoch, num_epochs):
    for i, batch in enumerate(dataloader):
        # Model inputs
        imgs_distorted = Variable(batch["A"].type(Tensor))
        imgs_good_gt = Variable(batch["B"].type(Tensor))

        ## Train Discriminator
        optimizer_D.zero_grad()
        
        imgs_fake = generator(imgs_distorted)
        pred_real = discriminator(imgs_good_gt)
        pred_fake = discriminator(imgs_fake)
        # ALL L_bce LOSSES WOULD BE BETTER IF THE SECOND
        # ARGUMENT IS MANUALLY PLACED (TENSOR SIZE OF IMAGE!)
        loss_D_gen = L_bce(imgs_fake, torch.zeros(transforms.functional.get_image_size(imgs_fake)))
        loss_D_real = L_bce(imgs_good_gt, torch.ones(transforms.functional.get_image_size(imgs_good_gt)))
        loss_D = loss_D_gen + loss_D_real #-torch.mean(pred_real) + torch.mean(pred_fake) # wgan 
        #gradient_penalty = L1_gp(discriminator, imgs_good_gt.data, imgs_fake.data)
        #loss_D += lambda_gp * gradient_penalty # Eq.2 paper 
        loss_D.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        ## Train Generator at 1:num_critic rate 
        if i % num_critic == 0:
            imgs_fake = generator(imgs_distorted)
            pred_fake = discriminator(imgs_fake.detach())
            #loss_gen = -torch.mean(pred_fake)
            loss_1 = L1_G(imgs_fake, imgs_good_gt)
            loss_bce = L_bce(imgs_fake, torch.ones(transforms.functional.get_image_size(imgs_fake)))
            #loss_gdl = L_gdl(imgs_fake, imgs_good_gt)
            # Total loss: Eq.6 in paper
            loss_G = loss_bce + lambda_1 * loss_1  #lambda_2 * loss_gdl   
            loss_G.backward()
            optimizer_G.step()

        ## Print log
        if not i%50:
            sys.stdout.write("\r[Epoch %d/%d: batch %d/%d] [DLoss: %.3f, GLoss: %.3f]"
                              %(
                                epoch, num_epochs, i, len(dataloader),
                                loss_D.item(), loss_G.item(),
                               )
            )
        ## If at sample interval save image
        batches_done = epoch * len(dataloader) + i
        if batches_done % val_interval == 0:
            imgs = next(iter(val_dataloader))
            imgs_val = Variable(imgs["val"].type(Tensor))
            imgs_gen = generator(imgs_val)
            img_sample = torch.cat((imgs_val.data, imgs_gen.data), -2)
            save_image(img_sample, "samples/%s/%s/%s.png" % (model_v, dataset_name, batches_done), nrow=5, normalize=True)

    ## Save model checkpoints
    if (epoch % ckpt_interval == 0):
        torch.save(generator.state_dict(), "checkpoints/%s/%s/generator_%d.pth" % (model_v, dataset_name, epoch))
        torch.save(discriminator.state_dict(), "checkpoints/%s/%s/discriminator_%d.pth" % (model_v, dataset_name, epoch))


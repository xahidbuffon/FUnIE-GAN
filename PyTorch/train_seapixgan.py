"""
 > Training pipeline for Sea-pix-GAN models
   * Original paper: https://doi.org/10.1016/j.jvcir.2023.104021
"""

# py libs
import os
import sys
import yaml
import argparse
from PIL import Image
# pytorch libs
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
# local libs
from nets.seapixgan import SeaPixGan_Nets
from nets.commons import Weights_Normal
from utils.data_utils import GetTrainingPairs, GetValImage

## get configs and training options
parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", type=str, default="configs/train_euvp.yaml")
parser.add_argument("--epoch", type=int, default=0, help="which epoch to start from")
parser.add_argument("--num_epochs", type=int, default=150, help="number of epochs of training")
parser.add_argument("--n_critic", type=int, default=5, help="training steps for D per iter w.r.t G")
args = parser.parse_args()

## training params
epoch = args.epoch
num_epochs = args.num_epochs
num_critic = args.n_critic
model_v = "Sea-pix-GAN" 
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


""" Sea-pix-GAN specifics: loss functions and specified hyperparams
-------------------------------------------------"""
L1_G  = torch.nn.L1Loss() # l1 loss term
L_BCE = torch.nn.BCELoss() # Binary cross entropy
lambda_1 = 100
batch_size = 64
lr = 2 * 10e-4
beta_1 = 0.5
beta_2 = 0.999 # not specified, use PyTorch default


# Initialize generator and discriminator
seapixgan_ = SeaPixGan_Nets(base_model='pix2pix')
generator = seapixgan_.netG
discriminator = seapixgan_.netD

# see if cuda is available
if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    L1_G = L1_G.cuda()
    L_BCE = L_BCE.cuda()
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
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta_1, beta_2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta_1, beta_2))


## Data pipeline
# TODO: make sure preprocessing is correct
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
        imgs_distorted = Variable(batch["A"].type(Tensor)) # x: input underwater img
        imgs_good_gt = Variable(batch["B"].type(Tensor)) # y: ground truth underwater img

        ## Train Discriminator
        optimizer_D.zero_grad()
        
        imgs_fake = generator(imgs_distorted)
        pred_real = discriminator(imgs_good_gt, imgs_distorted)
        pred_fake = discriminator(imgs_fake, imgs_distorted)
        # ALL L_bce LOSSES WOULD BE BETTER IF THE SECOND
        # ARGUMENT IS MANUALLY PLACED (TENSOR SIZE OF IMAGE!)
        loss_D_gen = L_BCE(pred_fake, torch.zeros_like(pred_fake))
        loss_D_real = L_BCE(pred_real, torch.ones_like(pred_real))
        loss_D = loss_D_gen + loss_D_real
        loss_D.backward()
        optimizer_D.step()

        ## Train Generator at 1:num_critic rate 
        optimizer_G.zero_grad()
        if i % num_critic == 0:
            # regenerate imgs
            imgs_fake = generator(imgs_distorted)
            pred_fake = discriminator(imgs_fake.detach())
            # calculate loss function
            loss_1 = L1_G(imgs_fake, imgs_good_gt)
            loss_cgan = L_BCE(pred_fake, torch.ones_like(pred_fake))
            loss_G = loss_cgan + lambda_1 * loss_1 # Total loss: Eq.4 in paper
            # backward & steps
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


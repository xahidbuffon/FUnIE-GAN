"""
 > Script for testing .pth models  
    * set model_name ('funiegan'/'ugan') and  model path
    * set data_dir (input) and sample_dir (output) 
"""
# py libs
import os
import time
import argparse
import numpy as np
from PIL import Image
from glob import glob
from ntpath import basename
from os.path import join, exists
# pytorch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms

## options
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/test/A/")
parser.add_argument("--sample_dir", type=str, default="data/output/")
parser.add_argument("--model_name", type=str, default="funiegan") # or "ugan"
parser.add_argument("--model_path", type=str, default="models/funie_generator.pth")
opt = parser.parse_args()

## checks
assert exists(opt.model_path), "model not found"
os.makedirs(opt.sample_dir, exist_ok=True)
is_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor 

## model arch
if opt.model_name.lower()=='funiegan':
    from nets import funiegan
    model = funiegan.GeneratorFunieGAN()
elif opt.model_name.lower()=='ugan':
    from nets.ugan import UGAN_Nets
    model = UGAN_Nets(base_model='pix2pix').netG
else: 
    # other models
    pass

## load weights
model.load_state_dict(torch.load(opt.model_path))
if is_cuda: model.cuda()
model.eval()
print ("Loaded model from %s" % (opt.model_path))

## data pipeline
img_width, img_height, channels = 256, 256, 3
transforms_ = [transforms.Resize((img_height, img_width), Image.BICUBIC),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
transform = transforms.Compose(transforms_)


## testing loop
times = []
test_files = sorted(glob(join(opt.data_dir, "*.*")))
for path in test_files:
    inp_img = transform(Image.open(path))
    inp_img = Variable(inp_img).type(Tensor).unsqueeze(0)
    # generate enhanced image
    s = time.time()
    gen_img = model(inp_img)
    times.append(time.time()-s)
    # save output
    img_sample = torch.cat((inp_img.data, gen_img.data), -1)
    save_image(img_sample, join(opt.sample_dir, basename(path)), normalize=True)
    print ("Tested: %s" % path)

## run-time    
if (len(times) > 1):
    print ("\nTotal samples: %d" % len(test_files)) 
    # accumulate frame processing times (without bootstrap)
    Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:]) 
    print ("Time taken: %d sec at %0.3f fps" %(Ttime, 1./Mtime))
    print("Saved generated images in in %s\n" %(opt.sample_dir))




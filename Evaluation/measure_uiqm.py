"""
# > Script for measuring quantitative performances in terms of
#    - Structural Similarity Metric (SSIM) 
#    - Peak Signal to Noise Ratio (PSNR)
# > Maintainer: https://github.com/xahidbuffon
"""
## python libs
import numpy as np
from PIL import Image, ImageOps
from glob import glob
from os.path import join
from ntpath import basename
## local libs
from uqim_utils import getUIQM


def measure_UIQMs(dir_name, im_res=(256, 256)):
    paths = sorted(glob(join(dir_name, "*.*")))
    uqims = []
    for img_path in paths:
        im = Image.open(img_path).resize(im_res)
        uiqm = getUIQM(np.array(im))
        uqims.append(uiqm)
    return np.array(uqims)


## UIQMs of the distorted input images 
inp_dir = "/home/xahid/datasets/released/EUVP/test_samples/Inp/"
inp_uqims = measure_UIQMs(inp_dir)
print ("Input UIQMs >> Mean: {0} std: {1}".format(np.mean(inp_uqims), np.std(inp_uqims)))

## UIQMs of the enhanceded output images
gen_dir = "exp_data/euvp_test/funie-gan/" 
gen_uqims = measure_UIQMs(gen_dir)
print ("Enhanced UIQMs >> Mean: {0} std: {1}".format(np.mean(gen_uqims), np.std(gen_uqims)))




#!/usr/bin/env python
"""
# > Script for testing FUnIE-GAN 
#    - Paper: https://arxiv.org/pdf/1903.09766.pdf
#
# > Notes and Usage:
#    - set data_dir and model paths
#    - python test_funieGAN.py
#
# Maintainer: Jahid (email: islam034@umn.edu)
# Interactive Robotics and Vision Lab (http://irvlab.cs.umn.edu/)
# Any part of this repo can be used for academic and educational purposes only
"""
import os
import time
import ntpath
import numpy as np
from scipy import misc
from keras.models import model_from_json
## local libs
from utils.plot_utils import save_test_samples_funieGAN
from utils.data_utils import getPaths, read_and_resize, preprocess, deprocess

## for testing arbitrary local data
data_dir = "data/test/random/"
from utils.data_utils import get_local_test_data
test_paths = getPaths(data_dir)
print ("{0} test images are loaded".format(len(test_paths)))

## create dir for log and (sampled) validation data
samples_dir = "data/output/"
if not os.path.exists(samples_dir): os.makedirs(samples_dir)

dataset_name = 'underwater_dark' # options: {'underwater_imagenet', 'underwater_dark'}
checkpoint_dir  = 'checkpoints/funieGAN/' + dataset_name + '/'
model_name_by_epoch = "model_26970_"
model_h5 = checkpoint_dir + model_name_by_epoch + ".h5"  
model_json = checkpoint_dir + model_name_by_epoch + ".json"
# sanity
assert (os.path.exists(model_h5) and os.path.exists(model_json))

# load model
with open(model_json, "r") as json_file:
    loaded_model_json = json_file.read()
funie_gan_generator = model_from_json(loaded_model_json)

times = []; s = time.time()
# load weights into new model
funie_gan_generator.load_weights(model_h5)
tot = time.time()-s
times.append(tot)
print("\nLoaded data and model")

# testing loop
for img_path in test_paths:
    # prepare data
    img_name = ntpath.basename(img_path).split('.')[0]
    im = read_and_resize(img_path, (256, 256))
    im = preprocess(im)
    im = np.expand_dims(im, axis=0) # (1,256,256,3)
    # generate enhanced image
    s = time.time()
    gen = funie_gan_generator.predict(im)
    gen = deprocess(gen) # Rescale to 0-1
    tot = time.time()-s
    times.append(tot)
    # save sample images
    misc.imsave(samples_dir+img_name+'_real.png', im[0])
    misc.imsave(samples_dir+img_name+'_gen.png', gen[0])

# some statistics    
num_test = len(test_paths)
if (num_test==0):
    print ("\nFound no images for test")
else:
    print ("\nTotal images: {0}".format(num_test)) 
    Ttime = sum(times)
    print ("Time taken: {0} sec at {1} fps".format(Ttime, num_test/Ttime))
    print("\nSaved generated images in in {0}\n".format(samples_dir))


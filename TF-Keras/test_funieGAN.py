"""
# > Script for testing FUnIE-GAN 
# > Notes and Usage:
#    - set data_dir and model paths
#    - python test_funieGAN.py
"""
import os
import time
import ntpath
import numpy as np
from PIL import Image
from os.path import join, exists
from keras.models import model_from_json
## local libs
from utils.data_utils import getPaths, read_and_resize, preprocess, deprocess

## for testing arbitrary local data
data_dir = "../data/test/A/"
from utils.data_utils import get_local_test_data
test_paths = getPaths(data_dir)
print ("{0} test images are loaded".format(len(test_paths)))

## create dir for log and (sampled) validation data
samples_dir = "../data/output/"
if not exists(samples_dir): os.makedirs(samples_dir)

## test funie-gan
checkpoint_dir  = 'models/gen_p/'
model_name_by_epoch = "model_15320_" 
## test funie-gan-up
#checkpoint_dir  = 'models/gen_up/'
#model_name_by_epoch = "model_35442_" 

model_h5 = checkpoint_dir + model_name_by_epoch + ".h5"  
model_json = checkpoint_dir + model_name_by_epoch + ".json"
# sanity
assert (exists(model_h5) and exists(model_json))

# load model
with open(model_json, "r") as json_file:
    loaded_model_json = json_file.read()
funie_gan_generator = model_from_json(loaded_model_json)
# load weights into new model
funie_gan_generator.load_weights(model_h5)
print("\nLoaded data and model")

# testing loop
times = []; s = time.time()
for img_path in test_paths:
    # prepare data
    inp_img = read_and_resize(img_path, (256, 256))
    im = preprocess(inp_img)
    im = np.expand_dims(im, axis=0) # (1,256,256,3)
    # generate enhanced image
    s = time.time()
    gen = funie_gan_generator.predict(im)
    gen_img = deprocess(gen)[0]
    tot = time.time()-s
    times.append(tot)
    # save output images
    out_img = np.hstack((inp_img, gen_img))
    img_name = ntpath.basename(img_path)
    Image.fromarray(out_img).save(join(samples_dir, img_name))

# some statistics    
num_test = len(test_paths)
if (num_test==0):
    print ("\nFound no images for test")
else:
    print ("\nTotal images: {0}".format(num_test)) 
    # accumulate frame processing times (without bootstrap)
    Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:]) 
    print ("Time taken: {0} sec at {1} fps".format(Ttime, 1./Mtime))
    print("\nSaved generated images in in {0}\n".format(samples_dir))


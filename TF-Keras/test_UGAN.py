"""
# > Script for testing UGAN with different settings. 
#    - A simplified implementation of the original repo 
#    - Original repo: github.com/cameronfabbri/Underwater-Color-Correction
# > Notes and Usage:
#    - set LOSS_METHOD, NETWORK, DATA, and data_dir
#    - python test_UGAN.py
# Maintainer: Jahid (email: islam034@umn.edu)
"""
# imports
import os
import sys
import time
import ntpath
from tqdm import tqdm
from scipy import misc
import tensorflow as tf
#export TF_CPP_MIN_LOG_LEVEL=2
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# local imports
sys.path.insert(0, 'nets/')
import numpy as np
from utils.data_utils import getPaths, read_and_resize, preprocess

# test set directories
data_dir = "../data/test/random/"
test_paths = getPaths(data_dir)
print ("{0} test images are loaded".format(len(test_paths)))

# change to load the right model (checkpoint_dir)
LOSS_METHOD   = 'wgan' # options: {'gan', 'least_squares', 'wgan'}
NETWORK       = 'pix2pix' # options: {'pix2pix', 'resnet'}
DATA          = 'underwater_imagenet' # options: {'underwater_imagenet', 'underwater_dark'}
checkpoint_dir  = 'checkpoints/UGAN/'+LOSS_METHOD+'_'+NETWORK+'_'+DATA+'/'

# local imports
if NETWORK == 'pix2pix': from pix2pix import *
if NETWORK == 'resnet': from resnet import *


# global step that is saved with a model to keep track of how many steps/epochs
global_step = tf.Variable(0, name='global_step', trainable=False)
# underwater image
image_u = tf.placeholder(tf.float32, shape=(1, 256, 256, 3), name='image_u')
# generated corrected colors
gen_image  = netG(image_u)

saver = tf.train.Saver(max_to_keep=1)
init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
sess = tf.Session()
sess.run(init)

# restore model
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and ckpt.model_checkpoint_path:
    print "Restoring previous model..."
    try:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print ("Model restored")
    except:
        print ("Could not restore model")
        pass

# keep track of time
times = []; s = time.time()

# keep the samples   
samples_dir = "../data/output/"
if not os.path.exists(samples_dir): os.makedirs(samples_dir)

# testing loop
step = int(sess.run(global_step))
for img_path in tqdm(test_paths):
    # prepare data
    img_name = ntpath.basename(img_path)
    img_name = img_name.split('.')[0]
    batch_images = np.empty((1, 256, 256, 3), dtype=np.float32)
    a_img = read_and_resize(img_path, (256, 256))
    a_img = preprocess(a_img)
    batch_images[0, ...] = a_img
    # generate enhanced image
    s = time.time()
    gen_images = sess.run(gen_image, feed_dict={image_u:batch_images})
    tot = time.time()-s
    times.append(tot)
    # save sample images
    gen_images = np.asarray(gen_images)
    for gen, real in zip(gen_images, batch_images):
        misc.imsave(samples_dir+img_name+'_real.png', real)
        misc.imsave(samples_dir+img_name+'_gen.png', gen)

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




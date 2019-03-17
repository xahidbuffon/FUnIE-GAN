
import os
import cv2
import sys
import time
import ntpath
from tqdm import tqdm
from scipy import misc
import tensorflow as tf
#export TF_CPP_MIN_LOG_LEVEL=2
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# my imports
sys.path.insert(0, 'nets/')
import numpy as np
from utils.data_utils import getPaths, read_and_resize, preprocess

data_dir = "data/test/"
test_paths = getPaths(data_dir)
print ("{0} test images are loaded".format(len(test_paths)))


LOSS_METHOD   = 'least_squares'
NETWORK       = 'resnet'
DATA          = 'underwater_imagenet'
checkpoint_dir  = 'checkpoints/UGAN/'+LOSS_METHOD+'_'+NETWORK+'_'+DATA+'/'

if NETWORK == 'pix2pix': from pix2pix import *
if NETWORK == 'resnet': from resnet import *

times = []; s = time.time()

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


ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and ckpt.model_checkpoint_path:
    print "Restoring previous model..."
    try:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print ("Model restored")
    except:
        print ("Could not restore model")
        pass

tot = time.time()-s
times.append(tot)
   

samples_dir = "data/output/uGAN/"
if not os.path.exists(samples_dir): os.makedirs(samples_dir)

step = int(sess.run(global_step))
for img_path in tqdm(test_paths):
    img_name = ntpath.basename(img_path)
    img_name = img_name.split('.')[0]
    batch_images = np.empty((1, 256, 256, 3), dtype=np.float32)

    a_img = read_and_resize(img_path, (256, 256))
    a_img = preprocess(a_img)
    batch_images[0, ...] = a_img

    s = time.time()
    gen_images = sess.run(gen_image, feed_dict={image_u:batch_images})
    tot = time.time()-s
    times.append(tot)
    gen_images = np.asarray(gen_images)
    for gen, real in zip(gen_images, batch_images):
        misc.imsave(samples_dir+img_name+'_real.png', real)
        misc.imsave(samples_dir+img_name+'_gen.png', gen)


num_test = len(test_paths)
if (num_test==0):
    print ("\nFound no images for test")
else:
    print ("\nTotal images: {0}".format(num_test)) 
    Ttime = sum(times)
    print ("Time taken: {0} sec at {1} fps".format(Ttime, num_test/Ttime))
    print("\nSaved generated images in in {0}\n".format(samples_dir))




import tensorflow as tf
from scipy import misc
from tqdm import tqdm
import numpy as np
import ntpath
import time
import sys
import cv2
import os
#export TF_CPP_MIN_LOG_LEVEL=2
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# my imports
sys.path.insert(0, 'nets/')
from utils.data_loader import getPaths, read_and_resize, preprocess

data_dir = "data/selected/human/"
test_paths = getPaths(data_dir)
print ("{0} test images are loaded".format(len(test_paths)))


NUM_LAYERS    = 8
LOSS_METHOD   = 'wgan'
NETWORK       = 'pix2pix'
DATA          = 'underwater_imagenet'
checkpoint_dir  = 'checkpoints/'+LOSS_METHOD+'_'+NETWORK+'_'+DATA+'/run2/'

if NETWORK == 'pix2pix': from pix2pix import *
if NETWORK == 'resnet': from resnet import *

# global step that is saved with a model to keep track of how many steps/epochs
global_step = tf.Variable(0, name='global_step', trainable=False)
# underwater image
image_u = tf.placeholder(tf.float32, shape=(1, 256, 256, 3), name='image_u')
# generated corrected colors
if NUM_LAYERS==8:
    layers     = netG8_encoder(image_u)
    gen_image  = netG8_decoder(layers)
elif NUM_LAYERS==16:
    layers     = netG16_encoder(image_u)
    gen_image  = netG16_decoder(layers)
else: pass

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
   

samples_dir = "data/samples/uGAN/"
step = int(sess.run(global_step))
times = []
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

avg_time = float(np.mean(np.asarray(times)))
print ("Time taken in average {0} secs".format(avg_time))



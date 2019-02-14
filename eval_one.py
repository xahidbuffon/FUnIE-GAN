'''

   Evaluation file for only a single image.

'''

import cPickle as pickle
import tensorflow as tf
from scipy import misc
from tqdm import tqdm
import numpy as np
import argparse
import random
import ntpath
import sys
import os
import time
import time
import glob
import cPickle as pickle
from tqdm import tqdm
import cv2

sys.path.insert(0, 'ops/')
sys.path.insert(0, 'nets/')
from tf_ops import *

import data_ops

if __name__ == '__main__':
   
   LEARNING_RATE = 0.001
   LOSS_METHOD   = "gan"
   UIQM_WEIGHT   = 0.0
   NUM_LAYERS    = 16
   BATCH_SIZE    = 1
   L1_WEIGHT     = 100.0
   IG_WEIGHT     = 0.0
   NETWORK       = "pix2pix"
   AUGMENT       = 0
   EPOCHS        = 1
   DATA          = 'underwater_imagenet'
   LAB           = False
   
   EXPERIMENT_DIR  = '/mnt/data2/color_correction_related/checkpoints/LOSS_METHOD_'+LOSS_METHOD\
                     +'/NETWORK_'+NETWORK\
                     +'/UIQM_WEIGHT_'+str(UIQM_WEIGHT)\
                     +'/NUM_LAYERS_'+str(NUM_LAYERS)\
                     +'/L1_WEIGHT_'+str(L1_WEIGHT)\
                     +'/IG_WEIGHT_'+str(IG_WEIGHT)\
                     +'/AUGMENT_'+str(AUGMENT)\
                     +'/DATA_'+DATA\
                     +'/LAB_'+str(LAB)+'/'\

   test_image = 'test_data/ramius_ir.png'

   print
   print 'LEARNING_RATE: ',LEARNING_RATE
   print 'LOSS_METHOD:   ',LOSS_METHOD
   print 'BATCH_SIZE:    ',BATCH_SIZE
   print 'L1_WEIGHT:     ',L1_WEIGHT
   print 'IG_WEIGHT:     ',IG_WEIGHT
   print 'NETWORK:       ',NETWORK
   print 'EPOCHS:        ',EPOCHS
   print 'DATA:          ',DATA
   print

   if NETWORK == 'pix2pix': from pix2pix import *
   if NETWORK == 'resnet':  from resnet import *

   # global step that is saved with a model to keep track of how many steps/epochs
   global_step = tf.Variable(0, name='global_step', trainable=False)

   # underwater image
   image_u = tf.placeholder(tf.float32, shape=(1, 256, 256, 3), name='image_u')

   # generated corrected colors
   if NUM_LAYERS == 16:
      layers    = netG16_encoder(image_u)
      gen_image = netG16_decoder(layers)
   if NUM_LAYERS == 12:
      layers    = netG12_encoder(image_u)
      gen_image = netG12_decoder(layers)
   if NUM_LAYERS == 10:
      layers    = netG10_encoder(image_u)
      gen_image = netG10_decoder(layers)
   if NUM_LAYERS == 8:
      layers    = netG8_encoder(image_u)
      gen_image = netG8_decoder(layers)
   if NUM_LAYERS == 4:
      layers    = netG4_encoder(image_u)
      gen_image = netG4_decoder(layers)

   saver = tf.train.Saver(max_to_keep=1)

   init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
   sess = tf.Session()
   sess.run(init)

   ckpt = tf.train.get_checkpoint_state(EXPERIMENT_DIR)
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         pass
   
   step = int(sess.run(global_step))

   img_name = ntpath.basename(test_image)
   img_name = img_name.split('.')[0]

   batch_images = np.empty((1, 256, 256, 3), dtype=np.float32)

   #a_img = misc.imread(test_image).astype('float32')
   a_img = cv2.imread(test_image)
   a_img = cv2.cvtColor(a_img, cv2.COLOR_BGR2RGB)
   a_img = a_img.astype('float32')
   a_img = misc.imresize(a_img, (256, 256, 3))
   a_img = data_ops.preprocess(a_img)
   a_img = np.expand_dims(a_img, 0)
   batch_images[0, ...] = a_img

   gen_images = np.asarray(sess.run(gen_image, feed_dict={image_u:batch_images}))

   misc.imsave('test_data/'+img_name+'_real.png', batch_images[0])
   misc.imsave('test_data/'+img_name+'_gen.png', gen_images[0])


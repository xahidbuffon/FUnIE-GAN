'''

   Takes in a video file and outputs two videos: One being the corrected video, and
   another being the comparison of the two.

   This will convert the video to 256x256 due to the architecture of the network.
   It will, however, keep the same framerate.

'''

import cPickle as pickle
import tensorflow as tf
from scipy import misc
from tqdm import tqdm
import numpy as np
import argparse
import random
import ntpath
import math
import time
import glob
import sys
import cv2
import os

sys.path.insert(0, 'ops/')
sys.path.insert(0, 'nets/')
from tf_ops import *
import data_ops

if __name__ == '__main__':

   if len(sys.argv) < 3:
      print 'Usage:'
      print 'python convert_video.py [info.pkl] [video.mp4]'
      exit()
   
   pkl_file = open(sys.argv[1], 'rb')
   a = pickle.load(pkl_file)

   video_file = sys.argv[2]

   video_dir     = video_file.split('.')[0]
   original_dir  = video_dir+'/original/'
   corrected_dir = video_dir+'/corrected/'

   try: os.makedirs(original_dir)
   except: pass
   try: os.makedirs(corrected_dir)
   except: pass

   LEARNING_RATE = a['LEARNING_RATE']
   LOSS_METHOD   = a['LOSS_METHOD']
   UIQM_WEIGHT   = a['UIQM_WEIGHT']
   NUM_LAYERS    = a['NUM_LAYERS']
   BATCH_SIZE    = a['BATCH_SIZE']
   L1_WEIGHT     = a['L1_WEIGHT']
   IG_WEIGHT     = a['IG_WEIGHT']
   NETWORK       = a['NETWORK']
   AUGMENT       = a['AUGMENT']
   EPOCHS        = a['EPOCHS']
   DATA          = a['DATA']
   LAB           = a['LAB']
   
   EXPERIMENT_DIR  = 'checkpoints/LOSS_METHOD_'+LOSS_METHOD\
                     +'/NETWORK_'+NETWORK\
                     +'/UIQM_WEIGHT_'+str(UIQM_WEIGHT)\
                     +'/NUM_LAYERS_'+str(NUM_LAYERS)\
                     +'/L1_WEIGHT_'+str(L1_WEIGHT)\
                     +'/IG_WEIGHT_'+str(IG_WEIGHT)\
                     +'/AUGMENT_'+str(AUGMENT)\
                     +'/DATA_'+DATA\
                     +'/LAB_'+str(LAB)+'/'\

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
   

   '''
      extracting video and send through network
      videos could potentially be very long, so just going to save out
      every image and string them together after. Don't want to run out
      of memory holding a ton of images
   '''
   vidcap = cv2.VideoCapture(video_file)
   fps = int(math.floor(vidcap.get(cv2.CAP_PROP_FPS)))
   success, vimg = vidcap.read()
   count = 0
   while success:
      success, vimg = vidcap.read()
      if not success: continue
      vimg = misc.imresize(vimg, (256,256,3))
      cv2.imwrite(video_dir+'/original/frame_%d.png'%count, vimg)
      count += 1

   for img_path in tqdm(sorted(glob.glob(original_dir+'*.png'))):

      img_name = ntpath.basename(img_path)
      img_name = img_name.split('.')[0]

      batch_images = np.empty((1, 256, 256, 3), dtype=np.float32)

      a_img = misc.imread(img_path).astype('float32')
      a_img = misc.imresize(a_img, (256, 256, 3))
      a_img = data_ops.preprocess(a_img)
      batch_images[0, ...] = a_img

      s = time.time()
      gen_images = np.asarray(sess.run(gen_image, feed_dict={image_u:batch_images}))

      for gen, real in zip(gen_images, batch_images):
         misc.imsave(corrected_dir+img_name+'.png', gen)

   # create corrected video
   cmd = 'ffmpeg -framerate '+str(fps)+' -i '+corrected_dir+'frame_%d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ' + video_dir+'/corrected.mp4'
   os.system(cmd)

   # create original video in size 256x256
   cmd = 'ffmpeg -framerate '+str(fps)+' -i '+original_dir+'frame_%d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ' + video_dir+'/original.mp4'
   os.system(cmd)

   # create side by side video
   cmd = 'ffmpeg -i ' + video_dir + '/original.mp4 -i ' + video_dir + '/corrected.mp4 -filter_complex \'[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]\' -map [vid] -c:v libx264 -crf 23 -preset veryfast '+video_dir+'/comparison.mp4'
   os.system(cmd)

   # remove image files
   os.system('rm -rf '+original_dir + ' ' + corrected_dir)

   print
   print
   print 'Videos saved to '+video_dir+'/'

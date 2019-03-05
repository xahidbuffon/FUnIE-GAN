'''

Operations used for data management

'''

from __future__ import division
from __future__ import absolute_import

from scipy import misc
from skimage import color
import collections
import tensorflow as tf
import numpy as np
import math
import time
import random
import glob
import os
import fnmatch
import cPickle as pickle
import cv2

# [-1,1] -> [0, 255]
def deprocess(x):
   return (x+1.0)*127.5

# [0,255] -> [-1, 1]
def preprocess(x):
   return (x/127.5)-1.0

'''
   Augment images - a is distorted
'''
def augment(a_img, b_img):

   # randomly interpolate
   a = random.random()
   a_img = a_img*(1-a) + b_img*a

   # flip image left right
   r = random.random()
   if r < 0.5:
      a_img = np.fliplr(a_img)
      b_img = np.fliplr(b_img)
   
   # flip image up down
   r = random.random()
   if r < 0.5:
      a_img = np.flipud(a_img)
      b_img = np.flipud(b_img)
   
   '''
   # kernel for gaussian blurring
   kernel = np.ones((5,5),np.float32)/25

   # perform some gaussian blur on distorted image
   r = random.random()
   if r < 0.5:
      a_img = cv2.filter2D(a_img,-1,kernel)

   # resize to 286x286 and perform a random crop
   r = random.random()
   if r < 0.5:
      a_img = misc.imresize(a_img, (286, 286,3))
      b_img = misc.imresize(b_img, (286, 286,3))

      rand_x = random.randint(0,50)
      rand_y = random.randint(0,50)

      a_img = a_img[rand_x:, rand_y:, :]
      b_img = b_img[rand_x:, rand_y:, :]

      a_img = misc.imresize(a_img, (256,256,3))
      b_img = misc.imresize(b_img, (256,256,3))
   '''

   return a_img, b_img

def getPaths(data_dir):
   exts = ['*.png','*.PNG','*.jpg','*.JPG', '*.JPEG']
   image_paths = []
   for pattern in exts:
      for d, s, fList in os.walk(data_dir):
         for filename in fList:
            if fnmatch.fnmatch(filename, pattern):
               fname_ = os.path.join(d,filename)
               image_paths.append(fname_)
   return np.asarray(image_paths)



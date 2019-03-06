'''

Operations used for data management

'''

from __future__ import division
from __future__ import absolute_import
import os
import random
import fnmatch
import numpy as np


def deprocess(x):
    # [-1,1] -> [0, 255]
    return (x+1.0)*127.5


def preprocess(x):
    # [0,255] -> [-1, 1]
    return (x/127.5)-1.0



def augment(a_img, b_img):
    """
       Augment images - a is distorted
    """
    # randomly interpolate
    a = random.random()
    a_img = a_img*(1-a) + b_img*a

    # flip image left right
    r = random.random()
    if r < 0.25:
        a_img = np.fliplr(a_img)
        b_img = np.fliplr(b_img)
    
    # flip image up down
    r = random.random()
    if r < 0.25:
        a_img = np.flipud(a_img)
        b_img = np.flipud(b_img) 
    return a_img, b_img



def getPaths(data_dir):
    exts = ['*.png','*.PNG','*.jpg','*.JPG', '*.JPEG']
    image_paths = []
    for pattern in exts:
        for d, s, fList in os.walk(data_dir):
            for filename in fList:
                if (fnmatch.fnmatch(filename, pattern)):
                    fname_ = os.path.join(d,filename)
                    image_paths.append(fname_)
    return np.asarray(image_paths)



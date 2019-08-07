#!/usr/bin/env python
"""
# > Various modules for handling data 
#
# Maintainer: Jahid (email: islam034@umn.edu)
# Interactive Robotics and Vision Lab (http://irvlab.cs.umn.edu/)
# Any part of this repo can be used for academic and educational purposes only
"""
from __future__ import division
from __future__ import absolute_import
import os
import random
import fnmatch
import numpy as np
from scipy import misc


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
    if (random.random() < 0.25):
        a_img = np.fliplr(a_img)
        b_img = np.fliplr(b_img)
    # flip image up down
    if (random.random() < 0.25):
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


def read_and_resize(path, img_res):
    img = misc.imread(path, mode='RGB').astype(np.float)  
    img = misc.imresize(img, img_res)
    return img


def read_and_resize_pair(pathA, pathB, img_res):
    img_A = misc.imread(pathA, mode='RGB').astype(np.float)  
    img_A = misc.imresize(img_A, img_res)
    img_B = misc.imread(pathB, mode='RGB').astype(np.float)
    img_B = misc.imresize(img_B, img_res)
    return img_A, img_B



def get_local_test_data(data_dir, img_res=(256, 256)):
    assert os.path.exists(data_dir), "local image path doesnt exist"
    imgs = []
    for p in getPaths(data_dir):
        img = read_and_resize(p, img_res)
        imgs.append(img)
    imgs = preprocess(np.array(imgs))
    return imgs



class DataLoader():
    def __init__(self, data_dir, dataset_name, img_res=(256, 256), test_only=False):
        self.img_res = img_res
        self.DATA = dataset_name
        self.data_dir = data_dir

        if not test_only:
            self.trainA_paths = getPaths(os.path.join(self.data_dir, "trainA")) # distorted
            self.trainB_paths = getPaths(os.path.join(self.data_dir, "trainB")) # enhanced
            if (len(self.trainA_paths)<len(self.trainB_paths)):
                self.trainB_paths = self.trainB_paths[:len(self.trainA_paths)]
            elif (len(self.trainA_paths)>len(self.trainB_paths)):
                self.trainA_paths = self.trainA_paths[:len(self.trainB_paths)]
            else: pass
            self.val_paths = getPaths(os.path.join(self.data_dir, "val"))
            self.num_train, self.num_val = len(self.trainA_paths), len(self.val_paths)
            print ("{0} training pairs\n".format(self.num_train))
        else:
            self.test_paths    = getPaths(os.path.join(self.data_dir, "test"))
            print ("{0} test images\n".format(len(self.test_paths)))


    def get_test_data(self, batch_size=1):
        idx = np.random.choice(np.arange(len(self.test_paths)), batch_size, replace=False)
        paths = self.test_paths[idx]
        imgs = []
        for p in paths:
            img = read_and_resize(p, self.img_res)
            imgs.append(img)
        imgs = preprocess(np.array(imgs))
        return imgs


    def load_val_data(self, batch_size=1):
        idx = np.random.choice(np.arange(self.num_val), batch_size, replace=False)
        pathsA = self.trainA_paths[idx]
        pathsB = self.trainB_paths[idx]

        imgs_A, imgs_B = [], []
        for idx in range(len(pathsB)):
            img_A, img_B = read_and_resize_pair(pathsA[idx], pathsB[idx], self.img_res)
            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = preprocess(np.array(imgs_A))
        imgs_B = preprocess(np.array(imgs_B))
        return imgs_A, imgs_B


    def load_batch(self, batch_size=1, data_augment=True):
        self.n_batches = self.num_train//batch_size

        for i in range(self.n_batches-1):
            batch_A = self.trainA_paths[i*batch_size:(i+1)*batch_size]
            batch_B = self.trainB_paths[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for idx in range(len(batch_A)): 
                img_A, img_B = read_and_resize_pair(batch_A[idx], batch_B[idx], self.img_res)

                if (data_augment):
                    img_A, img_B = augment(img_A, img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = preprocess(np.array(imgs_A))
            imgs_B = preprocess(np.array(imgs_B))

            yield imgs_A, imgs_B




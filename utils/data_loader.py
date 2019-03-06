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


def read_and_resize(pathA, pathB, img_res):
    img_A = misc.imread(pathA, mode='RGB').astype(np.float)  
    img_A = misc.imresize(img_A, img_res)
    img_B = misc.imread(pathB, mode='RGB').astype(np.float)
    img_B = misc.imresize(img_B, img_res)
    return img_A, img_B



class DataLoader():
    def __init__(self, data_dir, dataset_name, img_res=(256, 256)):
        self.img_res = img_res
        self.DATA = dataset_name
        self.data_dir = os.path.join(data_dir, dataset_name)

        self.trainA_paths = getPaths(os.path.join(self.data_dir, "trainA")) # underwater photos
        self.trainB_paths = getPaths(os.path.join(self.data_dir, "trainB")) # normal photos (ground truth)
        self.val_paths    = getPaths(os.path.join(self.data_dir, "val"))
        assert (len(self.trainA_paths)==len(self.trainB_paths)), "imbalanaced training pairs"
        self.num_train, self.num_val = len(self.trainA_paths), len(self.val_paths)
        print ("{0} training pairs\n".format(self.num_train))


    def load_val_data(self, batch_size=1):
        idx = np.random.choice(np.arange(self.num_val), batch_size, replace=False)
        pathsA = self.trainA_paths[idx]
        pathsB = self.trainB_paths[idx]

        imgs_A, imgs_B = [], []
        for idx in range(len(pathsB)):
            img_A, img_B = read_and_resize(pathsA[idx], pathsB[idx], self.img_res)
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
                img_A, img_B = read_and_resize(batch_A[idx], batch_B[idx], self.img_res)

                if (data_augment):
                    img_A, img_B = augment(img_A, img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = preprocess(np.array(imgs_A))
            imgs_B = preprocess(np.array(imgs_B))

            yield imgs_A, imgs_B




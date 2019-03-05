import cPickle as pickle
import tensorflow as tf
from scipy import misc
import numpy as np
import argparse
import random
import glob
import time
import sys
import cv2
import os

# my imports
sys.path.insert(0, 'ops/')
sys.path.insert(0, 'measures/')
sys.path.insert(0, 'nets/')
from tf_ops import *
import data_ops
import uiqm


LEARNING_RATE = 1e-4
UIQM_WEIGHT   = 0.0
LOSS_METHOD   = 'wgan'
BATCH_SIZE    = 16
NUM_LAYERS    = 8
L1_WEIGHT     = 100.0
IG_WEIGHT     = 0
NETWORK       = 'pix2pix'
AUGMENT       = True
EPOCHS        = 1
DATA          = 'underwater_imagenet'
LAB           = False

EXPERIMENT_DIR  = 'checkpoints/'+LOSS_METHOD+'_'+NETWORK+'_'+DATA+'/'
IMAGES_DIR      = EXPERIMENT_DIR+'samples/'

if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)
print ("Setup experimental directory at {0}".format(EXPERIMENT_DIR))


exp_info = dict()
exp_info['LOSS_METHOD']   = LOSS_METHOD
exp_info['BATCH_SIZE']    = BATCH_SIZE
exp_info['NUM_LAYERS']    = NUM_LAYERS
exp_info['NETWORK']       = NETWORK
exp_info['DATA']          = DATA
exp_pkl = open(EXPERIMENT_DIR+'info.pkl', 'wb')
data = pickle.dumps(exp_info)
exp_pkl.write(data)
exp_pkl.close()

print
print 'LOSS_METHOD:   ',LOSS_METHOD
print 'NUM_LAYERS:    ',NUM_LAYERS
print 'NETWORK:       ',NETWORK
print 'DATA:          ',DATA
print

if NETWORK == 'pix2pix': from pix2pix import *
if NETWORK == 'resnet': from resnet import *

# global step that is saved with a model to keep track of how many steps/epochs
global_step = tf.Variable(0, name='global_step', trainable=False)

# underwater image
image_u = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 256, 256, 3), name='image_u')

# correct image
image_r = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 256, 256, 3), name='image_r')

# generated corrected colors
layers     = netG8_encoder(image_u)
gen_image  = netG8_decoder(layers)

# send 'clean' water images to D
D_real = netD(image_r, LOSS_METHOD)

# send corrected underwater images to D
D_fake = netD(gen_image, LOSS_METHOD, reuse=True)

e = 1e-12
if LOSS_METHOD == 'least_squares':
    print 'Using least squares loss'
    errD_real = tf.nn.sigmoid(D_real)
    errD_fake = tf.nn.sigmoid(D_fake)
    errG = 0.5*(tf.reduce_mean(tf.square(errD_fake - 1)))
    errD = tf.reduce_mean(0.5*(tf.square(errD_real - 1)) + 0.5*(tf.square(errD_fake)))
if LOSS_METHOD == 'gan':
    print 'Using original GAN loss'
    errD_real = tf.nn.sigmoid(D_real)
    errD_fake = tf.nn.sigmoid(D_fake)
    errG = tf.reduce_mean(-tf.log(errD_fake + e))
    errD = tf.reduce_mean(-(tf.log(errD_real+e)+tf.log(1-errD_fake+e)))
if LOSS_METHOD == 'wgan':
    # cost functions
    errD = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
    errG = -tf.reduce_mean(D_fake)
    n_critic = 5


# gradient penalty
epsilon = tf.random_uniform([], 0.0, 1.0)
x_hat = image_r*epsilon + (1-epsilon)*gen_image
d_hat = netD(x_hat, LOSS_METHOD, reuse=True)
gradients = tf.gradients(d_hat, x_hat)[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = 10*tf.reduce_mean((slopes-1.0)**2)
errD += gradient_penalty


# tensorboard summaries
tf.summary.scalar('d_loss', tf.reduce_mean(errD))
tf.summary.scalar('g_loss', tf.reduce_mean(errG))


# get all trainable variables, and split by network G and network D
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]

G_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(errG, var_list=g_vars, global_step=global_step)
D_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(errD, var_list=d_vars)

saver = tf.train.Saver(max_to_keep=2)

init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
sess = tf.Session()
sess.run(init)

summary_writer = tf.summary.FileWriter(EXPERIMENT_DIR+'/logs/', graph=tf.get_default_graph())

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

merged_summary_op = tf.summary.merge_all()

data_dir = "/mnt/data2/color_correction_related/datasets/"
trainA_paths = data_ops.getPaths(data_dir+DATA+'/trainA/') # underwater photos
trainB_paths = data_ops.getPaths(data_dir+DATA+'/trainB/') # normal photos (ground truth)
test_paths   = data_ops.getPaths(data_dir+DATA+'/test/')
val_paths    = data_ops.getPaths(data_dir+DATA+'/val/')

print len(trainB_paths),'training pairs'
num_train = len(trainA_paths)
num_test  = len(test_paths)
num_val   = len(val_paths)

epoch_num = step/(num_train/BATCH_SIZE)



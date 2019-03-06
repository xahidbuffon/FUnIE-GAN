from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys

import numpy as np
import os
#export TF_CPP_MIN_LOG_LEVEL=2
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

###-----hyper-params-----
epochs = 1
batch_size = 16
sample_interval=50

###--------configure data-loader------------
from utils.data_loader import DataLoader
data_dir = "/mnt/data2/color_correction_related/datasets/"
dataset_name = "underwater_imagenet"
data_loader = DataLoader(data_dir, dataset_name)
if not os.path.exists("data/samples/"):
    os.makedirs("data/samples/")

###--------load model arch------------
from nets.funieGAN import FUNIE_GAN
funie_gan = FUNIE_GAN()

# Adversarial loss ground truths
valid = np.ones((batch_size,) + funie_gan.disc_patch)
fake = np.zeros((batch_size,) + funie_gan.disc_patch)

step = 0
TOTAL_STEP = epochs*(data_loader.num_train/batch_size)
while step < TOTAL_STEP:
    for _, (imgs_A, imgs_B) in enumerate(data_loader.load_batch(batch_size)):
        #  Train Discriminator
        fake_A = funie_gan.generator.predict(imgs_B)
        d_loss_real = funie_gan.discriminator.train_on_batch([imgs_A, imgs_B], valid)
        d_loss_fake = funie_gan.discriminator.train_on_batch([fake_A, imgs_B], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the generators
        g_loss = funie_gan.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])
        step += 1
        print ("Step {0}/{1}: lossD: {2}, lossG: {3}".format(step, TOTAL_STEP, d_loss[0], g_loss[0])) 

        # validate and save generated samples at regular intervals 
        if step % sample_interval == 0:
            imgs_A, imgs_B = data_loader.load_val_data(batch_size=3)
            fake_A = funie_gan.generator.predict(imgs_B)
            gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])
            gen_imgs = 0.5 * gen_imgs + 0.5 # Rescale to 0-1
            save_samples(gen_imgs, step)


def save_samples(gen_imgs, step, row=3, col=3):
    r, c = 3, 3
    titles = ['Condition', 'Generated', 'Original']
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(titles[i])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("data/samples/%d.png" % (step))
    plt.close()






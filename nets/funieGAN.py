from __future__ import print_function, division

from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

import numpy as np
import os
#export TF_CPP_MIN_LOG_LEVEL=2
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'



class FUNIE_GAN():
    def __init__(self, imrow=256, imcol=256, imchan=3):
        # Input shape
        self.img_rows, self.img_cols, self.channels = imrow, imcol, imchan
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf, self.df = 32, 32
        optimizer = Adam(0.0003, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'], loss_weights=[1, 100], optimizer=optimizer)



    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=3, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn: d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=3, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate: u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        print("Printing Generator model architecture")
        # Image input
        d0 = Input(shape=self.img_shape); print(d0)

        # Downsampling
        d1 = conv2d(d0, self.gf*1, f_size=5, bn=False); print(d1)
        #d2 = conv2d(d1, self.gf*2, f_size=5, bn=True) ;print(d2)
        d2 = conv2d(d1, self.gf*4, f_size=4, bn=True) ;print(d2)
        #d4 = conv2d(d3, self.gf*8, f_size=4, bn=True) ;print(d4)
        d3 = conv2d(d2, self.gf*8, f_size=4, bn=True) ;print(d3)
        d4 = conv2d(d3, self.gf*8, f_size=3, bn=True) ;print(d4)
        d5 = conv2d(d4, self.gf*8, f_size=3, bn=True) ;print(d5)

        print()
        # Upsampling
        u1 = deconv2d(d5, d4, self.gf*8) ;print(u1)
        u2 = deconv2d(u1, d3, self.gf*8) ;print(u2)
        u3 = deconv2d(u2, d2, self.gf*4) ;print(u3)
        u4 = deconv2d(u3, d1, self.gf*1) ;print(u4)

        #u5 = deconv2d(u4, d2, self.gf*2) ;print(u5)
        #u6 = deconv2d(u5, d1, self.gf) ;print(u5)

        u5 = UpSampling2D(size=2)(u4)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u5)
        print(output_img); print()

        return Model(d0, output_img)




    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=3, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn: d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        print("Printing Discriminator model architecture")
        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B]) ; print(combined_imgs)

        d1 = d_layer(combined_imgs, self.df, bn=False) ; print(d1)
        d2 = d_layer(d1, self.df*2) ; print(d2)
        d3 = d_layer(d2, self.df*4) ; print(d3)
        d4 = d_layer(d3, self.df*8) ; print(d4)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        print(validity); print()

        return Model([img_A, img_B], validity)



if __name__=="__main__":
    funie_gan = FUNIE_GAN()



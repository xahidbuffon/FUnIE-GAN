from __future__ import print_function, division
## python libs
import os
import numpy as np

## tf-Keras libs
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import vgg19



def VGG19_Content(dataset='imagenet'):
    # Load VGG, trained on imagenet data
    vgg = vgg19.VGG19(include_top=False, weights=dataset)
    vgg.trainable = False
    content_layers = ['block5_conv2']
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    return Model(vgg.input, content_outputs)



class FUNIE_GAN():
    def __init__(self, imrow=256, imcol=256, imchan=3, loss_meth='wgan'):
        ## input image shape
        self.img_rows, self.img_cols, self.channels = imrow, imcol, imchan
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        ## input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        ## conv 5_2 content from vgg19 network
        self.vgg_content = VGG19_Content()

        ## output shape of D (patchGAN)
        patch = int(self.img_rows/16)
        self.disc_patch = (patch, patch, 1)

        ## number of filters in the first layer of G and D
        self.gf, self.df = 32, 32
        optimizer = Adam(0.0003, 0.5)

        ## Build and compile the discriminator
        self.discriminator = self.FUNIE_discriminator()
        self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        ## Build the generator
        self.generator = self.FUNIE_generator()

        ## By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        ## For the combined model we will only train the generator
        self.discriminator.trainable = False

        ## Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])
        ## compute the comboned loss
        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', self.total_gen_loss], loss_weights=[1, 10], optimizer=optimizer)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)


    def total_gen_loss(self, org_content, gen_content):
        vgg_org_content = self.vgg_content(org_content)
        vgg_gen_content = self.vgg_content(gen_content)
        content_loss = K.mean(K.square(vgg_org_content - vgg_gen_content), axis=-1)
        mae_gen_loss = K.mean(K.abs(org_content-gen_content))
        gen_total_err = 0.7*mae_gen_loss+0.03*content_loss
        return gen_total_err


    def FUNIE_generator(self):
        """
           Inspired by the U-Net Generator with skip connections
           This is a much simpler architecture with fewer parameters
        """

        def conv2d(layer_input, filters, f_size=3, bn=True):
            ## for downsampling
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn: d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=3, dropout_rate=0):
            ## for upsampling
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate: u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        print("Printing Generator model architecture")
        ## input
        d0 = Input(shape=self.img_shape); print(d0)
        ## downsample
        d1 = conv2d(d0, self.gf*1, f_size=5, bn=False) ;print(d1)
        d2 = conv2d(d1, self.gf*4, f_size=4, bn=True)  ;print(d2)
        d3 = conv2d(d2, self.gf*8, f_size=4, bn=True)  ;print(d3)
        d4 = conv2d(d3, self.gf*8, f_size=3, bn=True)  ;print(d4)
        d5 = conv2d(d4, self.gf*8, f_size=3, bn=True)  ;print(d5); print();
        ## upsample
        u1 = deconv2d(d5, d4, self.gf*8) ;print(u1)
        u2 = deconv2d(u1, d3, self.gf*8) ;print(u2)
        u3 = deconv2d(u2, d2, self.gf*4) ;print(u3)
        u4 = deconv2d(u3, d1, self.gf*1) ;print(u4)
        u5 = UpSampling2D(size=2)(u4)    ;print(u5)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u5)
        print(output_img); print();

        return Model(d0, output_img)




    def FUNIE_discriminator(self):
        """
           Inspired by the pix2pix discriminator
        """
        def d_layer(layer_input, filters, strides_=2,f_size=3, bn=True):
            ## Discriminator layers
            d = Conv2D(filters, kernel_size=f_size, strides=strides_, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn: d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        print("Printing Discriminator model architecture")
        ## input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B]) ; print(combined_imgs)
        ## Discriminator layers
        d1 = d_layer(combined_imgs, self.df, bn=False) ; print(d1)
        d2 = d_layer(d1, self.df*2) ; print(d2)
        d3 = d_layer(d2, self.df*4) ; print(d3)
        d4 = d_layer(d3, self.df*8) ; print(d4)
        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        print(validity); print()

        return Model([img_A, img_B], validity)



if __name__=="__main__":
    # for testing the initialization
    funie_gan = FUNIE_GAN()



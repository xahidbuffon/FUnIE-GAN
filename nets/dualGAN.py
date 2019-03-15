from __future__ import print_function, division
## python libs
import os
import numpy as np

## tf-Keras libs
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D


class DUALGAN():
    def __init__(self, imrow=256, imcol=256, imchan=3):
        ## input image shape
        self.img_rows, self.img_cols, self.channels = imrow, imcol, imchan
        self.img_flat_dim = imrow*imcol*imchan
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)
        ## input images and their conditioning images
        imgs_A = Input(shape=(self.img_flat_dim,))
        imgs_B = Input(shape=(self.img_flat_dim,))

        # Build and compile the discriminators
        self.D_A = self.DUAL_discriminator()
        self.D_A.compile(loss=self.wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])
        self.D_B = self.DUAL_discriminator()
        self.D_B.compile(loss=self.wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])

        # Build the generators
        self.G_AB = self.DUAL_generator()
        self.G_BA = self.DUAL_generator()

        # For the combined model we will only train the generators
        self.D_A.trainable = False
        self.D_B.trainable = False

        # Generators translates the images to the opposite domain
        fake_B = self.G_AB(imgs_A)
        fake_A = self.G_BA(imgs_B)

        # The discriminators determines validity of translated images
        valid_A = self.D_A(fake_A)
        valid_B = self.D_B(fake_B)

        # Generators translate the images back to their original domain
        recov_A = self.G_BA(fake_B)
        recov_B = self.G_AB(fake_A)

        # The combined model  (stacked generators and discriminators)
        self.combined = Model(inputs=[imgs_A, imgs_B], outputs=[valid_A, valid_B, recov_A, recov_B])
        self.combined.compile(loss=[self.wasserstein_loss, self.wasserstein_loss, 'mae', 'mae'],
                            optimizer=optimizer, loss_weights=[1, 1, 100, 100])



    def DUAL_generator(self):
        X = Input(shape=(self.img_flat_dim,))
        model = Sequential()
        model.add(Dense(256, input_dim=self.img_flat_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.4))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.4))
        model.add(Dense(self.img_flat_dim, activation='tanh'))
        X_translated = model(X)
        return Model(X, X_translated)


    def DUAL_discriminator(self):
        img = Input(shape=(self.img_flat_dim,))
        model = Sequential()
        model.add(Dense(512, input_dim=self.img_flat_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1))
        validity = model(img)
        return Model(img, validity)


    def sample_generator_input(self, X, batch_size):
        # Sample random batch of images from X
        idx = np.random.randint(0, X.shape[0], batch_size)
        return X[idx]


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)



if __name__=="__main__":
    # for testing the initialization
    funie_gan = DUALGAN()


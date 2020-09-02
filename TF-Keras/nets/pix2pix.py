"""
# > Pix2Pix implementation, adopted as the original UGAN paper 
#    - Original repo: github.com/cameronfabbri/Underwater-Color-Correction
"""
import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.nn import relu, tanh


def lrelu(x, leak=0.2):
    return tf.maximum(leak*x, x)


def netG16_encoder(x):
    enc_conv1 = tcl.conv2d(x, 64, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv1')
    enc_conv1 = tcl.batch_norm(enc_conv1)
    enc_conv1 = lrelu(enc_conv1)
    print (enc_conv1)
    enc_conv2 = tcl.conv2d(enc_conv1, 128, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv2')
    enc_conv2 = tcl.batch_norm(enc_conv2)
    enc_conv2 = lrelu(enc_conv2)
    print (enc_conv2)
    enc_conv3 = tcl.conv2d(enc_conv2, 256, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv3')
    enc_conv3 = tcl.batch_norm(enc_conv3)
    enc_conv3 = lrelu(enc_conv3)
    print (enc_conv3)
    enc_conv4 = tcl.conv2d(enc_conv3, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv4')
    enc_conv4 = tcl.batch_norm(enc_conv4)
    enc_conv4 = lrelu(enc_conv4)
    print (enc_conv4)
    enc_conv5 = tcl.conv2d(enc_conv4, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv5')
    enc_conv5 = tcl.batch_norm(enc_conv5)
    enc_conv5 = lrelu(enc_conv5)
    print (enc_conv5)
    enc_conv6 = tcl.conv2d(enc_conv5, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv6')
    enc_conv6 = tcl.batch_norm(enc_conv6)
    enc_conv6 = lrelu(enc_conv6)
    print (enc_conv6)
    enc_conv7 = tcl.conv2d(enc_conv6, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv7')
    enc_conv7 = tcl.batch_norm(enc_conv7)
    enc_conv7 = lrelu(enc_conv7)
    print (enc_conv7)
    enc_conv8 = tcl.conv2d(enc_conv7, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv8')
    enc_conv8 = tcl.batch_norm(enc_conv8)
    enc_conv8 = lrelu(enc_conv8)
    print (enc_conv8); print("\n")
    layers = [enc_conv1, enc_conv2, enc_conv3, enc_conv4, enc_conv5, enc_conv6, enc_conv7, enc_conv8]
    return layers


def netG16_decoder(layers, lab=False):
    enc_conv1, enc_conv2, enc_conv3, enc_conv4, enc_conv5, enc_conv6, enc_conv7, enc_conv8 = layers[0], layers[1], layers[2], layers[3], layers[4], layers[5], layers[6], layers[7]
    # decoder, no batch norm
    dec_conv1 = tcl.convolution2d_transpose(enc_conv8, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv1')
    dec_conv1 = relu(dec_conv1)
    dec_conv1 = tf.concat([dec_conv1, enc_conv7], axis=3)
    print (dec_conv1)
    dec_conv2 = tcl.convolution2d_transpose(dec_conv1, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv2')
    dec_conv2 = relu(dec_conv2)
    dec_conv2 = tf.concat([dec_conv2, enc_conv6], axis=3)
    print (dec_conv2)
    dec_conv3 = tcl.convolution2d_transpose(dec_conv2, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv3')
    dec_conv3 = relu(dec_conv3)
    dec_conv3 = tf.concat([dec_conv3, enc_conv5], axis=3)
    print (dec_conv3)
    dec_conv4 = tcl.convolution2d_transpose(dec_conv3, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv4')
    dec_conv4 = relu(dec_conv4)
    dec_conv4 = tf.concat([dec_conv4, enc_conv4], axis=3)
    print (dec_conv4)
    dec_conv5 = tcl.convolution2d_transpose(dec_conv4, 256, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv5')
    dec_conv5 = relu(dec_conv5)
    dec_conv5 = tf.concat([dec_conv5, enc_conv3], axis=3)
    print (dec_conv5)
    dec_conv6 = tcl.convolution2d_transpose(dec_conv5, 128, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv6')
    dec_conv6 = relu(dec_conv6)
    dec_conv6 = tf.concat([dec_conv6, enc_conv2], axis=3)
    print (dec_conv6)
    dec_conv7 = tcl.convolution2d_transpose(dec_conv6, 64, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv7')
    dec_conv7 = relu(dec_conv7)
    dec_conv7 = tf.concat([dec_conv7, enc_conv1], axis=3)
    print (dec_conv7)
    c = 2 if lab else 3
    dec_conv8 = tcl.convolution2d_transpose(dec_conv7, c, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv8')
    dec_conv8 = tanh(dec_conv8)
    print (dec_conv1)

    return dec_conv8


def netG(x):
    print("\nnetG\n"); print (x)
    layers     = netG16_encoder(x)
    gen_im  = netG16_decoder(layers)
    return gen_im


def netD(x, LOSS_METHOD, reuse=False):
    print("\nnetD\n"); print (x)
    sc = tf.get_variable_scope()
    with tf.variable_scope(sc, reuse=reuse):
        conv1 = tcl.conv2d(x, 64, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv1')
        if LOSS_METHOD != 'wgan': conv1 = tcl.batch_norm(conv1)
        conv1 = lrelu(conv1)
        print (conv1)
        conv2 = tcl.conv2d(conv1, 128, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv2')
        if LOSS_METHOD != 'wgan': conv2 = tcl.batch_norm(conv2)
        conv2 = lrelu(conv2)
        print (conv2)
        conv3 = tcl.conv2d(conv2, 256, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv3')
        if LOSS_METHOD != 'wgan': conv3 = tcl.batch_norm(conv3)
        conv3 = lrelu(conv3)
        print (conv3)
        conv4 = tcl.conv2d(conv3, 512, 4, 1, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv4')
        if LOSS_METHOD != 'wgan': conv4 = tcl.batch_norm(conv4)
        conv4 = lrelu(conv4)
        print (conv4)
        conv5 = tcl.conv2d(conv4, 1, 1, 1, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv5')
        if LOSS_METHOD != 'wgan': conv5 = tcl.batch_norm(conv5)
        print (conv5); print("\n")

        return conv5



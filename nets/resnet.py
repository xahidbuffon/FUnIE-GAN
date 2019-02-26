import tensorflow as tf
import tensorflow.contrib.layers as tcl
import sys

sys.path.insert(0, 'ops/')
from tf_ops import *

def resBlock(x, num):

   x = relu(x)

   conv1 = tcl.conv2d(x, 256, 3, 1, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_resconv1_'+str(num))
   print 'res_conv1:',conv1

   conv2 = tcl.conv2d(conv1, 256, 3, 1, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_resconv2_'+str(num))
   print 'res_conv2:',conv2
   
   output = tf.add(x,conv2)
   print 'res_out:',output
   return output


def netG(x):
      
   conv1 = tcl.conv2d(x, 64, 7, 1, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv1')
   print 'conv1:',conv1

   conv2 = tcl.conv2d(conv1, 128, 3, 2, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv2')
   print 'conv2:',conv2
   
   conv3 = tcl.conv2d(conv2, 256, 3, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv3')
   print 'conv3:',conv3
   print
   
   res1 = resBlock(conv3, 1)
   res2 = resBlock(res1, 2)
   res3 = resBlock(res2, 3)
   res4 = resBlock(res3, 4)
   res5 = resBlock(res4, 5)
   res6 = resBlock(res5, 6)
   res7 = resBlock(res6, 7)
   res8 = resBlock(res7, 8)
   res9 = resBlock(res8, 9)
   res10 = resBlock(res9, 10)
   res11 = resBlock(res10, 11)
   res12 = resBlock(res11, 12)
   res13 = resBlock(res12, 13)
   res14 = resBlock(res13, 14)
   res15 = resBlock(res14, 15)
   res16 = resBlock(res15, 16)
   res17 = resBlock(res16, 17)
   res18 = resBlock(res17, 18)
   res19 = resBlock(res18, 19)
   res20 = resBlock(res19, 20)
   res21 = resBlock(res20, 21)
   res22 = resBlock(res21, 22)
   res23 = resBlock(res22, 23)
   res24 = resBlock(res23, 24)
   res25 = resBlock(res24, 25)
   res26 = resBlock(res25, 26)
   res27 = resBlock(res26, 27)
   res28 = resBlock(res27, 28)
   res29 = resBlock(res28, 29)
   res30 = resBlock(res29, 30)

   conv4 = tcl.conv2d_transpose(res30, 128, 3, 2, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv4')
   print 'conv4:',conv4
   
   conv5 = tcl.conv2d_transpose(conv4, 64, 3, 2, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv5')
   print 'conv5:',conv5
   
   conv6 = tcl.conv2d(conv5, 3, 7, 1, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv6')
   conv6 = tanh(conv6)
   print 'conv6:',conv6

   return conv6

def netD(x, LOSS_METHOD, reuse=False):
   print
   print 'netD'
   
   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):

      conv1 = tcl.conv2d(x, 64, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv1')
      if LOSS_METHOD != 'wgan':
         print 'Using batch norm in D'
         conv1 = tcl.batch_norm(conv1)
      conv1 = lrelu(conv1)
      
      conv2 = tcl.conv2d(conv1, 128, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv2')
      if LOSS_METHOD != 'wgan': conv2 = tcl.batch_norm(conv2)
      conv2 = lrelu(conv1)
      
      conv3 = tcl.conv2d(conv2, 256, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv3')
      if LOSS_METHOD != 'wgan': conv3 = tcl.batch_norm(conv3)
      conv3 = lrelu(conv3)
      
      conv4 = tcl.conv2d(conv3, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv4')
      if LOSS_METHOD != 'wgan': conv4 = tcl.batch_norm(conv4)
      conv4 = lrelu(conv4)
      
      conv5 = tcl.conv2d(conv4, 1, 1, 1, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv5')
      if LOSS_METHOD != 'wgan': conv5 = tcl.batch_norm(conv5)

      print conv1
      print conv2
      print conv3
      print conv4
      print conv5
      return conv5


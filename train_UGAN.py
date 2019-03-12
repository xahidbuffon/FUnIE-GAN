import os
import cv2
import sys
import random
import numpy as np
from scipy import misc
import tensorflow as tf
import cPickle as pickle
#export TF_CPP_MIN_LOG_LEVEL=2
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

 
## default settins as mentioned in the original paper
EPOCHS = 50
AUGMENT = True
BATCH_SIZE = 8
NUM_LAYERS = 16
L1_WEIGHT = 100.0
VAL_INTERVAL = 100
LEARNING_RATE = 1e-4
## feel free to change the following to try different mdoels
LOSS_METHOD = 'gan'  # options: {'gan', 'least_squares', 'wgan'}
NETWORK = 'pix2pix'   # options: {'pix2pix', 'resnet'}
DATA = 'underwater_imagenet'
EXPERIMENT_DIR = 'checkpoints/UGAN/'+LOSS_METHOD+'_'+NETWORK+'_'+DATA+'/run2/'
SAMPLES_DIR = os.path.join(EXPERIMENT_DIR, 'samples/')

## setup experimental and sample directories
if not os.path.exists(SAMPLES_DIR): os.makedirs(SAMPLES_DIR)
if not os.path.exists(EXPERIMENT_DIR): os.makedirs(EXPERIMENT_DIR)
print ("Setup experimental and sample directories")

## save the hyper-params just in case
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

print("\n")
print("DATA: {0}".format(DATA))
print("NETWORK: {0}".format(NETWORK))
print("LOSS_METHOD: {0}".format(LOSS_METHOD))
print("\n")

## my imports
sys.path.insert(0, 'nets/')
if NETWORK == 'resnet': from resnet import *
elif NETWORK == 'pix2pix': from pix2pix import *
else: pass
from utils.data_loader import getPaths, read_and_resize, preprocess, augment

##------------- training module begins here-----------------------
global_step = tf.Variable(0, name='global_step', trainable=False)
# underwater image
image_u = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 256, 256, 3), name='image_u')
# correct image
image_r = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 256, 256, 3), name='image_r')
# generated corrected colors
gen_image  = netG(image_u)

# send 'clean' water images to D
D_real = netD(image_r, LOSS_METHOD)
# send corrected underwater images to D
D_fake = netD(gen_image, LOSS_METHOD, reuse=True)

e = 1e-12; n_critic = 1
if LOSS_METHOD == 'least_squares':
    print("Using least squares loss")
    errD_real = tf.nn.sigmoid(D_real)
    errD_fake = tf.nn.sigmoid(D_fake)
    errG = 0.5*(tf.reduce_mean(tf.square(errD_fake - 1)))
    errD = tf.reduce_mean(0.5*(tf.square(errD_real - 1)) + 0.5*(tf.square(errD_fake)))
elif LOSS_METHOD == 'gan':
    print("Using original GAN loss")
    errD_real = tf.nn.sigmoid(D_real)
    errD_fake = tf.nn.sigmoid(D_fake)
    errG = tf.reduce_mean(-tf.log(errD_fake + e))
    errD = tf.reduce_mean(-(tf.log(errD_real+e)+tf.log(1-errD_fake+e)))
elif LOSS_METHOD == 'wgan':
    print("Using original GAN loss")
    n_critic = 5
    # cost functions
    errG = -tf.reduce_mean(D_fake)
    errD = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)

    # gradient penalty
    epsilon = tf.random_uniform([], 0.0, 1.0)
    x_hat = image_r*epsilon + (1-epsilon)*gen_image
    d_hat = netD(x_hat, LOSS_METHOD, reuse=True)
    gradients = tf.gradients(d_hat, x_hat)[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = 10*tf.reduce_mean((slopes-1.0)**2)
    errD += gradient_penalty
else: pass

l1_loss = tf.reduce_mean(tf.abs(gen_image-image_r))
errG += L1_WEIGHT*l1_loss


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
trainA_paths = getPaths(data_dir+DATA+"/trainA/") # underwater photos
trainB_paths = getPaths(data_dir+DATA+"/trainB/") # normal photos (ground truth)
test_paths   = getPaths(data_dir+DATA+"/test/")
val_paths    = getPaths(data_dir+DATA+"/val/")
num_train, num_test, num_val = len(trainA_paths), len(test_paths), len(val_paths)
print ("{0} training pairs\n".format(len(trainB_paths)))


TOTAL_STEP = (EPOCHS*num_train/BATCH_SIZE)
while step < TOTAL_STEP:
    # pick random images every time for D
    for itr in range(n_critic):
        idx = np.random.choice(np.arange(num_train), BATCH_SIZE, replace=False)
        batchA_paths = trainA_paths[idx]
        batchB_paths = trainB_paths[idx]
        batchA_images = np.empty((BATCH_SIZE, 256, 256, 3), dtype=np.float32)
        batchB_images = np.empty((BATCH_SIZE, 256, 256, 3), dtype=np.float32)
        
        for i,(a,b) in enumerate(zip(batchA_paths, batchB_paths)):
            a_img = misc.imread(a)
            b_img = misc.imread(b)

            # Data augmentation here - each has 50% chance
            if AUGMENT: 
                a_img, b_img = augment(a_img, b_img)
            batchA_images[i, ...] = preprocess(a_img)
            batchB_images[i, ...] = preprocess(b_img)
            
        sess.run(D_train_op, feed_dict={image_u:batchA_images, image_r:batchB_images})

    # also get new batch for G
    idx = np.random.choice(np.arange(num_train), BATCH_SIZE, replace=False)
    batchA_paths = trainA_paths[idx]
    batchB_paths = trainB_paths[idx]
    batchA_images = np.empty((BATCH_SIZE, 256, 256, 3), dtype=np.float32)
    batchB_images = np.empty((BATCH_SIZE, 256, 256, 3), dtype=np.float32)

    for i,(a,b) in enumerate(zip(batchA_paths, batchB_paths)):
        a_img = misc.imread(a)
        b_img = misc.imread(b)
        # Data augmentation here - each has 50% chance
        if AUGMENT: 
            a_img, b_img = augment(a_img, b_img)
        batchA_images[i, ...] = preprocess(a_img)
        batchB_images[i, ...] = preprocess(b_img)

    sess.run(G_train_op, feed_dict={image_u:batchA_images, image_r:batchB_images})
    D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op], feed_dict={image_u:batchA_images, image_r:batchB_images})
    summary_writer.add_summary(summary, step)
    step += 1
    print ("Step {0}/{1}: lossD: {2}, lossG: {3}".format(step, TOTAL_STEP, D_loss, G_loss)) 

    ##------------- validation and saving checkpoints -----------------------
    if (step%VAL_INTERVAL==0):
        print ("Saving model")
        saver.save(sess, EXPERIMENT_DIR+'checkpoint-'+str(step))
        saver.export_meta_graph(EXPERIMENT_DIR+"checkpoint-"+str(step)+".meta")

        idx = np.random.choice(np.arange(num_val), BATCH_SIZE, replace=False)
        batch_paths  = val_paths[idx]
        batch_images = np.empty((BATCH_SIZE, 256, 256, 3), dtype=np.float32)

        print ("Testing on validation data")
        for i,a in enumerate(batch_paths):
            a_img = misc.imread(a).astype("float32")
            batch_images[i, ...] = preprocess(misc.imresize(a_img, (256, 256, 3)))

        gen_images = np.asarray(sess.run(gen_image, feed_dict={image_u:batch_images}))
        for i, (gen, real) in enumerate(zip(gen_images, batch_images)):
            misc.imsave(os.path.join(SAMPLES_DIR, str(step)+ "_real.png"), real)
            misc.imsave(os.path.join(SAMPLES_DIR, str(step) + "_gen.png"), gen)
            if(i>=5): break




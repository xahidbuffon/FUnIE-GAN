"""
# > Script for training FUnIE-GAN on unpaired data (using cycle consistency) 
#    - Training pipeline is similar to that of a Cycle-GAN (different G and D)
#    - Paper: https://arxiv.org/pdf/1903.09766.pdf
# > Notes and Usage:
#    - set data_dir and other hyper-params
#    - python train_funieGAN_up.py
# > Maintainer: https://github.com/xahidbuffon
"""
## python libs
import os
import numpy as np

## local libs
from utils.data_utils import DataLoader
from nets.funieGAN_up import FUNIE_GAN_UP
from utils.plot_utils import save_val_samples_unpaired

## configure data-loader
data_dir = "/mnt/data1/color_correction_related/datasets/EUVP/"
dataset_name = "Unpaired"
data_loader = DataLoader(os.path.join(data_dir, dataset_name), dataset_name)

## create dir for log and (sampled) validation data
samples_dir = os.path.join("data/samples/funieGAN_up/", dataset_name)
checkpoint_dir = os.path.join("checkpoints/funieGAN_up/", dataset_name)
if not os.path.exists(samples_dir): os.makedirs(samples_dir)
if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

## hyper-params
num_epoch = 200
batch_size = 4
val_interval = 2000
N_val_samples = 2
save_model_interval = data_loader.num_train//batch_size
num_step = num_epoch*save_model_interval

## load model arch
funie_gan = FUNIE_GAN_UP()

## ground-truths for adversarial loss
valid = np.ones((batch_size,) + funie_gan.disc_patch)
fake = np.zeros((batch_size,) + funie_gan.disc_patch)

step = 0
all_D_losses = []; all_G_losses = []
while (step <= num_step):
    for _, (imgs_distorted, imgs_good) in enumerate(data_loader.load_batch(batch_size)):
        #  train the discriminator (domain A)
        fake_A = funie_gan.g_BA.predict(imgs_distorted)
        dA_loss_real = funie_gan.d_A.train_on_batch(imgs_good, valid)
        dA_loss_fake = funie_gan.d_A.train_on_batch(fake_A, fake)
        dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)
        #  train the discriminator (domain B)
        fake_B = funie_gan.g_AB.predict(imgs_good)
        dB_loss_real = funie_gan.d_B.train_on_batch(imgs_distorted, valid)
        dB_loss_fake = funie_gan.d_B.train_on_batch(fake_B, fake)
        dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
        d_loss = 0.5 * np.add(dA_loss, dB_loss)
        ## train the generator
        g_loss = funie_gan.combined.train_on_batch([imgs_good, imgs_distorted], 
                                   [valid, valid, imgs_good, imgs_distorted, imgs_good, imgs_distorted])
        ## increment step, save losses, and print them 
        step += 1; all_D_losses.append(d_loss[0]);  all_G_losses.append(g_loss[0]); 
        if (step<=1 or step%50==0):
            print ("Step {0}/{1}: lossD: {2}, lossG: {3}".format(step, num_step, d_loss[0], g_loss[0])) 
        ## validate and save generated samples at regular intervals 
        if (step % val_interval==0):
            imgs_good, imgs_distorted = data_loader.load_val_data(batch_size=N_val_samples)
            # Translate images to the other domain
            fake_A = funie_gan.g_BA.predict(imgs_distorted)
            fake_B = funie_gan.g_AB.predict(imgs_good)
            # Translate back to original domain
            reconstr_A = funie_gan.g_BA.predict(fake_B)
            reconstr_B = funie_gan.g_AB.predict(fake_A)
            gen_imgs = np.concatenate([imgs_good, fake_B, reconstr_A, imgs_distorted, fake_A, reconstr_B])
            gen_imgs = 0.5 * gen_imgs + 0.5 # Rescale to 0-1
            save_val_samples_unpaired(samples_dir, gen_imgs, step, N_samples=N_val_samples)

        if (step % save_model_interval==0):
            ## save model and weights
            model_name = os.path.join(checkpoint_dir, ("model_%d" %step))
            with open(model_name+"_.json", "w") as json_file:
                json_file.write(funie_gan.g_BA.to_json())
            funie_gan.g_BA.save_weights(model_name+"_.h5")
            print("\nSaved trained model in {0}\n".format(checkpoint_dir))
        # sanity
        if (step>=num_step): break
         


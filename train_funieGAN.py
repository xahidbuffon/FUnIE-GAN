
import os
import numpy as np

## local libs
from nets.funieGAN import FUNIE_GAN
from utils.data_loader import DataLoader
from utils.plot_utils import save_val_samples_funieGAN

## configure data-loader
data_dir = "/mnt/data2/color_correction_related/datasets/"
dataset_name = "underwater_imagenet"
data_loader = DataLoader(data_dir, dataset_name)

## create dir for log and (sampled) validation data
samples_dir = "data/samples/"
checkpoint_dir = "checkpoints/funieGAN/"
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

## hyper-params
num_epoch = 1
batch_size = 16
val_interval = 5
save_model_interval = 20 #data_loader.num_train//batch_size
num_step = num_epoch*save_model_interval

## load model arch
funie_gan = FUNIE_GAN()

## ground-truths for adversarial loss
valid = np.ones((batch_size,) + funie_gan.disc_patch)
fake = np.zeros((batch_size,) + funie_gan.disc_patch)

step = 0
while (step <= num_step):
    for _, (imgs_A, imgs_B) in enumerate(data_loader.load_batch(batch_size)):
        ##  train the discriminator
        fake_A = funie_gan.generator.predict(imgs_B)
        d_loss_real = funie_gan.discriminator.train_on_batch([imgs_A, imgs_B], valid)
        d_loss_fake = funie_gan.discriminator.train_on_batch([fake_A, imgs_B], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        ## train the generator
        g_loss = funie_gan.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])
        step += 1
        print ("Step {0}/{1}: lossD: {2}, lossG: {3}".format(step, num_step, d_loss[0], g_loss[0])) 

        ## validate and save generated samples at regular intervals 
        if (step % val_interval==0):
            imgs_A, imgs_B = data_loader.load_val_data(batch_size=3)
            fake_A = funie_gan.generator.predict(imgs_B)
            gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])
            gen_imgs = 0.5 * gen_imgs + 0.5 # Rescale to 0-1
            save_val_samples_funieGAN(samples_dir, gen_imgs, step)

        if (step % save_model_interval==0):
            ## save model and weights
            model_name = checkpoint_dir+("model_%d" %step)
            with open(model_name+"_.json", "w") as json_file:
                json_file.write(funie_gan.generator.to_json())
            funie_gan.generator.save_weights(model_name+"_.h5")
            print("\nSaved trained model in {0}\n".format(checkpoint_dir))








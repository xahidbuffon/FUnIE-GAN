## python libs
import os
import numpy as np

## local libs
from utils.data_loader import DataLoader
from nets.funieGAN_unpaired import FUNIE_GAN_UP
from utils.plot_utils import save_val_samples_funieGAN_UP

## configure data-loader
data_dir = "/mnt/data2/color_correction_related/datasets/"
dataset_name = "underwater_imagenet"
data_loader = DataLoader(data_dir, dataset_name)

## create dir for log and (sampled) validation data
samples_dir = os.path.join("data/samples/funieGAN_gp/", dataset_name)
checkpoint_dir = os.path.join("checkpoints/funieGAN_gp/", dataset_name)
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

## hyper-params
num_epoch = 50
batch_size = 8
val_interval = 100
N_val_samples = 1
save_model_interval = data_loader.num_train//batch_size
num_step = 10#num_epoch*save_model_interval

## load model arch
funie_gan = FUNIE_GAN_UP()

## ground-truths for adversarial loss
valid = np.ones((batch_size,) + funie_gan.disc_patch)
fake = np.zeros((batch_size,) + funie_gan.disc_patch)

step = 0
all_D_losses = []; all_G_losses = []
while (step <= num_step):
    for _, (imgs_A, imgs_B) in enumerate(data_loader.load_batch(batch_size)):
        ##  train the discriminator (both opposite domains)
        fake_A = funie_gan.g_BA.predict(imgs_B)
        dA_loss_real = funie_gan.d_A.train_on_batch(imgs_A, valid)
        dA_loss_fake = funie_gan.d_A.train_on_batch(fake_A, fake)
        dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

        fake_B = funie_gan.g_AB.predict(imgs_A)
        dB_loss_real = funie_gan.d_B.train_on_batch(imgs_B, valid)
        dB_loss_fake = funie_gan.d_B.train_on_batch(fake_B, fake)
        dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
        d_loss = 0.5 * np.add(dA_loss, dB_loss)

        ## train the generator
        g_loss = funie_gan.combined.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B, imgs_A, imgs_B])

        ## increment step, save losses, and print them 
        step += 1; all_D_losses.append(d_loss[0]);  all_G_losses.append(g_loss[0]); 
        print ("Step {0}/{1}: lossD: {2}, lossG: {3}".format(step, num_step, d_loss[0], g_loss[0])) 
        # more: adv:np.mean(g_loss[1:3]), recon:np.mean(g_loss[3:5]), id:np.mean(g_loss[5:6])


        ## validate and save generated samples at regular intervals 
        if (step % val_interval==0):
            imgs_A, imgs_B = data_loader.load_val_data(batch_size=N_val_samples)
            # Translate images to the other domain
            fake_A = funie_gan.g_BA.predict(imgs_B)
            fake_B = funie_gan.g_AB.predict(imgs_A)
            # Translate back to original domain
            reconstr_A = funie_gan.g_BA.predict(fake_B)
            reconstr_B = funie_gan.g_AB.predict(fake_A)
            gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])
            gen_imgs = 0.5 * gen_imgs + 0.5 # Rescale to 0-1
            save_val_samples_funieGAN_UP(samples_dir, gen_imgs, step, N_samples=N_val_samples)

        if (step % save_model_interval==0):
            ## save model and weights
            model_name = os.path.join(checkpoint_dir, ("model_%d" %step))
            with open(model_name+"_.json", "w") as json_file:
                json_file.write(funie_gan.g_BA.to_json())
            funie_gan.g_BA.save_weights(model_name+"_.h5")
            print("\nSaved trained model in {0}\n".format(checkpoint_dir))

        if (step>=num_step): break
         


## for visualization
viz = True
if viz:
    from utils.plot_utils import viz_gen_and_dis_losses
    viz_gen_and_dis_losses(all_D_losses, all_G_losses,checkpoint_dir)









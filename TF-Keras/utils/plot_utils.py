"""
# > Various modules for plotting samples
"""
import os
import matplotlib.pyplot as plt

def save_val_samples_funieGAN(samples_dir, gen_imgs, step, N_samples=3, N_ims=3):
    row=N_samples; col=N_ims;
    titles = ['Input', 'Generated', 'Original']
    fig, axs = plt.subplots(row, col)
    cnt = 0
    for j in range(col):
        for i in range(row): 
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(titles[j])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(os.path.join(samples_dir, ("%d.png" %step)))
    plt.close()


def save_val_samples_unpaired(samples_dir, gen_imgs, step, N_samples=1, N_ims=6):
    row=2*N_samples; col=N_ims//2;
    titles = ['Original','Translated','Reconstructed']
    fig, axs = plt.subplots(row, col)
    cnt = 0
    for i in range(row):
        for j in range(col): 
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(titles[j])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(os.path.join(samples_dir, ("_%d.png" %step)))
    plt.close()


def save_test_samples_funieGAN(samples_dir, gen_imgs, step=0):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(gen_imgs[0])
    axs[0].set_title("Input")
    axs[0].axis('off')
    axs[1].imshow(gen_imgs[1])
    axs[1].set_title("Generated")
    axs[1].axis('off')
    fig.savefig(os.path.join(samples_dir,("_test_%d.png" %step)))
    plt.close()


def viz_gen_and_dis_losses(all_D_losses, all_G_losses, save_dir=None):
    plt.plot(all_D_losses, 'r')
    plt.plot(all_G_losses, 'g')
    plt.title('Model convergence'); plt.ylabel('Losses'); plt.xlabel('# of steps');
    plt.legend(['Discriminator network', 'Generator network'], loc='upper right')
    plt.show();
    if not save_dir:
        plt.savefig(os.path.join(save_dir, '_conv.png'))


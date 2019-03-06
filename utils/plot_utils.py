import matplotlib.pyplot as plt

def save_val_samples_funieGAN(samples_dir, gen_imgs, step, row=3, col=3):
    r, c = 3, 3
    titles = ['Condition', 'Generated', 'Original']
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(titles[i])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(samples_dir+("%d.png" %step))
    plt.close()

import matplotlib.pyplot as plt

def save_val_samples_funieGAN(samples_dir, gen_imgs, step, row=3, col=3):
    titles = ['Condition', 'Generated', 'Original']
    fig, axs = plt.subplots(row, col)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(titles[i])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(samples_dir+("%d.png" %step))
    plt.close()


def save_test_samples_funieGAN(samples_dir, gen_imgs, step=0):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(gen_imgs[0])
    axs[0].set_title("Input")
    axs[0].axis('off')
    axs[1].imshow(gen_imgs[1])
    axs[1].set_title("Generated")
    axs[1].axis('off')
    fig.savefig(samples_dir+("_test_%d.png" %step))
    plt.close()

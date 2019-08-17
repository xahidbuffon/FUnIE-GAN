### Resources
- Implementations of **[FUnIE-GAN](https://arxiv.org/abs/1903.09766)** for underwater image enhancement
- Simplified implementations of **UGAN** and its variants ([original repo](https://github.com/cameronfabbri/Underwater-Color-Correction))
- **Cycle-GAN** and other relevant modules 
- Modules for quantifying image quality base on **UIQM**, **SSIM**, and **PSNR**
- Implementation: TensorFlow >= 1.11.0, Keras >= 2.2, and Python 2.7
  
| Perceptual enhancement | Color and sharpness   | Hue and contrast   | 
|:--------------------|:--------------------|:--------------------|
| ![det-1a](/data/fig1a.jpg) | ![det-1b](/data/col.jpg) | ![det-1c](/data/con.jpg)     |

| Enhanced underwater imagery | Improved detection and pose estimation  | 
|:--------------------|:--------------------|
| ![det-enh](/data/gif1.gif) | ![det-gif](/data/gif2.gif)     |


### Pointers
- Paper: https://arxiv.org/pdf/1903.09766.pdf
- Datasets: http://irvlab.cs.umn.edu/resources/euvp-dataset
- A few model weights are provided in saved_models/ directory
- Bibliography entry for citation:
```
article{islam2019fast,
    title={Fast Underwater Image Enhancement for Improved Visual Perception},
    author={Islam, Md Jahidul and Xia, Youya and Sattar, Junaed},
    journal={arXiv preprint arXiv:1903.09766},
    year={2019}
}
```

### Underwater Image Enhancement: Other Relevant Resources 
#### 2019
| Paper  | Theme/contributions | Code   | Data |
|:------------------------|:---------------------|:---------------------|:---------------------|
| [FUnIE-GAN](https://arxiv.org/abs/1903.09766)  | Simple and fast cGAN-based model, loss function design, dataset formulation | [GitHub](https://github.com/xahidbuffon/funie-gan) | [EUVP dataset](http://irvlab.cs.umn.edu/resources/euvp-dataset) |
| [Multiscale Dense GAN](https://ieeexplore.ieee.org/abstract/document/8730425)  | Residual multiscale dense block as generator | | |
| [Fusion GAN](https://arxiv.org/abs/1906.06819)  | FGAN-based model, loss function formulation |  | [U45](https://github.com/IPNUISTlegal/underwater-test-dataset-U45-) |
| [UDAE](https://arxiv.org/abs/1905.09000) | U-Net Denoising Autoencoder |  | | 
| [VDSR](https://ieeexplore.ieee.org/abstract/document/8763933) | ResNet-based model, loss function design  |  | | 
| [JWCDN](https://arxiv.org/abs/1907.05595) | Joint wavelength compensation and dehazing  | [GitHub](https://github.com/mohitkumarahuja/Underwater-Image-Enhancement-by-Wavelength-Compensation-and-Dehazing) |  | 






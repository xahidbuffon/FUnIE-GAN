## Resources
- Implementations of [FUnIE-GAN](https://arxiv.org/abs/1903.09766) for underwater image enhancement
- Simplified implementations of UGAN and its variants ([original repo](https://github.com/cameronfabbri/Underwater-Color-Correction))
- Cycle-GAN and other relevant modules 
- Modules for quantifying image quality metrics: UIQM, SSIM, and PSNR
- Implementation: TensorFlow (>= 1.11.0) and Keras (>= 2.2) (Python 2.7)
  
| Perceptual enhancement | Color and sharpness   | Hue and contrast   | 
|:--------------------|:--------------------|:--------------------|
| ![det-1a](/data/fig1a.jpg) | ![det-1b](/data/col.jpg) | ![det-1c](/data/con.jpg)     |

| Perceptual enhancement of underwater imagery | Performances improvement for visual detection  | 
|:--------------------|:--------------------|
| ![det-enh](/data/gif1.gif) | ![det-gif](/data/gif2.gif)     |




## Pointers
- Paper: https://arxiv.org/pdf/1903.09766.pdf
- Datasets: http://irvlab.cs.umn.edu/resources/euvp-dataset
- A few model weights are provided in saved_models/



#### Bib
```
article{islam2019fast,
    title={Fast Underwater Image Enhancement for Improved Visual Perception},
    author={Islam, Md Jahidul and Xia, Youya and Sattar, Junaed},
    journal={arXiv preprint arXiv:1903.09766},
    year={2019}
}
```

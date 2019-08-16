## Resources
- Implementations of FUnIE-GAN, our recent work on underwater image enhancement
- Simplified implementations of our previous work, UGAN and its variants ([original repo](https://github.com/cameronfabbri/Underwater-Color-Correction))
- Cycle-GAN and other relevant modules 
- Modules for quantifying image quality metrics: UIQM, SSIM, and PSNR
- Implementation: TensorFlow 1.11.0 (Python 2.7)
  
| Perceptual enhancement of underwater imagery | Performances improvement for visual detection  | 
|:--------------------|:--------------------|
| ![det-enh](/data/fig1a.jpg) | ![det-gif](/data/gif2.gif)     |

| Perceptual enhancement | Color and sharpness   | Hue and contrast   | 
|:--------------------|:--------------------|:--------------------|
| ![det-7](/data/fig1a.jpg) | ![det-7](/data/col.jpg) | ![det-7](/data/con.jpg)     |


## Pointers
- Paper: https://arxiv.org/pdf/1903.09766.pdf
- Datasets: http://irvlab.cs.umn.edu/resources/euvp-dataset
- A few model weights are provided in saved_models/








#### Citation
- Feel free to cite the paper you find the code/data useful:
```
article{islam2019fast,
    title={Fast Underwater Image Enhancement for Improved Visual Perception},
    author={Islam, Md Jahidul and Xia, Youya and Sattar, Junaed},
    journal={arXiv preprint arXiv:1903.09766},
    year={2019}
}```

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

	#### Usage
	- which scripts to run
	- which models to choose

	#### Constraints and Challenges
	- Trade-off between performance and running time
		- Constraints: running time >= 10 FPS on Jetson-TX2 and model size <= 17MB 
	- Issues with unpaired training (discussed in the paper: section 4.6)
		- Inconsistent coloring, inaccurate modeling of textures 

### Underwater Image Enhancement: Recent Research and Resources 
#### 2019
| Paper  | Theme | Code   | Data |
|:------------------------|:---------------------|:---------------------|:---------------------|
| [FUnIE-GAN](https://arxiv.org/abs/1903.09766)  | Fast cGAN-based model, loss function and dataset formulation | [GitHub](https://github.com/xahidbuffon/funie-gan) | [EUVP dataset](http://irvlab.cs.umn.edu/resources/euvp-dataset) |
| [Multiscale Dense-GAN](https://ieeexplore.ieee.org/abstract/document/8730425)  | Residual multiscale dense block as generator | | |
| [Fusion-GAN](https://arxiv.org/abs/1906.06819)  | FGAN-based model, loss function formulation |  | [U45](https://github.com/IPNUISTlegal/underwater-test-dataset-U45-) |
| [UDAE](https://arxiv.org/abs/1905.09000) | U-Net denoising autoencoder |  | | 
| [VDSR](https://ieeexplore.ieee.org/abstract/document/8763933) | ResNet-based model, loss function formulation  |  | | 
| [JWCDN](https://arxiv.org/abs/1907.05595) | Joint wavelength compensation and dehazing  | [GitHub](https://github.com/mohitkumarahuja/Underwater-Image-Enhancement-by-Wavelength-Compensation-and-Dehazing) |  | 
| [AWMD-Cycle-GAN](https://www.mdpi.com/2077-1312/7/7/200) | Adaptive weighting for multi-discriminator training  | | | 
| [WAug Encoder-Decoder](http://openaccess.thecvf.com/content_CVPRW_2019/html/AAMVEM/Jamadandi_Exemplar-based_Underwater_Image_Enhancement_Augmented_by_Wavelet_Corrected_Transforms_CVPRW_2019_paper.html) |  Encoder-decoder module with wavelet pooling and unpooling | [GitHub](https://github.com/AdarshMJ/Underwater-Image-Enhancement-via-Style-Transfer) | |

#### 2017-18
| Paper  | Theme | Code   | Data |
|:------------------------|:---------------------|:---------------------|:---------------------|
| [UGAN](https://ieeexplore.ieee.org/document/8460552)  | Several GAN-based models, dataset formulation | [GitHub](https://github.com/cameronfabbri/Underwater-Color-Correction) | [Uw-imagenet](http://irvlab.cs.umn.edu/resources/) |
| [Underwater-GAN](https://link.springer.com/chapter/10.1007/978-3-030-05792-3_7) | Loss function formulation, cGAN-based model | | |
| [LAB-MSR](https://www.sciencedirect.com/science/article/pii/S0925231217305246) | Multi-scale Retinex-based framework | | |
| [Water-GAN](https://ieeexplore.ieee.org/abstract/document/7995024) | Data generation from in-air image and depth pairings | [GitHub](https://github.com/kskin/WaterGAN) | [MHL, Field data](https://github.com/kskin/WaterGAN) |
| [UIE-Net](https://ieeexplore.ieee.org/abstract/document/8296508)| CNN-based model for color correction and haze removal | | |

#### Non-deep Models
- [Local Color Mapping Combined with Color Transfer](https://ieeexplore.ieee.org/abstract/document/8659313) ([code](https://github.com/rprotasiuk/underwater_enhancement))
- [Real-time Model-based Image Color Correction for Underwater Robots](https://arxiv.org/abs/1904.06437)
- [All-In-One Underwater Image Enhancement using Domain-Adversarial Learning](http://openaccess.thecvf.com/content_CVPRW_2019/papers/UG2+%20Prize%20Challenge/Uplavikar_All-in-One_Underwater_Image_Enhancement_Using_Domain-Adversarial_Learning_CVPRW_2019_paper.pdf) ([code](https://github.com/TAMU-VITA/All-In-One-Underwater-Image-Enhancement-using-Domain-Adversarial-Learning))
- [Difference Backtracking Deblurring Method for Underwater Images](https://link.springer.com/article/10.1007/s11042-019-7420-z)
- [Guided Trigonometric Bilateral Filter and Fast Automatic Color correction](https://ieeexplore.ieee.org/abstract/document/6738704)

#### Reviews, Metrics, and Benchmarks
- [Real-world Underwater Enhancement: Challenges, Benchmarks, and Solutions](https://arxiv.org/abs/1901.05320)
- [Human-Visual-System-Inspired Underwater Image Quality Measures](https://ieeexplore.ieee.org/abstract/document/7305804)
- [An Underwater Image Enhancement Benchmark Dataset and Beyond](https://arxiv.org/abs/1901.05495)
- [An Experimental-based Review of Image Enhancement and Image Restoration Methods](https://arxiv.org/abs/1907.03246)
- [Diving Deeper into Underwater Image Enhancement: A Survey](https://arxiv.org/abs/1907.07863)







TensorFlow and PyTorch implementations of the paper *[Fast Underwater Image Enhancement for Improved Visual Perception (RA-L 2020)](https://ieeexplore.ieee.org/document/9001231)* and other GAN-based models.

![funie-fig](/data/funie.jpg)

### Resources
- Training pipelines for **[FUnIE-GAN](https://ieeexplore.ieee.org/document/9001231)** and **[UGAN](https://ieeexplore.ieee.org/document/8460552)** on [TensorFlow (Keras)](/TF-Keras/) and [PyTorch](/PyTorch/) 
- Modules for image quality analysis based on **UIQM**, **SSIM**, and **PSNR** ([see Evaluation](/Evaluation/))

| Enhanced underwater imagery | Improved detection and pose estimation  | 
|:--------------------|:--------------------|
| ![det-enh](/data/gif1.gif) | ![det-gif](/data/gif2.gif)     |


### FUnIE-GAN Features
- Provides competitive performance for underwater image enhancement
- Offers real-time inference on single-board computers
	- 48+ FPS on Jetson AGX Xavier, 25+ FPS on Jetson TX2
	- 148+ FPS on Nvidia GTX 1080 
- Suitable for underwater robotic deployments for enhanced vision 


### FUnIE-GAN Pointers
- Paper: https://ieeexplore.ieee.org/document/9001231
- Preprint: https://arxiv.org/pdf/1903.09766.pdf
- Datasets: http://irvlab.cs.umn.edu/resources/euvp-dataset
- Bibliography entry for citation:
	```
	@article{islam2019fast,
	    title={Fast Underwater Image Enhancement for Improved Visual Perception},
	    author={Islam, Md Jahidul and Xia, Youya and Sattar, Junaed},
	    journal={IEEE Robotics and Automation Letters (RA-L)},
	    volume={5},
	    number={2},
	    pages={3227--3234},
	    year={2020},
	    publisher={IEEE}
	}
	```

### Underwater Image Enhancement: Recent Research and Resources 
#### 2019
| Paper  | Theme | Code   | Data |
|:------------------------|:---------------------|:---------------------|:---------------------|
| [Multiscale Dense-GAN](https://ieeexplore.ieee.org/abstract/document/8730425)  | Residual multiscale dense block as generator | | |
| [Fusion-GAN](https://arxiv.org/abs/1906.06819)  | FGAN-based model, loss function formulation |  | [U45](https://github.com/IPNUISTlegal/underwater-test-dataset-U45-) |
| [UDAE](https://arxiv.org/abs/1905.09000) | U-Net denoising autoencoder |  | | 
| [VDSR](https://ieeexplore.ieee.org/abstract/document/8763933) | ResNet-based model, loss function formulation  |  | | 
| [JWCDN](https://arxiv.org/abs/1907.05595) | Joint wavelength compensation and dehazing  |   | 
| [AWMD-Cycle-GAN](https://www.mdpi.com/2077-1312/7/7/200) | Adaptive weighting for multi-discriminator training  | | | 
| [WAug Encoder-Decoder](http://openaccess.thecvf.com/content_CVPRW_2019/html/AAMVEM/Jamadandi_Exemplar-based_Underwater_Image_Enhancement_Augmented_by_Wavelet_Corrected_Transforms_CVPRW_2019_paper.html) |  Encoder-decoder module with wavelet pooling and unpooling | [GitHub](https://github.com/AdarshMJ/Underwater-Image-Enhancement-via-Style-Transfer) | |
| [Water-Net](https://arxiv.org/abs/1901.05495) | Dataset and benchmark |[GitHub](https://github.com/Li-Chongyi/Water-Net_Code) | [UIEBD](https://li-chongyi.github.io/proj_benchmark.html) |


#### 2017-18
| Paper  | Theme | Code   | Data |
|:------------------------|:---------------------|:---------------------|:---------------------|
| [UGAN](https://ieeexplore.ieee.org/document/8460552)  | Several GAN-based models, dataset formulation | [GitHub](https://github.com/cameronfabbri/Underwater-Color-Correction) | [Uw-imagenet](http://irvlab.cs.umn.edu/resources/) |
| [Underwater-GAN](https://link.springer.com/chapter/10.1007/978-3-030-05792-3_7) | Loss function formulation, cGAN-based model | | |
| [LAB-MSR](https://www.sciencedirect.com/science/article/pii/S0925231217305246) | Multi-scale Retinex-based framework | | |
| [Water-GAN](https://ieeexplore.ieee.org/abstract/document/7995024) | Data generation from in-air image and depth pairings | [GitHub](https://github.com/kskin/WaterGAN) | [MHL, Field data](https://github.com/kskin/WaterGAN) |
| [UIE-Net](https://ieeexplore.ieee.org/abstract/document/8296508)| CNN-based model for color correction and haze removal | | |

#### Non-deep Models
- [Sea-Thru](http://openaccess.thecvf.com/content_CVPR_2019/papers/Akkaynak_Sea-Thru_A_Method_for_Removing_Water_From_Underwater_Images_CVPR_2019_paper.pdf) ([project page](https://www.deryaakkaynak.com/sea-thru))
- [Haze-line-aware Color Restoration](https://arxiv.org/pdf/1811.01343.pdf) ([code](https://github.com/danaberman/underwater-hl))
- [Local Color Mapping Combined with Color Transfer](https://ieeexplore.ieee.org/abstract/document/8659313) ([code](https://github.com/rprotasiuk/underwater_enhancement))
- [Real-time Model-based Image Color Correction for Underwater Robots](https://arxiv.org/abs/1904.06437) ([code](https://github.com/dartmouthrobotics/underwater_color_enhance))
- [All-In-One Underwater Image Enhancement using Domain-Adversarial Learning](http://openaccess.thecvf.com/content_CVPRW_2019/papers/UG2+%20Prize%20Challenge/Uplavikar_All-in-One_Underwater_Image_Enhancement_Using_Domain-Adversarial_Learning_CVPRW_2019_paper.pdf) ([code](https://github.com/TAMU-VITA/All-In-One-Underwater-Image-Enhancement-using-Domain-Adversarial-Learning))
- [Difference Backtracking Deblurring Method for Underwater Images](https://link.springer.com/article/10.1007/s11042-019-7420-z)
- [Guided Trigonometric Bilateral Filter and Fast Automatic Color correction](https://ieeexplore.ieee.org/abstract/document/6738704)
- [Red-channel Underwater Image Restoration](https://www.sciencedirect.com/science/article/pii/S1047320314001874) ([code](https://github.com/agaldran/UnderWater))


#### Reviews, Metrics, and Benchmarks
- [Real-world Underwater Enhancement: Challenges, Benchmarks, and Solutions](https://arxiv.org/abs/1901.05320)
- [Human-Visual-System-Inspired Underwater Image Quality Measures](https://ieeexplore.ieee.org/abstract/document/7305804)
- [An Underwater Image Enhancement Benchmark Dataset and Beyond](https://arxiv.org/abs/1901.05495)
- [An Experimental-based Review of Image Enhancement and Restoration Methods](https://arxiv.org/abs/1907.03246) ([code](https://github.com/wangyanckxx/Single-Underwater-Image-Enhancement-and-Color-Restoration))
- [Diving Deeper into Underwater Image Enhancement: A Survey](https://arxiv.org/abs/1907.07863)
- [A Revised Underwater Image Formation Model](http://openaccess.thecvf.com/content_cvpr_2018/papers/Akkaynak_A_Revised_Underwater_CVPR_2018_paper.pdf)




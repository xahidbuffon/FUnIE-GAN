
### Resources
- Implementation of **FUnIE-GAN** (paired) 
- Simplified implementations of **UGAN / UGAN-P** ([original repo](https://github.com/cameronfabbri/Underwater-Color-Correction))
- Implementation: PyTorch 1.6 (Python 3.8)

### Usage
- Download the data, setup data-paths in the [config files](/PyTorch/configs/)
- Use the training scripts for paired training of FUnIE-GAN or UGAN/UGAN-P 
- Use the [test.py](test.py) script for evaluation
- A sample model is provided in [models](/PyTorch/models/) 
- *Note that the [TF-Keras implementation](TF-Keras) is the official one; use those weights to reproduce results in the paper. The evaluation data is [provided here (on eval_data folder)](https://drive.google.com/drive/folders/1ZEql33CajGfHHzPe1vFxUFCMcP0YbZb3?usp=sharing) for convenience.*

### Acknowledgements
- https://phillipi.github.io/pix2pix/
- https://github.com/eriklindernoren/PyTorch-GAN
- https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

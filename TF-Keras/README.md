### Resources
- Implementations of **[FUnIE-GAN](https://ieeexplore.ieee.org/document/9001231)** for underwater image enhancement
- Simplified implementations of **[UGAN](https://ieeexplore.ieee.org/document/8460552)** and its variants ([original repo](https://github.com/cameronfabbri/Underwater-Color-Correction))
- Implementation: TensorFlow >= 1.11.0 and Keras >= 2.2 (Python 2.7)

#### Usage
- Download the data, setup data-paths in the training-scripts
- Use paired training for FUnIE-GAN or UGAN, and unpaired training for FUnIE-GAN-up 
- Use the test-scripts for evaluating different models
	- A few test images: ../data/test/A (ground-truth: GTr_A), ../data/test/random (unpaired)
	- Output: ../data/output 
- A few saved models are provided in saved_models/

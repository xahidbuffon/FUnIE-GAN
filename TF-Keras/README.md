### FUnIE-GAN: Fast Underwater Image Enhancement using GANs  

### Resources
- Implementations of **[FUnIE-GAN](https://ieeexplore.ieee.org/document/9001231)** for underwater image enhancement
- Simplified implementations of **[UGAN](https://ieeexplore.ieee.org/document/8460552)** and its variants ([original repo](https://github.com/cameronfabbri/Underwater-Color-Correction))
- Implementation: TensorFlow >= 1.11.0 and Keras >= 2.2 (Python 2.7)

#### Usage
- Download the data, setup data-paths in the training-scripts
- Use paired training for FUnIE-GAN or UGAN, and unpaired training for FUnIE-GAN-up 
	- Sample checkpoints: checkpoints/model-name/dataset-name
	- Data samples: data/samples/model-name/dataset-name
- Use the test-scripts for evaluating different models
	- A few test images: data/test/A (ground-truth: GTr_A), data/test/random (unpaired)
	- Output: data/output 
- A few saved models are provided in saved_models/

#### Constraints and Challenges
- Issues with unpaired training (as discussed in the paper)
	- Inconsistent coloring, inaccurate modeling of sunlight
	- Often poor hue rectification (dominant blue/green hue) 
	- Hard to achieve training stability
- Much better enhancement performance can be obtained 
	- With denser models at the cost of speed
	- By exploiting optical waterbody properties as prior


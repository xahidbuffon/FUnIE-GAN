
import os
import time
import ntpath
import numpy as np
from scipy import misc
from keras.models import model_from_json


## local libs
from nets.cycleGAN import CycleGAN
from utils.data_utils import getPaths, read_and_resize, preprocess

# for testing arbitrary local data
data_dir = "data/test/A/"
from utils.data_utils import get_local_test_data
test_paths = getPaths(data_dir)
print ("{0} test images are loaded".format(len(test_paths)))


## create dir for log and (sampled) validation data
## create dir for log and (sampled) validation data
samples_dir = "data/test/C/"
if not os.path.exists(samples_dir): os.makedirs(samples_dir)

checkpoint_dir = "checkpoints/cycleGAN/EUVP/"
model_name_by_epoch = "model_3163_"
model_h5 = checkpoint_dir + model_name_by_epoch + ".h5"  
model_json = checkpoint_dir + model_name_by_epoch + ".json"
# sanity
assert (os.path.exists(model_h5) and os.path.exists(model_json))

## load model arch
cycle_gan= CycleGAN()

# load json and create model
with open(model_json, "r") as json_file:
    loaded_model_json = json_file.read()
times = []; s = time.time()
cycle_gan_generator = model_from_json(loaded_model_json)

# load weights into new model
cycle_gan_generator.load_weights(model_h5)
tot = time.time()-s
times.append(tot)
print("\nLoaded data and model")


for img_path in test_paths:
    img_name = ntpath.basename(img_path).split('.')[0]
    im = read_and_resize(img_path, (256, 256))
    im = preprocess(im)
    im = np.expand_dims(im, axis=0) # (1,256,256,3)

    s = time.time()
    gen = cycle_gan_generator.predict(im)
    tot = time.time()-s
    times.append(tot)

    misc.imsave(samples_dir+img_name+'_real.png', im[0])
    misc.imsave(samples_dir+img_name+'_gen.png', gen[0])
    
num_test = len(test_paths)
if (num_test==0):
    print ("\nFound no images for test")
else:
    print ("\nTotal images: {0}".format(num_test)) 
    Ttime = sum(times)
    print ("Time taken: {0} sec at {1} fps".format(Ttime, num_test/Ttime))
    print("\nSaved generated images in in {0}\n".format(samples_dir))










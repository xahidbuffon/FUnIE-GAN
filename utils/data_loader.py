import os
import numpy as np
from scipy import misc
# local
from data_ops import getPaths, preprocess, augment

class DataLoader():
    def __init__(self, data_dir, dataset_name, img_res=(256, 256)):
        self.img_res = img_res
        self.DATA = dataset_name
        self.data_dir = os.path.join(data_dir, dataset_name)

        self.trainA_paths = getPaths(os.path.join(self.data_dir, "trainA")) # underwater photos
        self.trainB_paths = getPaths(os.path.join(self.data_dir, "trainB")) # normal photos (ground truth)
        self.val_paths    = getPaths(os.path.join(self.data_dir, "val"))
        assert (len(self.trainA_paths)==len(self.trainB_paths)), "imbalanaced training pairs"
        self.num_train, self.num_val = len(self.trainA_paths), len(self.val_paths)
        print ("{0} training pairs\n".format(self.num_train))



    def load_batch(self, batch_size=1, AUGMENT=False, is_validating=False):
        if (is_validating):
            idx = np.random.choice(np.arange(self.num_val), batch_size, replace=False)
        else:
            idx = np.random.choice(np.arange(self.num_train), batch_size, replace=False)

        batchA_paths = self.trainA_paths[idx]
        batchB_paths = self.trainB_paths[idx]
        batchA_images = np.empty((batch_size, 256, 256, 3), dtype=np.float32)
        batchB_images = np.empty((batch_size, 256, 256, 3), dtype=np.float32)

        for i,(a,b) in enumerate(zip(batchA_paths, batchB_paths)):
            a_img = misc.imread(a)
            b_img = misc.imread(b)
            # Data augmentation with 0.5 proba
            if AUGMENT:  a_img, b_img = augment(a_img, b_img)
            batchA_images[i, ...] = preprocess(a_img)
            batchB_images[i, ...] = preprocess(b_img)

        return batchA_images, batchB_images



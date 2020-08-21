"""
 > Modules for processing training/validation data  
 > Maintainer: https://github.com/xahidbuffon
"""
import os
import glob
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

 
class GetTrainingPairs(Dataset):
    """ Common data pipeline to organize and generate
         training pairs for various datasets   
    """
    def __init__(self, root, dataset_name, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.filesA, self.filesB = self.get_file_paths(root, dataset_name)
        self.len = min(len(self.filesA), len(self.filesB))

    def __getitem__(self, index):
        img_A = Image.open(self.filesA[index % self.len])
        img_B = Image.open(self.filesB[index % self.len])
        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        return {"A": img_A, "B": img_B}

    def __len__(self):
        return self.len

    def get_file_paths(self, root, dataset_name):
        if dataset_name=='EUVP':
            filesA, filesB = [], []
            sub_dirs = ['underwater_imagenet', 'underwater_dark', 'underwater_scenes']
            for sd in sub_dirs:
                filesA += sorted(glob.glob(os.path.join(root, sd, 'trainA') + "/*.*"))
                filesB += sorted(glob.glob(os.path.join(root, sd, 'trainB') + "/*.*"))
        elif dataset_name=='UFO-120':
                filesA = sorted(glob.glob(os.path.join(root, 'lrd') + "/*.*"))
                filesB = sorted(glob.glob(os.path.join(root, 'hr') + "/*.*"))
        return filesA, filesB 



class GetValImage(Dataset):
    """ Common data pipeline to organize and generate
         vaditaion samples for various datasets   
    """
    def __init__(self, root, dataset_name, transforms_=None, sub_dir='validation'):
        self.transform = transforms.Compose(transforms_)
        self.files = self.get_file_paths(root, dataset_name)
        self.len = len(self.files)

    def __getitem__(self, index):
        img_val = Image.open(self.files[index % self.len])
        img_val = self.transform(img_val)
        return {"val": img_val}

    def __len__(self):
        return self.len

    def get_file_paths(self, root, dataset_name):
        if dataset_name=='EUVP':
            files = []
            sub_dirs = ['underwater_imagenet', 'underwater_dark', 'underwater_scenes']
            for sd in sub_dirs:
                files += sorted(glob.glob(os.path.join(root, sd, 'validation') + "/*.*"))
        elif dataset_name=='UFO-120':
            files = sorted(glob.glob(os.path.join(root, 'lrd') + "/*.*"))
        return files


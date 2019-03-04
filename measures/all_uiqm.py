'''

   Goes through all checkpoints and tests the UIQM

'''

import os
import glob
import fnmatch
import numpy as np
from tqdm import tqdm
import cPickle as pickle
from uiqm import getUIQM
import scipy.misc as misc

def getPaths(data_dir):
   pkl_paths = []
   pattern = '*.pkl'
   for d, s, fList in os.walk(data_dir):
      for filename in fList:
         if fnmatch.fnmatch(filename, pattern):
            fname_ = os.path.join(d,filename)
            pkl_paths.append(fname_)
   return np.asarray(pkl_paths)


if __name__ == '__main__':

   # search for all pkl files in checkpoints directory
   pkl_paths = getPaths('../checkpoints/')

   # open file to write results in (open and close to clear it)
   f = open('uiqm_results.txt','w')
   f.close()
   f = open('uiqm_results.txt','a')

   for p in pkl_paths:

      pkl_file = open(p, 'rb')
      a = pickle.load(pkl_file)
   
      avg_uiqm = []

      LEARNING_RATE = a['LEARNING_RATE']
      LOSS_METHOD   = a['LOSS_METHOD']
      UIQM_WEIGHT   = a['UIQM_WEIGHT']
      NUM_LAYERS    = a['NUM_LAYERS']
      BATCH_SIZE    = a['BATCH_SIZE']
      L1_WEIGHT     = a['L1_WEIGHT']
      IG_WEIGHT     = a['IG_WEIGHT']
      NETWORK       = a['NETWORK']
      AUGMENT       = a['AUGMENT']
      EPOCHS        = a['EPOCHS']
      DATA          = a['DATA']
      LAB           = a['LAB']
      
      EXPERIMENT_DIR  = '../checkpoints/LOSS_METHOD_'+LOSS_METHOD\
                        +'/NETWORK_'+NETWORK\
                        +'/UIQM_WEIGHT_'+str(UIQM_WEIGHT)\
                        +'/NUM_LAYERS_'+str(NUM_LAYERS)\
                        +'/L1_WEIGHT_'+str(L1_WEIGHT)\
                        +'/IG_WEIGHT_'+str(IG_WEIGHT)\
                        +'/AUGMENT_'+str(AUGMENT)\
                        +'/DATA_'+DATA\
                        +'/LAB_'+str(LAB)+'/'\

      IMAGES_DIR     = EXPERIMENT_DIR+'test_images/'

      gen_dir  = IMAGES_DIR+'gen/'
      real_dir = IMAGES_DIR+'real/'
      gen_images = glob.glob(gen_dir+'*.png')

      for img in tqdm(gen_images):
         img = misc.imread(img)
         uiqm_score = getUIQM(img)
         avg_uiqm.append(uiqm_score)

      avg_uiqm = np.mean(avg_uiqm)
      f.write(EXPERIMENT_DIR+', UIQM: '+str(avg_uiqm)+'\n')
   f.close()

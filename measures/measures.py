import scipy.misc as misc
import sys
import numpy as np

def mse(img1, img2):
   err = np.sum((img1.astype('float') - img2.astype('float')) ** 2)
   err /= float(img1.shape[0]*img1.shape[1])
   return err

if __name__ == '__main__':
   img1 = misc.imread(sys.argv[1])
   img2 = misc.imread(sys.argv[2])
   print 'mse:',mse(img1, img2)

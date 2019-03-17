
import os
import ntpath
import numpy as np
from scipy import misc


from utils.data_utils import getPaths
from utils.uqim_utils import getUIQM
from utils.ssm_psnr_utils import getSSIM, getPSNR


# Underwater Image Quality Measure (UIQM)
REAL_im_dir = 'data/test/A/'   # real im-dir with {filename.ext}
GEN_im_dir  = "data/test/C/" # gen  im-dir with {filename_gen.ext}
GTr_im_dir  = 'data/test/B/' # gen  im-dir with {filename_gen.ext}
REAL_paths, GEN_paths = getPaths(REAL_im_dir), getPaths(GEN_im_dir)


def measure_UIQMs(dir_name):
    paths = getPaths(dir_name)
    uqims = []
    for img_path in paths:
        im = misc.imread(img_path)
        uqims.append(getUIQM(im))
    return np.array(uqims)


def measure_SSIM_PSNRs(GT_dir, Gen_dir):
    """
      Assumes:
        * GT_dir contain ground-truths {filename.ext}
        * Gen_dir contain generated images {filename_gen.png}
        * Images are of same-size
    """
    GT_paths, Gen_paths = getPaths(GT_dir), getPaths(Gen_dir)

    ssims, psnrs = [], []
    for img_path in GT_paths:
        name_split = ntpath.basename(img_path).split('.')
        gen_path = os.path.join(Gen_dir, name_split[0]+'_gen.png') #+name_split[1])
        if (gen_path in Gen_paths):
            r_im = misc.imread(img_path)
            g_im = misc.imread(gen_path)
            assert (r_im.shape==g_im.shape), "The images should be of same-size"
            ssim = getSSIM(r_im, g_im)
            psnr = getPSNR(r_im, g_im)
            #print ("{0}, {1}: {2}".format(img_path,gen_path, ssim))
            #print ("{0}, {1}: {2}".format(img_path,gen_path, psnr))
            ssims.append(ssim)
            psnrs.append(psnr)
    return np.array(ssims), np.array(psnrs)



SSIM_measures, PSNR_measures = measure_SSIM_PSNRs(GTr_im_dir, GEN_im_dir)
print ("SSIM >> Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures)))
print ("PSNR >> Mean: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures)))

### compute and compare UIQMs
"""
g_truth = measure_UIQMs(GTr_im_dir)
print ("Mean: {0} std: {1}".format(np.mean(g_truth), np.std(g_truth)))
gen_uqims = measure_UIQMs(GEN_im_dir)
print ("Mean: {0} std: {1}".format(np.mean(gen_uqims), np.std(gen_uqims)))
real_uqims = measure_UIQMs(REAL_im_dir)
print ("Mean: {0} std: {1}".format(np.mean(real_uqims), np.std(real_uqims)))
"""










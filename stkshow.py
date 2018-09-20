from skimage.external.tifffile import TiffWriter
import subprocess
from shlex import split
from skimage import img_as_uint
import numpy as np

def stkshow(images, fname='/home/rfor10/Downloads/tmp-stk.tif'):
    with TiffWriter(fname, bigtiff=False, imagej=True) as t:
        if len(images.shape)>2:
            for i in range(images.shape[2]):
                t.save(images[:,:,i].astype('uint16'))
        else:
            t.save(images.astype('uint16'))
            
from skimage.exposure import rescale_intensity
def imadjust(img, low=1, high=99):
    return rescale_intensity(img, in_range=tuple(np.percentile(img, (low, high))))
#     java_cmd = ["java", "-Xmx5120m", "-jar", "/Users/robertf/ImageJ/ImageJ.app/Contents/Resources/Java/ij.jar"]
#     image_j_args = ["-ijpath", "/Users/robertf/ImageJ/", fname]
#     subprocess.Popen(java_cmd+image_j_args, shell=False,stdin=None,stdout=None,stderr=None,close_fds=True)
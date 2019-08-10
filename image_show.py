import numpy as np
from skimage.exposure import rescale_intensity

def imadjust(img, low=0.01, high = 0.90):
    img = img.copy()
    np.place(img, np.isnan(img), 0.00001)
    low, high = np.percentile(img, [low*100, high*100])
#     return rescale_intensity(img, out_range=(low, high))
    img = img-low
    img = img/high
    return img*2**8
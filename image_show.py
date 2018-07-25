import numpy as np

def imadjust(img, low=0.01, high = 0.90):
    low, high = np.percentile(img, [low*100, high*100])
    img = img-low
    img = img/high
    return img*2**8
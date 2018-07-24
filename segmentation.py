import numpy as np
from skimage import filters
from skimage import morphology
from skimage.feature import peak_local_max
from scipy import ndimage


def clipping(im, val):

    im_temp = im.copy()

    if val != 0:
        im_temp[im > val] = val

    return im_temp


def background(im, val):

    im_temp = im.copy()

    if val != 0:
        im_temp = im_temp - filters.gaussian(im_temp, val)

    return im_temp

def blur(im, val):

    if val != 0:

        if val <= 5:
            im = filters.gaussian(im, val)

        else:

            im = filters.gaussian(im, (val / 2))
            im = filters.gaussian(im, (val / 2))
            im = filters.gaussian(im, (val / 2))

    im -= np.min(im.flatten())
    im /= np.max(im.flatten())

    return im


def threshold(im, val):

    im_bin = im > val

    return im_bin


def object_filter(im_bin, val):

    im_bin = morphology.remove_small_objects(im_bin, val)

    return im_bin


def cell_centers(im, im_bin, val):

    d_mat = ndimage.distance_transform_edt(im_bin)
    d_mat /= np.max(d_mat.flatten())

    im_cent = (1 - val) * im + val * d_mat
    im_cent[np.logical_not(im_bin)] = 0

    return [im_cent, d_mat]

def expand_im(im, wsize):

    vinds = np.arange(im.shape[0], im.shape[0] - wsize + 1, -1) - 1
    hinds = np.arange(im.shape[1], im.shape[1] - wsize + 1, -1) - 1
    rev_inds = np.arange(wsize + 1, 0, -1)

    vinds = vinds.astype(int)
    hinds = hinds.astype(int)
    rev_inds = rev_inds.astype(int)

    conv_im_temp = np.vstack((im[rev_inds, :], im, im[vinds, :]))
    conv_im = np.hstack((conv_im_temp[:, rev_inds], conv_im_temp, conv_im_temp[:, hinds]))

    return conv_im

def im_probs(im, clf, wsize, stride):

    conv_im = expand_im(im, wsize)
    X_pred = classifyim.classify_im(conv_im, wsize, stride, im.shape[0], im.shape[1])

    y_prob = clf.predict_proba(X_pred)
    y_prob = y_prob[:, 1]

    return y_prob.reshape(im.shape)

def open_close(im, val):

    val = int(val)

    if 4 > val > 0:

        k = morphology.octagon(val, val)

        im = filters.gaussian(im, val)

        im = morphology.erosion(im, k)

        im = morphology.dilation(im, k)

    if 8 > val >= 4:
        k = morphology.octagon(val//2 + 1, val//2 + 1)

        im = filters.gaussian(im, val)
        im = filters.gaussian(im, val)

        im = morphology.erosion(im, k)
        im = morphology.erosion(im, k)

        im = morphology.dilation(im, k)
        im = morphology.dilation(im, k)

    if val >= 8:
        k = morphology.octagon(val // 4 + 1, val // 4 + 1)

        im = filters.gaussian(im, val)
        im = filters.gaussian(im, val)
        im = filters.gaussian(im, val)
        im = filters.gaussian(im, val)

        im = morphology.erosion(im, k)
        im = morphology.erosion(im, k)
        im = morphology.erosion(im, k)
        im = morphology.erosion(im, k)

        im = morphology.dilation(im, k)
        im = morphology.dilation(im, k)
        im = morphology.dilation(im, k)
        im = morphology.dilation(im, k)

    return im

def fg_markers(im_cent, im_bin, val, edges):

    local_maxi = peak_local_max(im_cent, indices=False, min_distance=int(val), labels=im_bin, exclude_border=int(edges))
    k = morphology.octagon(2, 2)

    local_maxi = morphology.dilation(local_maxi, selem=k)
    markers = ndimage.label(local_maxi)[0]
    markers[local_maxi] += 1

    return markers


def sobel_edges(im, val):

    if val != 0:
        if val <= 5:
            im = filters.gaussian(im, val)

        else:

            im = filters.gaussian(im, (val / 2))
            im = filters.gaussian(im, (val / 2))
            im = filters.gaussian(im, (val / 2))

    im = filters.sobel(im) + 1
    im /= np.max(im.flatten())

    return im


def watershed(markers, im_bin, im_edge, d_mat, val, edges):

    k = morphology.octagon(2, 2)

    im_bin = morphology.binary_dilation(im_bin, selem=k)
    im_bin = morphology.binary_dilation(im_bin, selem=k)
    im_bin = morphology.binary_dilation(im_bin, selem=k)

    markers_temp = markers + np.logical_not(im_bin)
    shed_im = (1 - val) * im_edge - val * d_mat

    labels = morphology.watershed(image=shed_im, markers=markers_temp)
    labels -= 1

    if edges == 1:
        edge_vec = np.hstack((labels[:, 0].flatten(), labels[:, -1].flatten(), labels[0, :].flatten(),
                              labels[-1, :].flatten()))
        edge_val = np.unique(edge_vec)
        for val in edge_val:
            if not val == 0:
                labels[labels == val] = 0

    return labels


def segment_image(im, params=None):
    """
    Segment images.
    
    Parameters
    ----------
    im : numpy.array
        Integer numpy array
    params : dict
        Dictionary of parameters values
        
    Returns
    -------
    im : numpy.array
        Integer array where connected components are assigned a unique integer value at 
        all indexes of contained by the connected component.
        
    params example
    ------ -------
    params = {
    'clip_limit' : 0.17, # 0-1
    'background_blur' : 180, #block size I think?
    'image_blur' : 4.0, #gaussian kernal
    'threshold' : 0.20, #0-1
    'smallest_object' : 40, #pixels
    'dist_intensity_ratio' : 0.75, #0-1 weight
    'separation_distance' : 8, #pixels
    'edge_filter_blur' : 2.0, #kernel width in pixels
    'watershed_ratio' : 0.15, #0-1 ratio of distance from edge vs bwgeodesic
    'remove_image_edges': 1, # If true set all edges to 0
             }
        
    Code adapted from Sam Cooper's Nuclitrack 2.0.
    https://github.com/samocooper/nuclitrack
    
    """
    if params is None:
        params = {
    'clip_limit' : 0.17, # 0-1
    'background_blur' : 180, #block size I think?
    'image_blur' : 4.0, #gaussian kernal
    'threshold' : 0.20, #0-1
    'smallest_object' : 40, #pixels
    'dist_intensity_ratio' : 0.75, #0-1 weight
    'separation_distance' : 8, #pixels
    'edge_filter_blur' : 2.0, #kernel width in pixels
    'watershed_ratio' : 0.15, #0-1 ratio of distance from edge vs bwgeodesic
    'remove_image_edges': 1, # If true set all edges to 0
             }

    #im = movie.comb_im(params[15:].astype(int), frame)

    image = clipping(im, params['clip_limit'])
    image2 = background(image, params['background_blur'])
    image3 = blur(image2, params['image_blur'])

#     if clf is not 0:
# 
#         image3 = im_probs(image3, clf, int(params[13]), int(params[14]))
# 
#         if params[12] > 0:
#             image3 = open_close(image3, params[12])

    im_bin = threshold(image3, params['threshold'])
    im_bin = object_filter(im_bin, params['smallest_object'])
    [cell_center, d_mat] = cell_centers(image3, im_bin, params['dist_intensity_ratio'])
    markers = fg_markers(cell_center, im_bin, params['separation_distance'], params['remove_image_edges'])
    im_edge = sobel_edges(image, params['edge_filter_blur'])

    im = watershed(markers, im_bin, im_edge, d_mat, params['watershed_ratio'], params['remove_image_edges'])

    if params['remove_image_edges'] == 1:

        vals = np.unique(np.concatenate((im[0, :].flatten(),
                                        im[:, 0].flatten(),
                                        im[-1, :].flatten(),
                                        im[:, -1].flatten())))
        for val in vals:
            if val > 0:
                im[im == val] = 0

    return im

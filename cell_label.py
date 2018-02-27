# Segment imports
import numpy
from scipy.ndimage import binary_fill_holes, binary_erosion, distance_transform_edt, binary_closing, binary_dilation
from scipy.ndimage import binary_dilation
from skimage.morphology import remove_small_objects, disk, erosion, watershed
from skimage import filters, img_as_uint
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skfmm import distance

# Cell Label imports
import numpy
import numpy as np
from scipy.sparse import bsr_matrix
from scipy.spatial import KDTree
from collections import Counter, defaultdict

def parseVarargin(varargin, arg):
    for k, v in varargin.items():
        if k in arg:
            arg[k] = v
        else:
            raise TypeError('Argument not a default type.')
    return arg

#Segment Nuclei 20X
class CellLabel(object):
    def __init__(self, pth):
        self.T = []
        self.Reg = None
        self.saveToFile = False
        self.posname = []
        self.pth = pth
        self.cell2exclude = []
        self.spacedims = 2;
        self.tracked = None
        self.useCC = False
        self.cell_ids = []
        
        self.Labels = {}
    
    def getXY(self, T, varargin={}):
        if T not in self.Labels:
            if len(self.Labels)>0:
                print('Timestamp not found in labels. Returning nearest.')
        print('Not Implemented')

    def addLabel(self, newlabel, labeltype, T, varargin={}):
        arg = dict()
        arg['posname'] = 'None'
        arg['maxcelldistance'] = 25
        arg['relabel'] = 'nearest'
        arg = parseVarargin(varargin, arg)
        
        newlabel = newlabel.astype(numpy.uint16)
        
        if arg['relabel'] == 'nearest':
            nlabels = len(self.Labels)
            if nlabels==0:
                self.Labels[T] = bsr_matrix(newlabel)
                return
            elif nlabels==1:
                if self.Labels.keys()[0] == T:
                    self.Labels[T] = bsr_matrix(newlabel)
                    print('Warning - added a newlabel with same timestamp as preexisting label. Preexisting label was overwritten.')
                else:
                    existing_t = self.Labels.keys()
                    dist2existing = numpy.abs(numpy.subtract(existing_t, T))
                    nearest_t = existing_t[numpy.argmin(dist2existing)]
                    self.Labels[T] = bsr_matrix(self._relabel_nearest(newlabel, nearest_t))
            elif nlabels>1:
                if T in self.Labels.keys():
                    print('Warning - added a newlabel with same timestamp as preexisting label. Preexisting label was overwritten.')
                    existing_t = self.Labels.keys()
                    dist2existing = numpy.abs(numpy.subtract(existing_t, T))
                    secondNearest_t = existing_t[numpy.argsort(dist2existing)[1]]
                    self.Labels[T] = bsr_matrix(self._relabel_nearest(newlabel, secondNearest_t))
                else:
                    existing_t = self.Labels.keys()
                    dist2existing = numpy.abs(numpy.subtract(existing_t, T))
                    nearest_t = existing_t[numpy.argmin(dist2existing)]
                    self.Labels[T] = bsr_matrix(self._relabel_nearest(newlabel, nearest_t))
            self.tracked = True
        cell_ids = set(range(100000))
        for t, label in self.Labels.items():
            label = label.toarray()
            ids = set(label.flatten())
            cell_ids = cell_ids.intersection(set(ids))
        cell_ids.difference_update(set([0]))
        self.cell_ids = list(cell_ids)
            
    def _relabel_nearest(self, newlabel, nearest_t, varargin={}):
        arg = dict()
        arg['maxcelldistance'] = 25
        nearest_label = self.Labels[nearest_t]
        if isinstance(nearest_label, bsr_matrix):
            nearest_label = nearest_label.toarray()
        nearest = [(p.label, p.centroid, p.coords) for p in regionprops(nearest_label)]
        nearest_label, nearest_xy, coords = zip(*nearest)
        new = [(p.label, p.centroid, p.coords) for p in regionprops(newlabel)]
        new_label, new_xy, _ = zip(*new)
        labels_coords = {k:v for k, _, v in new}
        knntree = KDTree(nearest_xy)
        dists, idx = knntree.query(new_xy, k=2, eps=arg['maxcelldistance'])
        qdata = zip(new_label, dists, idx)
        label_map = {}
        for newlbl_idx, d, j in qdata:
            if d[0]>arg['maxcelldistance']:
                label_map[newlbl_idx] = 0
            else:
                label_map[newlbl_idx] = nearest_label[j[0]]
        counts = Counter(label_map.values())
        for l, c in counts.items():
            if c>1:
                label_map[l] = 0
        if 0 in counts:
            del counts[0]
        if 0 in label_map:
            del label_map[0]
        if counts.most_common(1)>1:
            print('Warning two cells were assigned the same label.', counts.most_common(4))
        for k, v in label_map.items():
            try:
                coords = labels_coords[k]
            except:
                print(k)
            for x, y in coords:
                newlabel[x, y] = v
        
        return bsr_matrix(newlabel)
    
    def applyFuncPerLabel(self, stk, T, func=numpy.mean, varargin={},
                          outtype='matrix'):
        assert stk.shape[2]==len(T)
        label_values = defaultdict(list)
        label_timestamps = self.Labels.keys()
        label_props = {t:regionprops(self.Labels[t].toarray()) for t in label_timestamps}
        label_map = {}
        for t in T:
            label_map[t] = label_timestamps[numpy.argsort(numpy.subtract(label_timestamps, t))[0]]
        data = numpy.zeros((len(self.cell_ids), len(T)))
        label_to_idx = {j:idx for idx, j in enumerate(self.cell_ids)}
        for idx, t in enumerate(T):
            props = label_props[label_map[t]]
            for p in props:
                if p.label not in self.cell_ids:
                    continue
                #get_vals(stk[:,:,idx], p.coords)
                vals = [stk[x,y,idx] for x,y in p.coords]
                data[label_to_idx[p.label], idx] = func(vals)
                label_values[p.label].append((t*1440, func(vals)))
        if outtype=='matrix':
            return data
        elif outtype=='dict':
            return label_values

def segmentNucleiOnly(nuc, varargin={}):
    ## define analysis parameters
    # parameters for nuclei detection
    arg = dict()
    arg['nuc_erode'] = disk(4); # initial erosion to enhance nuclei centers
    arg['nuc_smooth'] = 14; # filtering to smooth it out
    arg['nuc_suppress'] = 0.05; # supression of small peaks - units are in a [0 1] space
    arg['nuc_minarea'] = 30; # smaller then this its not a nuclei
    arg['nuc_maxarea'] = float('inf'); 
    arg['nuc_stretch'] = [1, 99]; 
    arg['nuc_channel'] = 'DeepBlue';
    arg['mindistancefromedge'] =150;
    arg['shrinkmsk'] = disk(50);
    arg['removetopprcentile'] = 0; 
#     arg.['positiontype'] = 'Position'; 
    arg['register'] = []; # optional registration object
    arg['registerreference'] = []; 
#     arg['timefunc'] = lambda t: np.true(t.shape);
    arg['project'] = False; 
    arg['specificframeonly'] = []; # will duplicate that frame for all timepoints
    arg['singleframe'] = False; # must be used in conjunction with specificframeonly. IF true will only return single timepoint
    arg['track_method'] = 'none';
    arg['threshold_method'] = 'otsu';

    arg = parseVarargin(varargin,arg);
    
    #nuc = MD.stkread(Position=well,Channel=arg['nuc_channel'], sortby='TimestampFrame')#,'timefunc',arg.timefunc);
#     nuc_t = MD.image
    NucLabels = numpy.zeros(nuc.shape, dtype=int)

#     nucprj = numpy.mean(nuc,2);
#     msk_lower = nucprj>numpy.percentile(nucprj.flatten(),5)
#     msk_upper = nucprj<=numpy.percentile(nucprj.flatten(),100-arg['removetopprcentile']);
#     msk = msk_lower and msk_upper
    
#     del msk_lower, msk_upper
    
#     msk = binary_fill_holes(msk);
#     msk  = remove_small_objects(msk, min_size=10000, connectivity=2);
#     if not arg['shrinkmsk'] is False:
#         msk[0,:]=0; 
#         msk[:,0]=0; 
#         msk[-1,:]=0; 
#         msk[:,-1]=0; 
#         msk = binary_erosion(msk,arg['shrinkmsk']); 


#     bnd=bwboundaries(msk);
#     bnd=bnd{1}(:,[2 1]);
    for i in range(nuc.shape[2]):
        current_img = nuc[:,:,i]
        nucbw = current_img > filters.threshold_otsu(current_img)
        nucbw = binary_closing(nucbw, disk(2))
        nucbw = binary_fill_holes(nucbw, disk(5))
        nucpeaks = erosion(current_img, arg['nuc_erode'])
        nucpeaks = filters.gaussian(nucpeaks, sigma=arg['nuc_smooth'])
        nucpeaks = numpy.clip(nucpeaks, numpy.percentile(nucpeaks, 1), 
                         numpy.percentile(nucpeaks, 99))
        nucpeaks = numpy.divide(nucpeaks, float(numpy.amax(nucpeaks)))
        numpy.place(nucpeaks, ~nucbw, 0)
        dist_img = distance(nucpeaks)
        nucpeaks_max = peak_local_max(dist_img, min_distance=10)#, threshold_abs=arg['nuc_suppress'])
        peak_img = numpy.zeros(current_img.shape)
        for yx in nucpeaks_max:
            peak_img[yx[0], yx[1]] = 1
        peak_labels = label(peak_img)
        props = regionprops(peak_labels)
        for p in props:
            if p.area==1:
                continue
            elif p.area==2:
                peak_img[p.coords[0][0], p.coords[0][1]] = 0
            elif p.area>2:
                centroid = p.centroid
                pcenter = np.argmin([np.sum(np.abs(centroid - jj)) for jj in p.coords])
                for idx, xy in enumerate(p.coords):
                    if not idx == pcenter:
                        peak_img[xy[0], xy[1]] = 0
        labels = label(peak_img)
        lbls = watershed(dist_img*-1, labels, connectivity=1, watershed_line=True, mask=nucbw)
#         return dist_img, lbls, nucpeaks, numpy.multiply(lbls, peak_img)
        lbls = binary_erosion(lbls, disk(3))
        lbls = binary_dilation(lbls, disk(2))
        lbls = label(lbls)
        lbls = remove_small_objects(lbls, arg['nuc_minarea'])
#         lbls = binary_opening(lbls, disk(4))
        NucLabels[:,:,i] = lbls
    return NucLabels
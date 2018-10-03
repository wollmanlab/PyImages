# Segment imports
import warnings
import numpy
from scipy.ndimage import binary_fill_holes, binary_erosion, distance_transform_edt, binary_closing, binary_dilation
from scipy.ndimage import binary_dilation
from skimage.morphology import remove_small_objects, disk, erosion, watershed
from skimage import filters, img_as_uint
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
# from skfmm import distance
# Tracking imports
import lap

# Cell Label imports
import numpy
import numpy as np
from scipy.sparse import bsr_matrix
from scipy.spatial import KDTree, distance_matrix, distance
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
        
        self.Labels = defaultdict(dict)
    
    def getXY(self, T, labeltype='nuc', return_as='dict', varargin={}):
        if T not in self.Labels[labeltype]:
            pass
        lbl = self.Labels[labeltype][T]
        regprops = regionprops(lbl)
        if return_as=='dict':
            return {p['label']: p['centroid'] for p in regprops}

    def relabel(self, tracks, labeltype='nuc'):
        lbls = self.Labels[labeltype]
        for tp_i, l in lbls.items():
            newlbl = np.zeros_like(l)
            props = regionprops(l)
            lbl_coords = {p.centroid: p.coords for p in props}
            for tp_centroid, newlbl_id in tracks.items():
                for tp_j, centroid in tp_centroid:
                    if tp_i == tp_j:
                        for coord in lbl_coords[centroid]:
                            newlbl[coord[0], coord[1]] = newlbl_id
                    else:
                        continue
            lbls[tp_i] = newlbl

    
    def trackLabels(self, labeltype='nuc', max_displacement=50,
                    max_discontinuity=3):
        lbls = {tp: list(self.getXY(tp).values())
                for tp in self.Labels[labeltype].keys()}
        
        if len(lbls)==0:
            raise ValueError("No labels to track.")
        elif len(lbls)==1:
            warnings.warn("There is only one label so nothing was tracked.")
            return
        timepoints = sorted(list(lbls.keys()))
        consecutive_tp_pairs = [(timepoints[i], timepoints[i+1])
                        for i in range(len(timepoints)-1)]
        tp2idx = {tp:i for i, tp in enumerate(timepoints)}
        trackdict = {}
        # Loop over all consecutive timepoints and use LAP to track
        # Cost is simply the distance of nuclei centroids
        for ti, tj in consecutive_tp_pairs:
            lbl_i, lbl_j = lbls[ti], lbls[tj] # Assuming these are centroids
            cost_matrix = distance_matrix(lbl_i, lbl_j)
            total_cost, column2row, row2column = lap.lapjv(cost_matrix,
                                                           cost_limit=max_displacement,
                                                           extend_cost=True)
            # Parse LAP output to build dictionary of paired items
            for col, row in enumerate(column2row):
                if row == -1:
                    trackdict[(ti, tuple(lbl_i[col]))] = -1
                else:
                    trackdict[(ti, tuple(lbl_i[col]))] = (tj, tuple(lbl_j[row]))
        # Merge all pairs that link together into merged track
        # Pairs are linked if they connected through a common element.
        merged_tracks = []
        for k, v in list(trackdict.items()):
            if v in trackdict: # Condition means v links to another pair from different timepoint
                merged_track = [k] # Start building the merged track of linked pairs
                while v in trackdict: # Keep building until no more
                    merged_track += [v]
                    trackdict.pop(k)
                    k = v
                    v = trackdict[v]
                trackdict.pop(k)
                if not v == -1: # If last element is unpaired add just the key
                    merged_track += [v] # else add the last linked item
                merged_tracks.append(merged_track)
        # Remove complete tracks from further linking (i.e. all timepoints are connected)
        incomplete_tracks = [t for t in merged_tracks if len(t)<len(lbls)]
        tracks = incomplete_tracks
        # Build data structures for gap closing cost matrix creation
        track_starts = np.array([i[0][0] for i in tracks])
        track_ends = np.array([i[-1][0] for i in tracks])

        track_xy_start = np.array([i[0][1] for i in tracks])
        track_xy_end = np.array([i[-1][1] for i in tracks])
        
        # Try to link more tracks by allowing gaps within track
        gap_cost_mat = np.ones((n, n))*(max_displacement+1) # this should be bigger than max_displacement
        for idx, trck in enumerate(tracks):
            tstart = trck[0][0]
            tend = trck[-1][0]
            xy_start = trck[0][1]
            xy_end = trck[-1][1]
            tracks_starting_after_this_ends = np.where((track_starts>tend) & (track_starts<tend+max_discontinuity))[0]
            if len(tracks_starting_after_this_ends)>0:
                start_tree = KDTree(track_xy_start[tracks_starting_after_this_ends])
                possmerge_start = start_tree.query(xy_end, k=50, distance_upper_bound=max_displacement)
                for d, tidx in zip(possmerge_start[0], possmerge_start[1]):
        #             if idx == tidx:
        #                 continue
                    if d>max_displacement:
                        break
                    else:
                        gap_cost_mat[idx, tracks_starting_after_this_ends[tidx]] = d
            tracks_ending_before_this_track = np.where((track_ends<tstart) & (track_ends>tstart-max_discontinuity))[0]
            if len(tracks_ending_before_this_track)>0:
                end_tree = KDTree(track_xy_end[tracks_ending_before_this_track])
                possmerge_end = end_tree.query(xy_start, k=50, distance_upper_bound=max_displacement)
                for d, tidx in zip(possmerge_end[0], possmerge_end[1]):
        #             if idx == tidx:
        #                 continue
                    if d>max_displacement:
                        break
                    else:
                        cc.append(d)
                        ii.append(idx)
                        kk.append(tidx)
                        gap_cost_mat[idx, tracks_ending_before_this_track[tidx]] = d
        a,b,c = lap.lapjv(gap_cost_mat, cost_limit=max_displacement,
                          extend_cost=True)
        leftover_merges = []
        for idx, i in enumerate(c):
            if i == -1:
                continue
            track_i = tracks[idx]
            track_j = tracks[i]
            leftover_merges.append(track_i+track_j)
        # Build final output and ensure there are no duplicate tracks
        final_tracks = {}
        track_id = 1
        for track in merged_tracks+leftover_merges:
            track = tuple(sorted(track, key=lambda x: x[0]))
            if track not in final_tracks:
                final_tracks[track] = track_id
                track_id += 1
        self.tracks = final_tracks
        return final_tracks
    #print(merged_xys)
    
    def addLabel(self, newlabel, labeltype, T, varargin={}):
        arg = dict()
        arg['posname'] = 'None'
        arg['maxcelldistance'] = 25
        arg['relabel'] = None
        arg = parseVarargin(varargin, arg)
        
        newlabel = newlabel.astype(numpy.uint16)
        self.Labels[labeltype][T] = newlabel
    
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

class Registration(object):
    def __init__(self, tforms=dict()):
        self.tforms=tforms
    def find_translation(self, src, dest, timepoint):
        tvec = register_translation(dest, src)
        tform = AffineTransform(translation=(tvec[0][1], tvec[0][0]))
        self.tforms[timepoint] = tform
        return tform
    def apply_tform(self, dest, dest_timepoint=None, order=1):
        dest = dest.copy()
        if isinstance(dest_timepoint, skimage.transform._geometric.AffineTransform):
            pass
        else:
            dest_timepoint = self.tforms[dest_timepoint]
        dest = warp(dest, dest_timepoint, preserve_range=True, order=order)
        return dest
    def map_and_apply_tforms(self, dest_stack, dest_timepoints, grouping='nearest', order=1):
        dest_stack = dest_stack.copy()
        source_timepoints = list(self.tforms.keys())
        if grouping == 'nearest':
            grouping = []
            for dest_tp in dest_timepoints:
                tdist = np.abs(np.subtract(source_timepoints, dest_tp))
                closest_source_tp = np.argmin(tdist)
                grouping.append(source_timepoints[closest_source_tp])
        dest_stack = np.stack([self.apply_tform(dest_stack[:,:,i], self.tforms[grouping[i]], order=order) 
                              for i in range(dest_stack.shape[2])])
        return dest_stack
    
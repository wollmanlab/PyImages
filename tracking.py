# Placeholder for tracking framework
import numpy as np
from collections import defaultdict
from scipy.spatial import distance_matrix, KDTree, distance
import lap

lbl_t1 = np.random.randint(20, 2028, size=(1000, 2))
lbl_t2 = np.random.randint(-15, 15, size=(1000,2))+lbl_t1
lbl_t3 = np.random.randint(-15, 15, size=(1000,2))+lbl_t2

max_displacement = 20
max_discontinuity = 3

timepoints = [1,2,3]
consecutive_tp_pairs = [(timepoints[i], timepoints[i+1])
                        for i in range(len(timepoints)-1)]
lbls = {1: lbl_t1, 2: lbl_t2, 3: lbl_t3}
tp2idx = {tp:i for i, tp in enumerate(timepoints)}
tracks = []
segment_list = []
track_id = 1
for ti, tj in consecutive_tp_pairs:
    lbl_i, lbl_j = lbls[ti], lbls[tj] # Assuming these are centroids
    cost_matrix = distance_matrix(lbl_i, lbl_j)
    total_cost, column2row, row2column = lap.lapjv(cost_matrix,
                                                   cost_limit=max_displacement,
                                                   extend_cost=True)
    for col, row in enumerate(column2row):
        if col == -1:
            tracks.append(([ti], [lbl_i[col]])) # time and xy
        else:
            tracks.append(([ti, tj], [lbl_i[col], lbl_j[row]]))
        track_id += 1
        
        
track_starts = np.array([i[0][0] for i in tracks])
track_ends = np.array([i[0][1] for i in tracks])

track_xy_start = np.array([i[1][0] for i in tracks])
track_xy_end = np.array([i[1][1] for i in tracks])

n = len(tracks)
cc = [] # Finite costs
ii = [] # indices of rows
kk = []
gap_cost_mat = np.ones((n, n))*1000
for idx, (tps, xys) in enumerate(tracks):
    tstart = tps[0]
    tend = tps[-1]
    xy_start = xys[0]
    xy_end = xys[-1]
    tracks_starting_after_this_ends = np.where((track_starts>=tend) & (track_starts<tend+max_discontinuity))[0]
    if len(tracks_starting_after_this_ends)>0:
        start_tree = KDTree(track_xy_start[tracks_starting_after_this_ends])
        possmerge_start = start_tree.query(xy_end, k=50, distance_upper_bound=max_displacement)
        for d, tidx in zip(possmerge_start[0], possmerge_start[1]):
#             if idx == tidx:
#                 continue
            if d>max_displacement:
                break
            else:
                cc.append(d)
                ii.append(idx)
                kk.append(tidx)
                gap_cost_mat[idx, tracks_starting_after_this_ends[tidx]] = d
    tracks_ending_before_this_track = np.where((track_ends<=tstart) & (track_ends>tstart-max_discontinuity))[0]
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
for idx, i in enumerate(c):
    if i == -1:
        continue
    track_i = tracks[idx]
    track_j = tracks[i]
    if track_i[0][0]>track_j[0][0]:
        track_j, track_i = track_i, track_j
    ti_tp = track_i[0]
    ti_xy = track_i[1]
    
    tj_tp = track_j[0]
    tj_xy = track_j[1]
    if ti_tp[-1]==tj_tp[0]:
#         if ti_tp[0]==tj_tp[0]:
        merged_tps = ti_tp[:-1]+tj_tp
        merged_xys = ti_xy[:-1]+tj_xy
    elif ti_tp[-1]<tj_tp[0]:
        merged_tps = ti_tp+tj_tp
        merged_xys = ti_xy+tj_xy
    elif ti_tp[-1]>tj_tp[0]:
        merged_tps = tj_tp+ti_tp
        merged_xys = tj_xy+ti_xy
    merged_xys = list(map(tuple, merged_xys))
    if len(merged_xys)>3:
        print(track_i, track_j, idx, i)
    #print(merged_xys)
    
    
a,b,c = lap.lapjv(gap_cost_mat, cost_limit=max_displacement,
                  extend_cost=True)
track_dict = {}
for idx, i in enumerate(c):
    if i == -1:
        continue
    track_i = tracks[idx]
    track_j = tracks[i]
    if track_i[0][0]>track_j[0][0]:
        track_j, track_i = track_i, track_j
    ti_tp = track_i[0]
    ti_xy = track_i[1]
    
    tj_tp = track_j[0]
    tj_xy = track_j[1]
    if ti_tp[-1]==tj_tp[0]:
#         if ti_tp[0]==tj_tp[0]:
        merged_tps = ti_tp[:-1]+tj_tp
        merged_xys = ti_xy[:-1]+tj_xy
    elif ti_tp[-1]<tj_tp[0]:
        merged_tps = ti_tp+tj_tp
        merged_xys = ti_xy+tj_xy
    elif ti_tp[-1]>tj_tp[0]:
        merged_tps = tj_tp+ti_tp
        merged_xys = tj_xy+ti_xy
    merged_xys = tuple(map(tuple, merged_xys))
    if len(merged_xys)>3:
        print(track_i, track_j, idx, i)
    track_dict[merged_xys] = merged_tps
    #print(merged_xys)
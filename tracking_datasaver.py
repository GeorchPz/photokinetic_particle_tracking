# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 11:13:20 2023

@author: Jorge Pottiez López-Jurado

Objective
---------
    Store all Trackpy's DataFrames related to a video in an easy to access way.

How to
------
    Feed the following inputs:
    · locate & tracking (tp.link + tp.filter_stubs) dictionaries 
    (with all the optional parameters we want),
    · Video file's path.
"""

from pandas import HDFStore, Series
import trackpy as tp
import pims
# import numpy as np


@pims.pipeline
def as_grey(frame):
    'RGB to grey image'
    R = frame[:, :, 0]; G = frame[:, :, 1]; B = frame[:, :, 2]
    
    grey = 0.2125*R + 0.7154*G + 0.0721*B
    return grey

def locate_n_track(frames, fast_performance = True):
    """
    Computes all of Trackpy's necessary dataframes.

    Parameters
    ----------
    frames : pims.frame.Frame
    fast_performance : bool, optional

    Returns DataFrames
    -------
    f : located particles
    t : tracked particles
    t1: t without the shortest trajectories

    """
    
    tp_engine= ('numba' if fast_performance else 'python')
    
    'locate'
    f = tp.batch(
        frames, **loc_pms, invert= True, processes= 1, engine= tp_engine
        )
    # f.columns: 'y', 'x', 'mass', 'size', ... , 'frame'
    
    'tracking'
    # We unpack tracking parameters to be passed to both func()
    link_pms  = {key: val for key, val in trck_pms.items() if key != 'threshold'}
    threshold = trck_pms['threshold']
    
    t = tp.link(f, **link_pms)
    # t.columns: 'y', 'x', 'mass', 'size', 'ecc', ..., 'frame', 'particle'
    
    # Filter spurious/short trajectories
    t1 = tp.filter_stubs(t, threshold)
    t1.reset_index(drop= True, inplace= True) # reset index to default
    
    # Compare the num of particles in the unfiltered and filtered data
    print( 'Before filtering: ', t['particle'].nunique() )
    print( 'After:      ', t1['particle'].nunique() )
    # .nunique(): number of distinct elements

    return f, t, t1

def storing(storage_path, vid_path, dfs_dict):
    """
    Creates HDF5 file and stores all computed data inside.
    
    Parameters
    ----------
    storage_path : str
    vid_path : str
        so that the .h5 document as the same name as the video
    dfs_dict : dict
        of pandas' DataFrames or Series
        the dict's keys will be used to access the df from the .h5 file
    names : list of str
    """
    
    vid_file = vid_path.split('//')[-1]
    storage_name = vid_file.split('.')[0] # removes the .avi extension
    store = HDFStore(f'{storage_path}//{storage_name}.h5')
    
    for name, df in dfs_dict.items():
        store.put(name,df)
    
    print(store.info())
    store.close()



'--- --- --- --- --- --- --- INPUTS --- --- --- --- --- --- ---'
'Possible paths'
# Normal Swim, 1365 frames
path1 = "00_Experiments_685nm//020223//020123_NormalSwim.avi"
## UV flash (30s, 63s)
path2 = "00_Experiments_685nm//020223//020123_UV5percentSwim.avi"
# UV flash (30s, 3.5mins), 4511 frames
path3 = "02_Experiments_715nm//300523_715nmFilter_NODCMU//Area2_UV10_300523.avi"


loc_pms = dict(
    diameter = 21,
    minmass  = 800,
    maxsize  = None,
    separation = 21+1,
    noise_size= 1
    )

trck_pms = dict(
    search_range= 18,
    memory= 3,
    threshold= 10,
    )

'--- --- --- --- --- --- --- DATA STORAGE --- --- --- --- --- --- ---'
def main(loc_pms, trck_pms, vid_path):
    """
    Import video, compute and store dataframes,
    including Trackpy's parameters we've used.
    """
    frames= as_grey(pims.open(vid_path))[0,10]
    f, t, t1 = locate_n_track(frames)
    
    dfs = dict(
        parameters  = Series({**loc_pms, **trck_pms}),
        locating    = f,
        tracking    = t,
        tracking_cut= t1
        )
    
    storage_path = '.datafiles'
    storing(storage_path, vid_path, dfs)
    
    return frames, f, t, t1

frames, f, t, t1 = main(loc_pms, trck_pms, path2)

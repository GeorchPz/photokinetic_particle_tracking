# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 12:09:53 2023

@author: Jorge Pottiez López-Jurado

Objective
---------
As the name suggests, this file is for checking the kinematics,
but mainly the fitting & the n parameter (related to the number
of spline knots).

how to
------
To use this file, it has to share folder with Kinematics_datasaver.

Feed the following inputs:
    · path of the Tracking .h5 datafile (so that we can extract
      the particle's position),
    · p_num: a particle's number/labbel,
    · N: the ratio between the number of spline knots and the
      number of particles.
"""

import matplotlib.pyplot as plt
from pandas import HDFStore

from kinematics_datasaver import Particle_Kinematics, System_Kinematics


def Particle_Kin_checker(p_df, n):
    "Checks the fitting & the speed & angle"
    # Computing Kinematics and appending it to df
    PK = Particle_Kinematics(p_df, n)
    
    # Plots trajectory data & fitting
    PK.check_traj_fit()
    PK.plot_vel_polar()
    
    # Expand p_df, adding kinematics values to it
    return PK.to_df()

def Syst_Kin_checker(p_num, n):
    """
    Checks that for a given particle, the values are coherent
    & well appended to k, the Kinematics df
    """
    k_df = System_Kinematics(t1, n).to_df()
    p_df = k_df.groupby('particle').get_group(p_num)
    
    plt.figure(figsize= (8,8))
    plt.plot(p_df.frame, p_df.speed)


'--- --- --- --- --- --- --- INPUTS --- --- --- --- --- --- ---'
h5_path = '.datafiles//020123_NormalSwim.h5'
p_num = 2 # Particle labbel we are interested in
n = .5

'--- --- --- --- --- --- --- TESTING --- --- --- --- --- --- ---'
'Import data'
store = HDFStore(h5_path, mode= 'r')
# t = store.get('tracking')
t1 = store.get('tracking_cut') #threshold= 10
store.close()

'pº particle dataframe'
gb = t1.groupby(['particle']) # subsetting dataframe
p_df = gb.get_group(p_num)
# p_df = p_df[:50] # if we use just a few datapoints


Particle_Kin_checker(p_df, n)
# Syst_Kin_checker(p_num, n)
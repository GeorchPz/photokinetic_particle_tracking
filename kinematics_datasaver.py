# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 13:53:56 2023

@author: Jorge Pottiez López-Jurado

Objective
---------
    Following the same data structure as Trackpy's, we create & save a 
    DataFrame with the kinematic values (velocity in cartesian & polar
    coordenates) of all the particles.
    
How to
------
Feed the following inputs:
    path of the Tracking .h5 datafile (so that we can extract
    the particle's position).

if we want to play around with the number of spline knots,
use Particle_Kinematics_checker.py.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import interpolate


class Particle_Kinematics:
    """
    Computes kinematics of a particle with a cubic spline fitting
    given its dataframe.

    Parameters
    ----------
    particle_df : DataFrame
        with frame, x & y columns
    n : float, optional
        between 0.01 and 1, it represents the ratio between the number
        of spline knots and the number of particles. The default is 0.5.
    """
    
    def __init__(self, particle_df, n= 0.5):
        self.p_df = particle_df
        
        # Time in frames 
        self.t = self.p_df.frame
        # Lengths in pixels (px)
        self.x = self.p_df.x
        self.y = self.p_df.y
        
        # Number of datapoints
        self.N_frames = self.t.shape[0]
        # Ratio of ~ N_knots/N_particles
        # n = .5 +/- .2 : range of variability with a priori fine results
        self.n = n
    
    
    'GENERAL CLASS FUNCTIONS'
    def spline_fit(self, X):
        """
        Cubic spline fitting:
            · time (frames) as the independent variable
            · number of knots as a free parameter, determinated by n
            · n: ratio of ~ N_knots/N_particles

        Parameters
        ----------
        X : array or Series
            Basically, X= x or y
        
        Returns
        -------
        X_fit : scipy.interpolate.BSpline class
            works as a X(t) function
        """
        
        # If not specified otherwise, N_knots ~ N/2
        N_knots = round(self.n*self.N_frames)
        
        t_q = np.linspace(0, 1, N_knots+2)[1:-1] 
        q_knots = np.quantile(self.t, t_q)
        
        tck = interpolate.splrep(self.t, X, t= q_knots, k= 3)
        # tck = (t,c,k): knots vector, B-spline coefs, spline's degree
        X_fit = interpolate.BSpline(*tck) # *tck: unpacks t,c,k values
        
        return X_fit
    
    def V_func(self, X):
        'Speed function for X= x or y'
        X_f = self.spline_fit(X) # X as function
        V_f = X_f.derivative()
        return V_f
    
    def A_func(self, X):
        'Acceleration function for X= x or y'
        V_f = self.V_func(X)
        A_f = V_f.derivative()
        return A_f
    
    'PARTICLES PROPERTIES'
    def vel(self):
        'Velocity vector calculated from dataframe values'
        vx_f = self.V_func(self.x)
        vy_f = self.V_func(self.y)
        
        v_x, v_y = vx_f(self.t), vy_f(self.t)
        return (v_x, v_y)

    def speed(self):
        "Velocity's modue"
        v_x, v_y = self.vel()
        return np.sqrt( v_x**2 + v_y**2 )
    
    def angle(self):
        "Velocity's angle (as in polar coordenates)"
        v_x, v_y = self.vel()
        return np.angle( v_x + v_y*1j )
    
    def acc(self):
        'Acceleration vector'
        ax_f = self.A_func(self.x)
        ay_f = self.A_func(self.y)
        
        a_x, a_y = ax_f(self.t), ay_f(self.t)
        return (a_x, a_y)
    
    def to_df(self):
        'Creates new Kinematics dataframe'
        PK_df = self.p_df.copy()
        
        v_x, v_y = self.vel()
        PK_df.insert(8, 'v_y', v_y )
        PK_df.insert(9, 'v_x', v_x )
        PK_df.insert(10, 'speed', self.speed() )
        PK_df.insert(11, 'angle', self.angle() )
        # 8,...,11 : column position in the df where it will be inserted
        
        # note: a_x, a_y are not added to df since I had no use for them,
        # but adding them (for a Kalman filter ?) is trivial
        return PK_df
    
    'PLOTS'
    def check_traj_fit(self):
        """
        Plots the evolution of x & y over time of a given particle, where:
            · the data position is set as a scatter plot,
            · the spline fitting is set as a continous lines.
        Plots the datapoints & fitting of the trajectory (y over x).
        """
        # Fitting functions
        x_f = self.spline_fit(self.x)
        y_f = self.spline_fit(self.y)
        
        # Plotting arrays
        tmin, tmax = self.t.min(), self.t.max()
        t_arr = np.linspace(tmin, tmax, self.N_frames*10)
        x_arr = x_f(t_arr)
        y_arr = y_f(t_arr)
        
        # Plots
        fig, (ax1, ax2) = plt.subplots(1,2, figsize= (12,6))
        ## x,y vs t
        ax1.scatter(self.t, self.x, s=2, label='x data')
        ax1.scatter(self.t, self.y, s=2, label='y data')
        ax1.plot(t_arr, x_arr, linewidth=0.5, label='x(t) fitting')
        ax1.plot(t_arr, y_arr, linewidth=0.5, label='y(t) fitting')
        ax1.set(xlabel= 't', ylabel= 'x, y')
        ax1.legend()
        ## trajectory: y vs x
        ax2.scatter(x_arr, y_arr, s=1, c= 'g', label= 'data')
        ax2.plot(x_arr, y_arr, linewidth=0.5, c= 'g', label='fitting')
        ax2.set(xlabel= 'x', ylabel= 'y')
        ax2.legend()
    
    def plot_vel_polar(self):
        """
        Plots the speed (velocity's module) & angle evolution
        for a given particle.
        """
        # Arrays
        t, v, a = self.t, self.speed(), self.angle()*180/np.pi
        # Plot
        fig, (ax1, ax2) = plt.subplots(1,2, figsize= (12,6))
        
        ax1.plot(t,v)
        ax1.set(xlabel= 't (frames)', ylabel= 'Speed (px/frame)')
        ax2.plot(t,a)
        ax2.set(xlabel= 't (frames)', ylabel= 'Vel angle (º)')


class System_Kinematics:
    """
    Computes Kinematics for all particles/trajectories
    using Particle_Kinematics class
    
    Parameters
    ----------
    tracking_df : DataFrame
        with frame, x, y & particle columns  
    n : float, optional
        between 0.01 and 1, it represents the ratio between the number
        of spline knots and the number of particles. The default is 0.5.
    """
    
    def __init__(self, tracking_df, n= 0.5):

        # All tracked paths' DataFrame (from tp.link)
        self.trck = tracking_df
        self.gb = self.trck.groupby('particle')
        
        # Number of particles
        self.N_particles = self.gb.ngroups
        # Highiest particle label
        self.M_particle = self.trck.particle.max()
        # Ratio of ~ N_knots/N_particles
        self.n = n # n = .5 +/- .2
    
    def to_df(self):
        print('\n','Computing Kinematics:','\n')
        
        p_dfs_list = [] # List of all particles' df
        for particle, p_df in self.gb:
            PK = Particle_Kinematics(p_df, n= self.n)
            p_dfs_list.append( PK.to_df() )
            
            if particle%100 == 0:
                print(f'Particle {particle} out of {self.M_particle}')
        
        k = pd.concat(p_dfs_list)
        k.sort_index(inplace= True)
        return k


if __name__ == '__main__':

    'INPUTS: path'
    path1 = '.datafiles//020123_NormalSwim.h5'
    path2 = '.datafiles//020123_UV5percentSwim.h5'
    path3 = '.datafiles//Area2_UV10_300523.h5'
    
    def main(h5_path):
        "store Kinematics dataframes in already created .h5 file"
        store = pd.HDFStore(h5_path)
        
        # t  = store.get('tracking')
        t1 = store.get('tracking_cut') #threshold= 10
        
        # Compute
        k = System_Kinematics(t1, n= 0.5).to_df()
        # Export
        store.put('kinematics',k)
        print(store.info())
        store.close()
    main(path3)
    
    '''
    We must use tracking_cut df
    if we try:
        P_K = Particles_Kinematics(t)
        P_K.to_df()
    we get:
        TypeError: m > k must hold (from interpolate.splrep)
    why?
        https://stackoverflow.com/questions/29934831/matplotlib-draw-spline-from-3-points
        The number of spline knots (m) must be greater than the degree of the spline (k=3)
    '''

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 15:21:06 2023

@author: Jorge Pottiez López-Jurado

Objective
---------
Extracting information of the tracking & kinematics of the system's particles.

How to
------
Feed the following inputs:
    · path of the Tracking .h5 datafile.

The python functions to get some information are gathered in the
Analyse_Kinematics class.

""" #???

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gif_maker

class Analyse_Kinematics:
    """
    Gathering of functions that gives us some kind of information about
        · evolving_hist, for a given series of values
        · categorise, for a given series

    Parameters
    ----------
    kin_df : TYPE
        DESCRIPTION.
    vid_title : TYPE
        DESCRIPTION.
    """
    
    # Instance variables
    dpi = 500   # figures' dots per inch
    bins= 200   # histogram's bins
    spf = 1     # gif's secs per frame
    folder_path = 'figs'
    temp_path = f'{folder_path}//.temp_figs'
    
    def __init__(self, kin_df, vid_title):
        # Kinematics dataframe
        self.k = kin_df
        # Figure Frame specifications
        self.vid_title = vid_title
        
        'Physical magnitudes'
        self.t     = kin_df.frame # time (in frames)
        # self.speed = kin_df.speed
        # self.angle = kin_df.angle
        
        gb =  kin_df.groupby('particle')
        # Tracking lengths (Number of frames each particle apears in)
        self.length_track = gb.frame.size().rename('track_length')
    
    'GENERAL FUNCTIONS'
    def better_hist(self, series, title= None, x_max= None):
        """
        General-purpose histogram with optional title and x boundry
        ( based in pandas df.hist() ).

        Parameters
        ----------
        series : DataFrame's column or Series
            values to be plotted.
        title : str, optional
            Histogram's title.
            The default is f'{series.name}\n{self.vid_title}'.
        x_max : scalar, optional
            If we are creating a gif, we want to set the same x-axis
            boundry for every frame. The default is False.

        Returns
        -------
        fig : matplotlib figure
        """
        
        self.title = (
            title if title != None
            else f'{series.name}\n{self.vid_title}'
            )
        x_lim = ((-1, x_max) if x_max != None else False)
        
        fig, ax = plt.subplots()
        
        series.hist(ax= ax, bins= self.bins)
        ax.set(
            title= self.title, xlim = x_lim,
            xlabel= series.name, ylabel= 'count'
            )
        
        return fig
    
    def save_frame(self, fig, force_path= None):
        """
        Saves figure, optionally the path of the .png can be changed.
        
        Parameters
        ----------
        fig : matplotlib figure
        force_path : str, optional
            Path of the figure's file.
            The default is to store the fig in self.folder_path.
        """
        # Path where the Frame's figures will be stored
        filename = self.title.replace('\n', '  ').replace(' ', '_')
        fig_path = (force_path if force_path!= None
                    else f'{self.folder_path}//{filename}.png')
        
        fig.savefig(fig_path, dpi = self.dpi, bbox_inches='tight',
                    facecolor="white", transparent = False)
        return None
    
    'EXTRACTING INFORMATION'
    def track_length_info(self):
        """
        Gives back length_track's statistics & distribution
        """
        l_track = self.length_track
        ### note:
        # l_track[ (l_track >= 100) ] is equivalent
        # to set Trackpy's threshold to 100
        
        # Generate descriptive statistics
        desc_str = l_track.describe().to_string()
        print(f'{l_track.name} description:\n{desc_str}\n')
        
        # Plots distribution of tracking lenghts
        fig = self.better_hist(l_track)
        self.save_frame(fig)
        
        # Series to df
        traj = l_track.reset_index()
        return traj
    
    def evolving_hist(self, series, step):
        """
        Creates a gif that shows the time evolution of a Series' population
        as a compilation of histograms.

        Parameters
        ----------
        series : DataFrame's column or Series.
        step : int
            Time step, number of video's frames used for each histogram.
        """
        
        s_max = np.ceil(series.max()) #ceil: rounds up to whole number
        
        semistep = int(step/2)
        # Time boundaries
        t_m, t_M = self.t.min(), self.t.max()
        # Array of instants that separated by 1 step
        t_stars = np.arange(
            t_m -semistep, t_M +semistep,
            step)[1:-1] # [1:-1]: we remove first and last vals
        
        'set max of y axes? How?' # ???
        
        print(f'computing gif for {series.name}s:')
        print(f't*: {t_stars}')
        
        images_path= []
        
        for t_star in t_stars:
            # Ranges we're interested for each plot-frame.
            t_rg = (t_star - semistep, t_star + semistep)
            s_rg = series[(t_rg[0] <= self.t) & (self.t <= t_rg[1])]
            
            print(f't in {t_rg}')
            
            # figure
            fig = self.better_hist(
                s_rg, x_max= s_max, title = f'frames $\in$ {t_rg}'
                )
            plt.show()
            
            # Path where the temporary figures will be stored
            t_path = f'{self.temp_path}//{t_star}.png'
            
            images_path.append(t_path)
            self.save_frame(fig, force_path = t_path)
            # plt.close(fig)
        
        gif_name= f'{series.name}_evolving_hist__{self.vid_title}'
        gif_path= f'{self.folder_path}//{gif_name}.gif'
        # Creates gif
        Gif_maker.gif_maker(images_path, gif_path,  spf= self.spf)
        print('gif created','\n')
        Gif_maker.remove_imgs(images_path)
        
        return None
    
    def categorise(self, series, categories, separation, make_fig= True):
        """
        Separates data into different bins/categories for each frame.
        Then, it creates a figure of:
            · the bins'edges on top of the series' histogram,
            · the time-evolution of the frecuencies for each category.
        
        Parameters
        ----------
        series : DataFrame's column or Series.
        
        categories : sequence of str (len == n)
            Names of the categories we want to create.
        separation : sequence of scalars (len == n-1)
            Defines the bin middle edges, not having into account the first
            & last edges since these are the min $ max values of the series.
        make_fig : bool, optional
            If True, it creates the figure. The default is True.
        
        Returns
        -------
        categs_vs_t : TYPE
            DESCRIPTION.
        
        categ = ['slow (<1 px/fr)', 'medium', 'fast (>6 px/fr)']
        speed_sep = [1, 6] # in px/frame
        """
        'CALCULATIONS'
        # Series boundaries
        m, M = series.min(), series.max()
        
        ### scalar: sets the number of equal-width bins
        # edges = 3 
        ### list: manually sets the edges of the bins, non-uniform width
        edges = [m-1, *separation, M+1]
        
        # Segment and sort series into bins
        cut = pd.cut(series, bins= edges, labels= categories)
        self.k[f'{series.name}_categ'] = cut
        
        # Frequencies distribution for the series' categories
        frec_categ = cut.value_counts(normalize= True)
        print(
            f'{series.name} distribution for all frames:\n',
            frec_categ.to_string()
            )
        
        gb_t =  self.k.groupby('frame')
        # Frequencies distribution for each categories & frame
        prob = gb_t[f'{series.name}_categ'].value_counts(normalize= True)
        prob = prob.rename('probability')
        
        # DataFrame of categories' frequencies with time (frames) as index
        categs_vs_t = prob.unstack()
        
        'PLOTS'
        if make_fig:
            fig, (ax1,ax2) = plt.subplots(2,1, figsize= (12,8))
            
            self.title = f'{series.name} categories for' +'\n'+ self.vid_title
            ax1.set_title(self.title, fontsize= 16)
            
            "1º plot: bins'edges on series' histogram"
            series.hist(ax= ax1, bins= self.bins)
            ax1.set(xlabel= series.name, ylabel= 'count', yscale= 'linear')
            
            n = len(edges)
            colours = plt.cm.plasma(np.linspace(.2, 1-.2, n)) # edges' colours
            labels  = categories + ['End'] # edges' labels
            
            for i in range(n):
                ax1.axvline(x = edges[i], label= labels[i], color= colours[i])
            ax1.legend(title= 'Beginning of:')
            
            "2º plot: time evolution for each category"
            categs_vs_t.plot(
                ax= ax2, use_index= True, ylabel= prob.name,
                colormap = 'plasma', grid= True
                )
            
            self.save_frame(fig)
        return categs_vs_t

if __name__ == '__main__':
    
    '--- --- --- --- --- --- ---   INPUT   --- --- --- --- --- --- ---'
    path = ('.datafiles//020123_NormalSwim.h5',
            '.datafiles//020123_UV5percentSwim.h5', # UV flash at: 30s=450fr
            '.datafiles//Area2_UV10_300523.h5')[0]
    
    'Import data'
    store = pd.HDFStore(path, mode= 'r')
    kin_df  = store.get('kinematics')
    store.close()
    
    fig_title = path.split('//')[-1].replace('.h5','')
    
    '--- --- --- --- --- --- --- ANALYSES --- --- --- --- --- --- ---'
    An_K = Analyse_Kinematics(kin_df, fig_title)
    
    'Track length'
    traj = An_K.track_length_info()
    
    'Evolving histograms'
    step = 100
    An_K.evolving_hist(kin_df.speed, step)
    # An_K.evolving_hist(kin_df.angle, step)
    
    'Categorise speeds'
    speed_sep = [1, 6] # in px/frame
    categs = ['slow (<1 px/fr)', 'medium', 'fast (>6 px/fr)']
    
    categs_vs_t = An_K.categorise(kin_df.speed, categs, speed_sep)
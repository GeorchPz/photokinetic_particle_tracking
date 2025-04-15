# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 10:11:17 2023

@author: Jorge Pottiez López-Jurado

Objective
---------
    Optimize the parameters of Trackpy's functions for a better particle
    detection and tracking.
    From my personal experience, its best to do this for every movie we want
    to analyse.

How to
------
Its important to conceptually separate two key procedures using Trackpy:
    Locate:     finding features' locations in an image/frame.
    Tracking:   follow particle from frame to frame, giving
                a label to each trajectory computed.

Therefore, we must first find the best-possible parameters for locate
& then for the tracking.
"""

import matplotlib.pyplot as plt
import trackpy as tp
import pims

import gif_maker


@pims.pipeline
def as_grey(frame):
    '''
    An RGB image uses 3 color channels: Red, Green, & Blue,
    but we need a 1-channel image for Trackpy.
    
    For that, we have several option:
        - weighted average of the channels
        - take only one of the channels
    
    The objective being to maximize the contrast of the 1-channel

    Parameters
    ----------
    frame : array (3 dim)
        length x Width x RGB-Colour channels

    Returns
    -------
    grey : array (2 dim)
        Length x Width
    '''
    R = frame[:, :, 0]
    G = frame[:, :, 1]
    B = frame[:, :, 2]
    # Different options:
    grey = 0.2125*R + 0.7154*G + 0.0721*B
    # red  = 1*R + 0*G + 0*B
    return grey

def dict_to_str(dic):
    'Create a text-description of a dictionary (of Trackpy parameters)'
    s = '\n\n'.join( f'{key}:\n{val}' for key, val in dic.items())
    return s


class Analyse_Frame: 
    '''
    This class is aimed for extracting information of a given frame
    
    Parameters
    ----------
    video_path : str
    frame : pims.frame.Frame
    frame_title : str
    
    locate_parameters : dict
        all the tp.locate variables we want to work with
    tracking_parameters : dict, optional
        tp.link & tp.filter_stubs variables
    
    How to optimize computation?
    ----------------------------
    if we are just interested in toggling the parameters for locate:
        tracking_parameters= None
    '''
    
    'Instance variables'
    bins_per_hist = 20                  # num. of histogram bins
    dpi = 500                           # figures' resolution (dots per inch)
    folder_path = 'figs//.temp_figs'    # where the temporary gif's frames will be stored
    
    def __init__(
            self, video_path, frame, frame_title,
            locate_parameters, tracking_parameters= None
            ):

        self.vid_name = video_path.split("//")[-1]
        self.loc_pms  = locate_parameters
        self.trck_pms = tracking_parameters
        # Frame specifications
        self.frame   = frame
        self.title   = frame_title 
        
        # Frame's path
        self.path = f'{self.folder_path}//{self.title}.png'
        
        
        if tracking_parameters == None:
            # we aren't tracking the particles, only locating them
            self.loc_df = tp.locate(
                self.frame, **self.loc_pms, invert= True
                                    ) # invert = True: if the cells are black!
            # loc_df.columns == ['y', 'x', 'mass', 'size', 'ecc', ..., 'frame']
        else:
            # resource optimization: if we use tracking_parameters
            # then we don't need to locate each individual frame
            pass
        
        # Text-tag to distinguish each analysis done with different parameters
        self.fig_txt_loc = dict_to_str({
            'file': self.vid_name, **self.loc_pms
            })
    
    def better_annotate(self):
        '''
        For locate:
            Marks identified features with circles for a given frame
            (with the specific videofile & locate parameters written alongside)
        
        Returns
        -------
        fig : matplotlib figure
        '''
        fig, ax = plt.subplots()
        fig_color = 'mediumslateblue'
        
        tp.annotate(
            self.loc_df, self.frame, ax= ax,
            plot_style= {'markersize': 0.3}, color= fig_color
                    ) # ATTENTION: ideally 0.3 shoud be dependent on diameter
        
        # ax.set(yticks=[], xticks=[]) # removes numbers in axes
        ax.set_title(self.title, color= fig_color)
        
        ax.text(
            0.78,0.15, self.fig_txt_loc, color= fig_color,
            fontsize= 6, transform=plt.gcf().transFigure
                )
        
        plt.show()
        return fig
    
    def better_subpx_bias(self):
        '''
        For locate:
            If we use a mask size that is too small,
            the histogram often shows a dip in the middle -> bias.
            Ideally, the histogram should be uniform.
            (with the specific parameters written alongside)
        Based on: tp.subpx_bias
        
        Returns
        -------
        fig : matplotlib figure
        '''
        
        x_n_y = self.loc_df[ ['x', 'y'] ]
        subpx = lambda x: x % 1
        
        fig, axs = plt.subplots(1,2, sharey= True)
        x_n_y.applymap(subpx).hist(
            ax = axs, bins = int(self.bins_per_hist/2)
            )
        
        plt.suptitle(self.title)
        plt.text(
            0.92,0.15, self.fig_txt_loc, fontsize= 6,
            transform=plt.gcf().transFigure
                )
        
        plt.show()
        return fig
    
    def hist_scatter(self, xcol, ycol):
        '''
        For locate: 
            Scatters two quantities with its distributions as histograms at the sides
            (e.g: size vs mass).
            Useful when we want to eliminate spurious features from being located.
        Based on:
            https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html
        
        Parameters
        ----------
        xcol : pandas Series or array
        ycol : same type and size as xcol
            these are the series 

        Returns
        -------
        fig : matplotlib figure
        '''
        
        # Scattering data
        x = self.loc_df[xcol];  y = self.loc_df[ycol]
        
        try:
            xlabel = x.name; ylabel = y.name
        except AttributeError:
            xlabel = 'x'; ylabel = 'y'
        
        'Plot'
        fig = plt.figure(figsize=(9, 9))
        gs = fig.add_gridspec(
            2, 2,  width_ratios=(3, 1), height_ratios=(1, 3),
            left=0.1, right=0.9, bottom=0.1, top=0.9,
            wspace=0.05, hspace=0.05
                            )
        # Create the Axes
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        # Add details
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.grid()
        # Draw the scatter plot and marginals.
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)
        
        ax.scatter(x, y)
        ax_histx.hist(x, bins= self.bins_per_hist)
        ax_histy.hist(y, bins= self.bins_per_hist, orientation='horizontal')
        
        plt.text(
            0.71,0.71, self.fig_txt_loc, fontsize= 7,
            transform=plt.gcf().transFigure
                )
        
        plt.show()
        return fig
    
    def better_plot_traj(self, trck_df):
        '''
        For tracking:
            Plots traces of trajectories for each particle
            for a given frame as background image.

        Parameters
        ----------
        trck_df : Trajectory DataFrame (from tp.link)
        
        Returns
        -------
        fig : matplotlib figure
        '''
        fig, ax = plt.subplots()
        fig_color = 'black' #'mediumslateblue'
        fig_txt_trck = dict_to_str({
            'file': self.vid_name, **self.loc_pms, **self.trck_pms
            })
        
        tp.plot_traj(
            trck_df, ax= ax, superimpose= self.frame,
            plot_style= {'linewidth': 0.5}
            )
        
        # ax.set(yticks=[], xticks=[]) # removes numbers in axes
        ax.set_title(self.title, color= fig_color)
        ax.text(
            0.78,0.15, fig_txt_trck, color= fig_color,
            fontsize= 6, transform=plt.gcf().transFigure
                )
        
        plt.show()
        return fig
    
    def save_frame(self, fig):
        fig.savefig(self.path, dpi = self.dpi, bbox_inches='tight',
                    facecolor="white", transparent = False)


class Parameters_Tester:
    """
    Using a compilation of Analyse_Frame's figures, we create videos
    (gifs) that show the evolution of Trackpy's detection.
    
    Parameters
    ----------
    video_path : str
    test_title : str
    
    locate_parameters : dict
        all the tp.locate variables we want to work with.
    tracking_parameters : dict, optional
        all the tp.link variables we want & tp.filter_stubs's threshold.
    frame_range : tuple, optional
        (starting_frame, ending_frame): used when we want to make a smaller 
        gif, instead of one of the whole movie.
    """
    
    # Instance variables
    spf = 0.5               # gif's secs per frame
    folder_path = 'figs'    # where the gif will be stored
    
    def __init__(
            self, video_path, test_title, locate_parameters,
            tracking_parameter= None, frame_range= None
            ):
        # Video
        self.path   = video_path
        self.frames = as_grey( pims.open(self.path) )
        self.N_frames = len(self.frames)
        
        self.title = test_title
        # Trackpy parameters
        self.loc_pms  = locate_parameters
        self.trck_pms = tracking_parameter
        
        if isinstance(frame_range, tuple):
            n_i, n_f = frame_range
            
            # Subset of all frames
            self.frames = self.frames[n_i:n_f]
            self.N_frames = n_f - 1 # -1 since indexing stars at 0
    
    
    def for_locate(self, fig_key, fr_analyses = []):
        """
        Given a fig_key that selects what figure we want to do,
        it creates a gif of the plot.
        
        Parameters
        ----------
        fig_key : str
            Possible keys: 'annotate', 'subpx_bias', 'size_vs_mass',
            'ecc_vs_size' ...
        fr_analyses : list of Analyse_Frame classes, optional
        
        How to optimize computation?
        ----------------------------
        By default: we analyse each frame (with Analyse_Frame),
            this process is computational costly.
        Therefore: we skip this process if it was already done,
            by feeding fr_analyses as a non-empty list.
        """
        compute_analyses = (True if fr_analyses == [] else False)
        images_paths = []
        
        for i, frame in enumerate(self.frames):
            fr_title= f'Frame {i}'
            
            if compute_analyses:
                An_Fr = Analyse_Frame(self.path, frame, fr_title, self.loc_pms)
                print(f'Locate: Frame {i} out of {self.N_frames}')
                fr_analyses.append(An_Fr)
            else:
                An_Fr = fr_analyses[i]
                
            if fig_key == 'annotate':
                fig = An_Fr.better_annotate()
            
            elif fig_key == 'subpx_bias':
                fig = An_Fr.better_subpx_bias()
            
            elif '_vs_' in fig_key:
                y,x = fig_key.split('_vs_')
                fig = An_Fr.hist_scatter(xcol= x, ycol= y)
            else:
                s = f"fig_key: {fig_key} isn't available"
                raise NameError(s)
            
            images_paths.append(An_Fr.path)
            An_Fr.save_frame(fig)
        
        gif_name = f'{self.title} - {fig_key}'
        gif_path = f'{self.folder_path}//{gif_name}.gif'
        Gif_maker.gif_maker(images_paths, gif_path,  spf= self.spf)
        print(fig_key + ' gif created')
        Gif_maker.remove_imgs(images_paths)
        
        return fr_analyses
    
    def for_tracking(self):
        """
        better_plot_traj
        Concatenates frames of the movie, so we can check if 
        particles follow Trackpy's trajectories.
        """
        
        'Computation'
        
        # locate for multiple frames
        loc_df = tp.batch(self.frames, **self.loc_pms, invert= True, processes= 1)
        # loc_df.columns: 'y', 'x', 'mass', 'size', 'ecc', ..., 'frame'
        
        # We unpack tracking parameters to be passed to their respective func()
        link_pms  = {key: val for key, val in self.trck_pms.items() if key != 'threshold'}
        threshold = self.trck_pms['threshold']
        
        trck_df = tp.link(loc_df, **link_pms)
        # trck_df.columns: 'y', 'x', 'mass', 'size', 'ecc', ..., 'frame', 'particle'
        
        # Filter spurious/short trajectories
        filter_trck_df = tp.filter_stubs(trck_df, threshold)
        
        # Compare the number of particles in the unfiltered and filtered data.
        print( 'Before filtering: ', trck_df['particle'].nunique() )
        print( 'After:      ', filter_trck_df['particle'].nunique() )
        # .nunique(): number of distinct elements
        
        'Plotting trajectories'
        
        images_paths = []
        for i, frame in enumerate(self.frames):
            fr_title= f'Frame {i}'
            print(f'Frame {i} out of {self.N_frames}')
            
            Fr_An = Analyse_Frame(
                self.path, frame, fr_title, self.loc_pms, self.trck_pms
                )
            fig = Fr_An.better_plot_traj(filter_trck_df)
            
            images_paths.append(Fr_An.path)
            Fr_An.save_frame(fig)
        
        gif_name = f'{self.title} - plot_traj'
        gif_path = f'{self.folder_path}//{gif_name}.gif'
        Gif_maker.gif_maker(images_paths, gif_path,  spf= self.spf)
        print('plot_traj' + ' gif created')
        Gif_maker.remove_imgs(images_paths)



'--- --- --- --- --- --- --- INPUTS --- --- --- --- --- --- ---'
## Normal Swim 
path1 = "00_Experiments_685nm//020223//020123_NormalSwim.avi"
## UV flash (30s, 63s)
path2 = "00_Experiments_685nm//020223//020123_UV5percentSwim.avi"
## UV flash (30s, 3.5mins)
path3 = "02_Experiments_715nm//300523_715nmFilter_NODCMU//Area2_UV10_300523.avi"


locate_parameters = dict( # Values used for path1's video
    diameter = 21,              # 21 +/- 2 px
    minmass  = 000,             # 3000 +/- 1000
    maxsize  = None,            # None : not relevant
    separation = 21+1,          # Default == diameter + 1
    noise_size= 1               # Default == 1
    )    # diameter: gives the feature’s extent (odd number!)

tracking_parameters = dict(
    search_range= 18,
    memory= 3,
    threshold= 10,
    )   # threshold: minimum number of frames to survive filtering


'--- --- --- --- --- --- --- TESTING --- --- --- --- --- --- ---'

def Analyse_1_frame(path, frame_num):
    """
    Recomendation: use it before creating a gif so that we have an
    educated guess on the order of the parameters.
    ( no figures are saved! )
    """
    
    frames = as_grey( pims.open(path) )
    
    An_Fr = Analyse_Frame(
        path, frames[frame_num], f'Frame {frame_num}', locate_parameters
        )
    # print(An_Fr.loc_df.columns)
    
    '(un)comment the following lines to toggle the figs we want to see'
    f1 = An_Fr.better_subpx_bias()
    f2 = An_Fr.hist_scatter(xcol= 'mass', ycol= 'size')
    An_Fr.better_annotate()
    
    # An_Fr.save_frame(f1)
    An_Fr.save_frame(f2)
    return An_Fr
    
frame_num = [0, 100, 250, 451][2]
'(un)comment to deactivate:'
An_Fr = Analyse_1_frame(path2, frame_num)


def Multiple_frames(path, exp_label, frame_range):
    """
    Recomendation: first try a small frame_range
    
    To be able to save the gifs, the folders defined in the beginning
    of both classes have to exist.
    """
    # to distinguish diffent tests with the same video:
    test_title = f'{exp_label}_frames{frame_range} ' + \
                'd={diameter};m_min={minmass}'.format(**locate_parameters)    
                # (only {these parameters} will appear on the gif's name)
    
    Test = Parameters_Tester(
        path, test_title, locate_parameters,
        tracking_parameters, frame_range
        )
    
    
    '(un)comment the following lines to toggle the gifs we want to save'
    
    analyse_frames = [] # empty until passed through 'annotate'
    analyse_frames = Test.for_locate('annotate', analyse_frames)
    Test.for_locate('subpx_bias', analyse_frames)
    Test.for_locate('size_vs_mass', analyse_frames)
    
    Test.for_tracking()
    
    return Test

exp_label   = 'UV' # tag for recognising the experiment/video
frame_range = (0,40)

'(un)comment to deactivate gif generation:'
# Multiple_frames(path2, exp_label, frame_range)
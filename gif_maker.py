# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:29:09 2023

@author: Jorge Pottiez López-Jurado
"""

import imageio
import os


'''
Other useful OS functions:

os.remove()     # removes a file.
os.rmdir()      # removes a directory.
shutil.rmtree() # deletes a directory & all its contents.

# Create newpath
if not os.path.isdir(newpath):
    os.mkdir(newpath)
'''

def gif_maker(images_path_list, gif_path, spf= 0.5):
    """
    Creates gif concatenating images, given their paths as a list
    
    Parameters
    ----------
    images_path_list : list of str
    gif_path : str
    
    spf : float, optional
        Seconds per Frame == 1/fps, default: 0.5.
    """
    images = [imageio.imread(path) for path in images_path_list]
    # make the last frame last n x longer
    n = 2 
    images.extend( [images[-1]]*(n-1) )
    
    imageio.mimsave(gif_path, images, 'GIF', duration = spf)
    # duration is in seconds per frame (weird, innit?)

def remove_imgs(images_path_list):
    'Removes all files in the given list'
    for img_path in images_path_list:
        os.remove(img_path)


if __name__ == '__main__':
    'Create gif manually'
    # this part of the code won't be run
    # if this script is imported into another script
    
    temp_folder = 'figs//.temp_figs//'
    
    files = os.listdir(temp_folder) 
    paths = [temp_folder + f for f in files]
    
    title = 'd={diameter};m_min={minmass}'
    gif_maker(paths, title)
    remove_imgs(paths)
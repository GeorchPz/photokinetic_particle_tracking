# Photokinetic Particle Tracking

This repository contains tools for tracking, analysing, and visualizing particles in video recordings, with a focus on photokinetic response experiments.

## Overview

This project provides a comprehensive toolkit for:
- Detecting particles in video frames using Trackpy
- Tracking particles across multiple frames
- Computing kinematic properties (velocity, angles, acceleration)
- Analysing and visualizing particle movements and distributions
- Saving results in HDF5 format for further analysis

## Files and Modules

- `tracking_datasaver.py`: Core module for particle detection and tracking
- `kinematics_datasaver.py`: Calculates kinematic properties using spline fitting
- `kinematics_analysis.py`: Tools for analysing kinematic data
- `Determinate_parameters.py`: Tools for optimizing Trackpy parameters
- `particle_kinematics_checker.py`: Tool for verifying kinematic calculations
- `gif_maker.py`: Utility for creating GIF animations

## Dependencies

- Python 3.x
- NumPy
- Pandas
- Trackpy
- PIMS (Python Image Sequence)
- SciPy
- Matplotlib
- imageio

## How to Use

### Basic Workflow

1. **Parameter Determination**:
   ```python
   # Use Determinate_parameters.py to find optimal parameters
   Analyse_1_frame(video_path, frame_num)
   ```

2. **Particle Tracking**:
   ```python
   # Use tracking_datasaver.py to detect and track particles
   frames, f, t, t1 = main(locate_parameters, tracking_parameters, video_path)
   ```

3. **Kinematic Analysis**:
   ```python
   # Use kinematics_datasaver.py to calculate kinematic properties
   store = pd.HDFStore(h5_path)
   t1 = store.get('tracking_cut')
   k = System_Kinematics(t1, n=0.5).to_df()
   store.put('kinematics', k)
   store.close()
   ```

4. **Data Visualization**:
   ```python
   # Use kinematics_analysis.py to visualize results
   An_K = Analyse_Kinematics(kin_df, fig_title)
   traj = An_K.track_length_info()
   An_K.evolving_hist(kin_df.speed, step)
   ```

## Data Structure

Data is stored in HDF5 files with the following structure:
- `parameters`: Series containing tracking parameters
- `locating`: DataFrame with located particle positions
- `tracking`: Raw tracking data
- `tracking_cut`: Filtered tracking data
- `kinematics`: DataFrame with velocity and angle data

## Author

Jorge Pottiez López-Jurado
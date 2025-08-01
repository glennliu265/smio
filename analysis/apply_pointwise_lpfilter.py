#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 09:58:49 2025

@author: gliu
"""


import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import scipy as sp

import matplotlib.patheffects as PathEffects

import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time
import cmcrameri as cm

#%% 

# local device (currently set to run on Astraeus, customize later)
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import cvd_utils as cvd


#%% Indicate which file to load (by experiment name)

# Set Paths
figpath         = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20250507/"
sm_output_path  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/"
proc.makedir(figpath)

# Experiment Name
expname         = "SST_ERA5_1979_2024"

#%% Load the file

ds              = dl.load_smoutput(expname,sm_output_path)
lpfilter        = lambda x: proc.lp_butter(x,120,6)

#%% Anomalize and deseason

dsa   = proc.xrdeseason(ds)
dsadt = proc.xrdetrend(dsa.SST)


#%% Apply Low Pass Filter

#Runs in approx 160 sec
# Ran in 8.26 sec for ERA5
st         = time.time()
lpout = xr.apply_ufunc(
    lpfilter,
    dsadt,
    input_core_dims=[['time']],
    output_core_dims=[['time']],
    vectorize=True,
)
print("Function applied in in %.2fs" % (time.time()-st))

#%% 



#%% Compute the 






#%% Indicate what frequency to filter to

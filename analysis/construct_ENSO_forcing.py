#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Construct ENSO forcing

Following Claude's advice...
Reconstruct ENSO-based surface heat flux for the SST model
to retrieve the time-varying SST response to ENSO, which
can then be remove from the simulation.

Focus analysis on the SPGNE domain for the SMIO paper.

Copied upper section of [regress_enso_SPGNE]


Created on Wed Dec 17 09:37:14 2025

@author: gliu

"""


import sys
import time
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import xarray as xr
import sys
import glob 
import scipy as sp
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
from scipy.io import loadmat
import matplotlib as mpl

import importlib
from tqdm import tqdm

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

ensopath = "/Users/gliu/Downloads/02_Research/01_Projects/07_ENSO/03_Scripts/ensobase/"
sys.path.append(ensopath)
import utils as ut


#%% User Edits

# SST Files (Anomalized and Detrended)
exppath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/SST_ERA5_1979_2024/Output/"
expnc   = "SST_runid00.nc"

# ENSO Files
ensonc   = "ERA5_ensotest_ENSO_detrendGMSSTmon_pcs3_1979to2024.nc"
ensopath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/enso/"

# SST Files (from ENSO calculations)
noensonc = "ERA5_sst_detrendGMSSTmon_ENSOcmp_lag1_pcs3_monwin3_1979to2024.nc"

# Set Figure Path
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20251218/"
proc.makedir(figpath)

# Load GMSST For ERA5 Detrending
detrend_obs_regression  = True
dpath_gmsst             = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
nc_gmsst                = "ERA5_GMSST_1979_2024.nc"

bbox_spgne    = [-40,-15,52,62]

#%% Load relevant data

ds_sst        = xr.open_dataset(exppath + expnc).load()
ds_sst_noenso = xr.open_dataset(ensopath + noensonc).load()
ensoid        = xr.open_dataset(ensopath + ensonc).load()



# ========================================================
#%% Load ERA5 and preprocess
# ========================================================
# Copied largely from area_average_variance_ERA5

# Load SST
dpath_era   = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
ncname_era  = dpath_era + "ERA5_sst_NAtl_1979to2024.nc"
ds_era      = xr.open_dataset(ncname_era).sst.load()

# Load Flux
ncname_era_flx = dpath_era + "ERA5_qnet_NAtl_1979to2024.nc"
ds_era_flx = xr.open_dataset(ncname_era_flx).qnet.load()

# Load Mask
dsmask_era  = dl.load_mask(expname='ERA5')
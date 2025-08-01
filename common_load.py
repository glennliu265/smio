#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Central reference for loading things often used in smio scripts...

Note: this is for running on <Astraeus>

Created on Tue Jul 15 17:34:59 2025

@author: gliu
"""


#%% Modules

import time
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import xarray as xr
import sys
import tqdm
import glob 
import scipy as sp
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec

from scipy.io import loadmat
import matplotlib as mpl

#%% Custome Modules (Astraeus)

# local device
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import cvd_utils as cvd

#%% Load relevant variables (ERA5)

# Load GMSST 
dpath_gmsst     = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
nc_gmsst        = "ERA5_GMSST_1979_2024.nc"
ds_gmsst        = xr.open_dataset(dpath_gmsst + nc_gmsst).load()#.GMSST_MeanIce.load()
detrend_gm      = lambda ds_in: proc.detrend_by_regression(ds_in,ds_gmsst.Mean_Ice)

# Load GMSST (Older)
nc_gmsst_pre    = "ERA5_GMSST_1940_2078.nc"
ds_gmsst_pre    = xr.open_dataset(dpath_gmsst + nc_gmsst).load()#.GMSST_MeanIce.load()

ds_gmsst_merge  = xr.concat([ds_gmsst,ds_gmsst_pre],dim='time')

# Load ENSO
ensopath        = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/enso/"
ensonc          = "ERA5_ENSO_detrend1_pcs3_1979to2024.nc"
ds_enso         = xr.open_dataset(ensopath + ensonc)
ensoids         = ds_enso.pcs.load()


# Load Ice Masks
dsmask_era5     = dl.load_mask(expname='ERA5').mask
dsmaskplot      = xr.where(np.isnan(dsmask_era5),0,1)

# Load Land-Ice Mask


# Load SSH


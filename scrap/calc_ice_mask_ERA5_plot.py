#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate the Ice Mask for Plottting 

(using the 15% concentration of the median wintertime ice as advised by Martha)

Created on Fri Oct 17 14:21:10 2025

@author: gliu
"""


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
from scipy.io import loadmat

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

#%% Load ERA5 SST and Ice Cover

# Case for 1979 to 2024
dpath       = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
ncice       = dpath + "siconc_1979_2024.nc"#"ERA5_siconc_1940_2024_NATL.nc"
outname     = "%sERA5_GMSST_1979_2024.nc" % (dpath)
outname_ice = "%sERA5_IceMask_Global_1979_2024.nc" % (dpath)

# # Case for 1940 to 1979
# dpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/ERA5/"
# dpath_out = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
# ncice = dpath + "siconc_1940_1978.nc"
# outname  = "%sERA5_GMSST_1940_1978.nc" % (dpath_out)
# outname_ice = "%sERA5_IceMask_Global_1940_1978.nc" % (dpath)

#%% Load Sea Ice

# Load Sea Ice and Crop
ds_ice = xr.open_dataset(ncice).siconc.load()
ds_ice = ds_ice.rename(dict(valid_time='time'))

#%% Get Winter Months

selmon_nh = [12,1,2]
selmon_sh = [6,7,8]

winter_nh = proc.selmon_ds(ds_ice,selmon_nh)
winter_sh = proc.selmon_ds(ds_ice,selmon_sh)

#%% Compute the Median Sea Ice


median_nh     = winter_nh.median('time')
median_sh     = winter_sh.median('time')
median_winter = xr.where(median_nh > median_sh,median_nh,median_sh)
median_mask15 = xr.where( (median_winter > 0.15) | np.isnan(median_winter),np.nan,1)


ds_out = xr.merge([median_winter.rename("siconc_winter_median"),median_mask15.rename("median_mask_15pct")])
edict  = proc.make_encoding_dict(ds_out)
outname_ice = "%sERA5_IceMask_Global_1979_2024_Median15.nc" % (dpath)

ds_out.to_netcdf(outname_ice,encoding=edict)
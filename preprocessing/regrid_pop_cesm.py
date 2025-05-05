#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Regrid from POP to CAM Grid (CESM)
based on quick_regrid.py from reemergence module

Created on Mon May  5 16:44:43 2025

@author: gliu

"""

import time
import numpy as np
import xarray as xr
from tqdm import tqdm
import xesmf as xe
import sys


# -------------
#%% User Edits
# -------------

# Indicate Machine
machine       = "Astraeus"

# Indicate the Input Variable
varname     = "VVEL"
ncname      = "CESM1_HTR_VVEL_NATL_scycle.nc"
ncpath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LE/proc/NATL/"

# Indicate netcdf information for reference grid
ncref       = "CESM1LE_SST_NAtl_19200101_20050101_bilinear_stdev.nc"
ncref_path  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"


# Output Information
outname     = "CESM1_HTR_VVEL_NATL_scycle_regrid_bilinear.nc"
outpath     = ncpath

method      = "bilinear"  # regridding method

#%% Open Datasets

dstarg = xr.open_dataset(ncpath+ncname).load()
dsref  = xr.open_dataset(ncref_path + ncref).load()

# ---------------------------------
#%% Load Lat/Lon Universal Variable
# ---------------------------------

# Set up reference lat/lon
lat     = dsref.lat
lon     = dsref.lon
xx,yy   = np.meshgrid(lon,lat)

newgrid = xr.Dataset({'lat':lat,'lon':lon})

# ----------
#%% Set some additional settings based on user input
# ----------

# Rename Latitude/Longitude to prepare for regridding
ds          = dstarg.rename({"TLONG": "lon", "TLAT": "lat"})
oldgrid     = ds.isel(ens=0,z_t=0) # Make it 3D


# Set up Regridder
regridder   = xe.Regridder(oldgrid,newgrid,method,periodic=True)

# Regrid
st = time.time()
daproc = regridder(ds) # Need to input dataarray
print("Regridded in %.2fs" % (time.time()-st))


savename = outpath + outname
daproc.to_netcdf(savename,
                 encoding={varname: {'zlib': True}})

print("Saved output to %s" % savename)

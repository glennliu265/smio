#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Check how regridding impacts ERA5 Re-emergence
Works with output from coarse_grain_ERA5

Created on Wed Jun 18 14:48:13 2025

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
import matplotlib as mpl

#%% Load Custom Modules

# local device (currently set to run on Astraeus, customize later)
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import cvd_utils as cvd

#%% Load the data (original resolution)

# Load the SST
dpath       = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
nc1         = "ERA5_sst_NAtl_1979to2024.nc"
nc2         = "ERA5_sst_NAtl_1940to1978.nc"
ncs         = [nc2,nc1]

dsall       = [xr.open_dataset(dpath + nc).load() for nc in ncs]

dsraw       = xr.concat(dsall,dim='time')
sstraw      = dsraw.sst

# Load GMSST For ERA5 Detrending
detrend_obs_regression = True
dpath_gmsst     = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
nc_gmsst        = "ERA5_GMSST_1979_2024.nc"
ds_gmsst      = xr.open_dataset(dpath_gmsst + nc_gmsst).GMSST_MeanIce.load()

#%% Load Regridded versions

degs        = [1,2]
method      = 'bilinear'
ncnames     = ["%sERA5_sst_NAtl_1940to2024_regrid_%ideg_%s.nc" % (dpath,deg,method) for deg in degs]
dsregrids   = [xr.open_dataset(nc).load().sst for nc in ncnames]

#%% Prepare for preprocessing

dsin     = [sstraw,] + dsregrids
expnames = ["0.25$\degree$","1$\degree$","2$\degree$"]
expcols  = ["k","b",'cornflowerblue']

dsin     = [ds.sel(time=slice('1979-01-01','2024-12-31')) for ds in dsin]

#%% Preprocess

def preproc_ds(ds):
    dsa           = proc.xrdeseason(ds)
    dtout         = proc.detrend_by_regression(dsa,ds_gmsst)
    dtfin      = dtout.sst
    return dtfin

dsproc = [preproc_ds(ds) for ds in dsin]


#%% Compute Area Weighted Average over SPGNE, then compute the re-emergence and see how this is different

regsel = [-40,-15,52,62]
dsreg  = [proc.sel_region_xr(ds,regsel) for ds in dsproc]
dsaavg = [proc.area_avg_cosweight(ds) for ds in dsreg]

#%% See the differences in the timeseries

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(10,4.5))

for ii in range(3):
    ax.plot(dsaavg[ii],label=expnames[ii],c=expcols[ii])
ax.legend()

# ================================
#%% Try computing re-emergence
# ================================

lags    = np.arange(61)
sstin   = [ds.data for ds in dsaavg]
tsm     = scm.compute_sm_metrics(sstin,lags=lags)

#%% Try plotting the re-emergence

kmonth  = 6
xtks    = lags[::6]

fig,ax = plt.subplots(1,1,figsize=(10,4.5),constrained_layout=True)

ax,_ = viz.init_acplot(kmonth,xtks,lags,ax=ax)
for ii in range(3):
    plotvar = tsm['acfs'][kmonth][ii]
    ax.plot(lags,plotvar,label=expnames[ii],c=expcols[ii])
ax.legend()

#%% See variance (expect it to go own)

mons3  = proc.get_monstr()
fig,ax = viz.init_monplot(1,1,figsize=(10,4.5))
for ii in range(3):
    plotvar = tsm['monvars'][ii]
    ax.plot(mons3,plotvar,label=expnames[ii],c=expcols[ii])
ax.legend()

#%% Plot the power spectra


fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))

# for ii in range(3):
#     plotvar = tsm['monvars'][ii]
#     ax.plot(mons3,plotvar,label=expnames[ii],c=expcols[ii])
# ax.legend()


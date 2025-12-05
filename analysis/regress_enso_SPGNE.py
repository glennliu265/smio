#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Examine relationship of ENSO to SPGNE SST

Created on Wed Dec  3 14:12:29 2025

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

#%% Load relevant data

ds_sst        = xr.open_dataset(exppath + expnc).load()

ds_sst_noenso = xr.open_dataset(ensopath + noensonc).load()

ensoid        = xr.open_dataset(ensopath + ensonc).load()

#%% Preprocess

#%% Take Average Over SPGNE

bbox_spgne    = [-40,-15,52,62]

sstspg        = proc.sel_region_xr(ds_sst['SST'],bbox_spgne)
sstspg_noenso = proc.sel_region_xr(ds_sst_noenso['sst'],bbox_spgne)

#%% Note this turned out to be too raw

fig,ax        = plt.subplots(1,1,constrained_layout=True,figsize=(12.5,4.5))

plotvar       = proc.area_avg_cosweight(sstspg)

ax.plot(plotvar.time,plotvar,label="SST With ENSO")

# plotvar = proc.area_avg_cosweight(sstspg_noenso)
# ax.plot(plotvar.valid_time,plotvar,label="SST Without ENSO")

ax.legend()



# ========================================================
#%% Load ERA5 and preprocess
# ========================================================
# Copied largely from area_average_variance_ERA5

# Load SST
dpath_era   = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
ncname_era  = dpath_era + "ERA5_sst_NAtl_1979to2024.nc"
ds_era      = xr.open_dataset(ncname_era).sst.load()

# Load Flux
# ncname_era_flx = dpath_era + "ERA5_qnet_NAtl_1979to2024.nc"
# ds_era_flx = xr.open_dataset(ncname_era_flx).qnet.load()

# Load Mask
dsmask_era  = dl.load_mask(expname='ERA5')

#%% Load GMSST and also detrend pointwise

# Detrend by Regression
dsa_era       = proc.xrdeseason(ds_era)
#flxa_era      = proc.xrdeseason(ds_era_flx)

# Detrend by Regression to the global Mean
ds_gmsst      = xr.open_dataset(dpath_era + nc_gmsst).GMSST_MaxIce.load()
dtout         = proc.detrend_by_regression(dsa_era,ds_gmsst,regress_monthly=True)
sst_era       = dtout.sst
#dtout_flx     = proc.detrend_by_regression(flxa_era,ds_gmsst,regress_monthly=True)
#flx_era       = dtout_flx.qnet

#proc.printtime(st,print_str="Loaded and procesed data") #15.86s


sst_spgne = proc.sel_region_xr(sst_era,bbox_spgne)
spgne_aavg = proc.area_avg_cosweight(sst_spgne)


#%% Calculate Lead Lag relationship

pc1     = ensoid.pcs.isel(pc=0).data.flatten()
tcoords = dict(time=sst_spgne.time)
pc1     = xr.DataArray(pc1,coords=tcoords,dims=tcoords,name="enso")

lags    = np.arange(-36,37,1)


llcorr  = ut.calc_lag_regression_1d(sp.signal.detrend(pc1.data),spgne_aavg.data,lags,correlation=True)

llcorr  = np.array(llcorr)

#%%

xtks = np.arange(-24,26,2)
fig,ax = plt.subplots(1,1)
ax.plot(lags,llcorr,marker="o")

ax.axhline([0],ls='solid',lw=.55,c='k')
ax.axvline([0],ls='solid',lw=.55,c='k')

ax.set_xlim([-24,24])
ax.set_xticks(xtks)
ax.set_ylim([-.3,.3])

ax.set_xlabel("<-- SPGNE SST Leads | ENSO Leads -->")
ax.set_ylabel("Correlation")

#%% Calculate Lead Lag relationship with each point

ntime,nlat,nlon = sst_spgne.shape
rr = np.zeros((len(lags),nlat,nlon)) * np.nan
for o in range(nlon):
    for a in range(nlat):
        ptdata     = sst_spgne.isel(lon=o,lat=a).data
        rr[:,a,o]  = ut.calc_lag_regression_1d(sp.signal.detrend(pc1.data),ptdata,lags,correlation=True)
        
        
        

#%%


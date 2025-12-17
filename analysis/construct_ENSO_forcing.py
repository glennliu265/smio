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

#%% Additional Modules

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

# ========================================================
#%% User Edits
# ========================================================





# Load GMSST For ERA5 Detrending
detrend_obs_regression  = True
dpath_gmsst             = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
nc_gmsst                = "ERA5_GMSST_1979_2024.nc"

bbox_spgne              = [-40,-15,52,62]

# Set Figure Path
figpath  = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20251218/"
proc.makedir(figpath)


# ========================================================
#%% Load ERA5 and preprocess
# ========================================================
# Copied largely from area_average_variance_ERA5

# # Load SST
dpath_era   = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
# ncname_era  = dpath_era + "ERA5_sst_NAtl_1979to2024.nc"
# ds_era      = xr.open_dataset(ncname_era).sst.load()

# Load Flux
ncname_era_flx    = dpath_era + "ERA5_qnet_NAtl_1979to2024.nc"
ds_era_flx        = xr.open_dataset(ncname_era_flx).qnet.load()

# Load Fprime
dpath_era_fprime  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/reanalysis/proc/NATL_proc_obs/"
ncname_era_Fprime = dpath_era_fprime + "ERA5_Fprime_QNET_timeseries_QNETgmsst_nroll0_NAtl.nc"
ds_era_fprime     = xr.open_dataset(ncname_era_Fprime).Fprime.load().squeeze()

# Load ENSO Files
ensonc        = "ERA5_ensotest_ENSO_detrendGMSSTmon_pcs3_1979to2024.nc"
ensopath      = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/enso/"
ensoid        = xr.open_dataset(ensopath + ensonc).load()

# Load Mask
dsmask_era  = dl.load_mask(expname='ERA5')

#%% Remove Seasonal Cycle

# Function Moved to amv.proc
# def check_scycle(ds,tol=1e-6,verbose=True):
#     ds = ds.groupby('time.month').mean('time')
#     if np.any(np.abs(ds) > tol):
#         if verbose:
#             ds_flat      = np.abs(ds.data.flatten())
#             id_above_tol = np.where(ds_flat > tol)[0]
#             print("%i values above %.2e were detected in monthly mean!" % (len(id_above_tol),tol))
#             maxval       = np.nanmax(ds_flat[id_above_tol])
#             print("Maximum Value was %.2f" % maxval)
#         return True
#     return False

flxa    = proc.xrdeseason(ds_era_flx)
fprimea = proc.xrdeseason(ds_era_fprime)

proc.check_scycle(flxa)
proc.check_scycle(fprimea)

#check_scycle(ds_era_flx)
#check_scycle(ds_era_fprime)

#%% Detrend by Regression

# Detrend by Regression to the Global Mean SST
ds_gmsst        = xr.open_dataset(dpath_era + nc_gmsst).GMSST_MaxIce.load()
inflxs          = [flxa,fprimea]
flxnames        = ['qnet','Fprime']
flxs_detrended  = [proc.detrend_by_regression(inflxs[ii],ds_gmsst,regress_monthly=True)[flxnames[ii]] for ii in range(2)]

# ==================================================
#%% Find ENSO-related component, using 1-month lag regression
# ==================================================

# Indicate # of Modes
nmode   = 2
ensolag = 1 # Month to lag fluxes ahead of ENSO


rpattern_byflx = []
ensocomp_byflx = []

for flx in flxs_detrended:
    # Prepare Data (separate year and month)
    flx     = flx.transpose('lon','lat','time')
    nlon,nlat,ntime = flx.shape
    nyr     = int(ntime/12)
    flxrs   = flx.data.reshape(nlon,nlat,nyr,12)
     
    
    # Preallocate
    flx_enso = np.zeros((nmode,nyr-1,12,nlon,nlat)) * np.nan# Note, due to lag, drop first year...
    rpatterns = np.zeros((nmode,12,nlon,nlat)) * np.nan
    for N in range(nmode):
        
        pcin = ensoid.pcs.isel(pc=N)
        
        
        for im in tqdm(range(12)):
            
            pcmon = pcin.isel(month=im).data[:(nyr-ensolag)]
            flxin = flxrs[:,:,ensolag:,im]
            
            rout     = proc.regress2ts(flxin,pcmon,verbose=False)
            rpatterns[N,im,:,:] = rout.copy()
            
            ensocomp = rout[None,:,:] * pcmon[:,None,None]
            flx_enso[N,:,im,:,:] = ensocomp.copy()
    
    # coords1 = dict(pc=np.arange(nmode)+1,
    #                )
    
    ensocomp_byflx.append(flx_enso)
    rpattern_byflx.append(rpatterns)
    
# =======================================    
#%% Reshape and Replace in to DataArrays
# =======================================

ds_ensocomps = []
ds_rpatterns = []


times_shift = flxa.time[(12*ensolag):]

for ii in range(2):
    
    ensocomp    = ensocomp_byflx[ii]
    ensocomp    = ensocomp.reshape(nmode,(nyr-1)*12,nlon,nlat)
    
    coords1     = dict(pc=np.arange(nmode)+1,time=times_shift,lon=flxa.lon,lat=flxa.lat)
    ds_ensocomp = xr.DataArray(ensocomp,coords=coords1,dims=coords1,name=flxnames[ii])
    ds_ensocomps.append(ds_ensocomp.transpose('pc','time','lat','lon'))
    
    # Repeat for Regression Pattern
    rpattern    = rpattern_byflx[ii]
    coords2     = dict(pc=np.arange(nmode)+1,month=np.arange(1,13,1),lon=flxa.lon,lat=flxa.lat)
    ds_rpattern = xr.DataArray(rpattern,coords=coords2,dims=coords2,name=flxnames[ii])
    ds_rpatterns.append(ds_rpattern.transpose('pc','month','lat','lon'))


enso_patterns = xr.merge(ds_rpatterns)
enso_flxes    = xr.merge(ds_ensocomps)

# =========================================================
#%% Save Output, Formatted Like Stochastic Model Parameters
# =========================================================

outpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/enso/"

for ii in range(2):
    
    outname = "%sERA5_%s_ENSO_related_forcing_ensolag%i.nc" % (outpath,flxnames[ii],ensolag)
    edict    = proc.make_encoding_dict(enso_flxes[flxnames[ii]])
    enso_flxes[flxnames[ii]].to_netcdf(outname,encoding=edict)
    
    outname = "%sERA5_%s_ENSO_related_pattern_ensolag%i.nc" % (outpath,flxnames[ii],ensolag)
    enso_patterns[flxnames[ii]].to_netcdf(outname,encoding=edict)
    
#%% Just Check how it looks like over the SPGNE

spgneflx = proc.sel_region_xr(enso_flxes,bbox_spgne)
aavg     = proc.area_avg_cosweight(spgneflx)

#%%
aavg.isel(pc=0).Fprime.plot()
aavg.isel(pc=0).qnet.plot()


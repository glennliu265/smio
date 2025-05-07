#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# Investigate EN4


Created on Tue May  6 14:08:41 2025

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


#%%
ncpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/EN4/proc/"
ncname  = "EN4_concatenate_1900to2021_lon-80to00_lat00to65.nc"
ds      = xr.open_dataset(ncpath+ncname)

figpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20250507/"


#%% Load all the files

import glob

rawpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/EN4/"
ncsearch    = "%sEN.4.*.nc" % rawpath
nclist      = glob.glob(ncsearch)
nclist.sort()
# Drop last 8 years
nclist      = nclist[:1464] # (Until 2021)

# Open all the datasets
dsall       = xr.open_mfdataset(nclist,concat_dim='time',combine='nested')

# Flip longitude
temp        = dsall.temperature
temp180     = proc.lon360to180_ds(temp,lonname='lon')

# Limit to North Atlantic
bboxnatl    = [-80,0,0,65]
tempnatl    = proc.sel_region_xr(temp180,bboxnatl)

# Load and save
st          = time.time()
tempnatl    = tempnatl.load()
print("Loaded in %.2fs" % (time.time()-st))

#temp = temp.rename(dict(longitude='lon',latutude='lat'))

# Save output
outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/EN4/proc/"
outname = outpath + "EN4_3D_TEMP_NAtl_1900_2021.nc"
edict   = proc.make_encoding_dict(tempnatl)
tempnatl.to_netcdf(outname,encoding=edict)

#%% Focus on subpolar Gyre Box

bbsel = [-40,-15,52,62]
spgne = proc.sel_region_xr(tempnatl,bbsel)

#%% Preprocess

tempa    = proc.xrdeseason(spgne)
tempa_dt = proc.xrdetrend(tempa)

#%% Take an area weighted average

aavg = proc.area_avg_cosweight(tempa_dt)


z = tempa.depth
times = tempa.time

#%% Check Hovmuller Diagram

cints  = np.arange(-1,1.05,.05)
vmax   = 1
fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12.5,4))
#pcm    = ax.pcolormesh(times,z,aavg.T,cmap='cmo.balance',vmin=-vmax,vmax=vmax)
pcm  = ax.contourf(times,z,aavg.T,cmap='cmo.balance',levels=cints)
ax.set_ylim([0,500])
ax.invert_yaxis()

ax.set_xlim([times[-240],times[-1]])

#%% Check re-emergence structure

aavg         = aavg.sel(time=slice(None,'2021-12-31'))
ntime,ndepth = aavg.shape
nyr          = int(ntime/12)


refval       = aavg.sel(depth=slice(0,50)).mean('depth').data.reshape(nyr,12) #[time]
indata       = aavg.data.reshape(nyr,12,ndepth)

#%%
lags      = np.arange(61)
nlags     = len(lags)
depthcorr = np.zeros((12,nlags,ndepth)) * np.nan

for zz in tqdm.tqdm(range(ndepth)):
    
    for km in range(12):
        
        var1 = refval.T #[mon x year]
        var2 = indata[...,zz].T
        if np.any(np.isnan(var2)):
            continue
        acf  = proc.calc_lagcovar(var1,var2,lags,km+1,1,debug=False)
        
        depthcorr[km,:,zz]=acf
        
        
#%% Plot Depth vs Lag

mons3 = proc.get_monstr()
cints = np.arange(0,1.05,.05)
for kmonth in range(12):
    #kmonth = 1
    
    fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(10,4))
    
    cf = ax.contourf(lags,aavg.depth,depthcorr[kmonth,:,:].T,cmap='cmo.amp',levels=cints)
    cl = ax.contour(lags,aavg.depth,depthcorr[kmonth,:,:].T,
                    colors="k",levels=cints,linewidths=0.75)
    clb = ax.clabel(cl,fontsize=12)
    viz.add_fontborder(clb)
    
    ax.set_ylim([0,250])
    ax.set_xlim([0,36])
    ax.invert_yaxis()
    
    ax.set_title("EN4 SST Lag Correlation with Top 50 meters (%s, SPGNE)" % mons3[kmonth],fontsize=18)
    ax.set_xlabel("Lag from %s (Months)" % mons3[kmonth],fontsize=14)
    ax.set_ylabel("Depth (meters)",fontsize=14)
    
    ax.tick_params(labelsize=16)
    figname = "%sEN4_SPGNE_Depth_v_Lag_50m_ACF_mon%02i.png" % (figpath,kmonth+1)
    plt.savefig(figname,dpi=150,bbox_inches='tight')

#%%

uppernatl = tempnatl.sel(depth=slice(0,500))


#%% Get Mean Mixed-Layer Depth for the region




#ntime,ndepth,nlat,nlon = teampa




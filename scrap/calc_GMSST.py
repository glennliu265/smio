#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate GMSST

Created on Tue May 13 13:47:23 2025

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
dpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
ncsst = dpath + "sst_1979_2024.nc"
ncice = dpath + "siconc_1979_2024.nc"#"ERA5_siconc_1940_2024_NATL.nc"
outname  = "%sERA5_GMSST_1979_2024.nc" % (dpath)
outname_ice = "%sERA5_IceMask_Global_1979_2024.nc" % (dpath)

# Case for 1940 to 1979
dpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/ERA5/"
dpath_out = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
ncsst = dpath + "sst_1940_1978.nc"
ncice = dpath + "siconc_1940_1978.nc"
outname  = "%sERA5_GMSST_1940_1978.nc" % (dpath_out)
outname_ice = "%sERA5_IceMask_Global_1940_1978.nc" % (dpath)

# Load SST
ds_sst = xr.open_dataset(ncsst).sst.load()#.sst


# Load Sea Ice and Crop
ds_ice = xr.open_dataset(ncice).siconc.load()
#ds_ice = ds_ice.sel(time=slice('1979-01-01',None))


#ds_sst = 
ds_in = [ds_sst,ds_ice]

#%% Format DS

latname     = 'latitude'
lonname     = 'longitude'
timename    = 'valid_time'
ds_out      = [proc.format_ds(ds,latname=latname,lonname=lonname,timename=timename) for ds in ds_in]
#ds_sst = proc.format_ds(ds_sst,latname=latname,lonname=lonname,timename=timename)
#ds_ice = proc.format_ds(ds_ice,latname=latname,lonname=lonname,timename=timename)

#%% Make an ice mask

maskmax         = xr.where(ds_out[1].max('time') > 0.05,np.nan,1)
maskmax         = xr.where(np.isnan(ds_out[0].isel(time=0)),np.nan,maskmax)

maskmean        = xr.where(ds_out[1].mean('time') > 0.05,np.nan,1)
maskmean        = xr.where(np.isnan(ds_out[0].isel(time=0)),np.nan,maskmean)

# Take mean by season, and largest mean across the 4 seasons (to get winter hemispheres)
season_mean     = ds_out[1].groupby('time.season').mean('time')
maskmean_winter = xr.where(season_mean.max('season')>0.05,np.nan,1) # Find maximum extent by season
maskmean_winter = xr.where(np.isnan(ds_out[1].isel(time=0)),np.nan,maskmean_winter)



#%% Save Ice Masks

maskmax         = maskmax.rename("Max_Mask")
maskmean        = maskmean.rename("Mean_Mask")
maskmean_winter = maskmean_winter.rename("Winter_Mean_Mask")

ds_icemasks     = xr.merge([maskmax,maskmean,maskmean_winter])

edict           = proc.make_encoding_dict(ds_icemasks)
ds_icemasks.to_netcdf(outname_ice,encoding=edict)


#%% Take Area Averages

# Apply Mask
inssts      = [ds_out[0],ds_out[0] * maskmax,ds_out[0] * maskmean,ds_out[0]*maskmean_winter]

# Compute GMSST
aavgs       = [proc.area_avg_cosweight(ds) for ds in inssts]

# DeSeason
aavgs_ds    = [proc.xrdeseason(ds) for ds in aavgs]

#%% Set names

expnames = ["No Mask","Mask Max","Mask Mean","Mask Winter Mean"]
expcols  = ["cyan","red","midnightblue","orange"]
els      = ['solid',"dashed",'dotted','solid']
nexps    = len(inssts)

#%% Plot the different GMSST

times = aavgs_ds[0].time
times = [str(t.data)[:4] for t in times]

xtks  = np.arange(len(times))
fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,3.5))

for ex in range(nexps):
    if ex == 3:
        lw = 1
    else:
        lw = 2
    ax.plot(xtks,aavgs_ds[ex],label=expnames[ex],c=expcols[ex],ls=els[ex],lw=lw)

ax.legend()

ax.set_ylim([-0.75,0.75])
ax.axhline([0],c="k",lw=0.55)
ax.set_xlim([xtks[0],xtks[-1]])
ax.set_xticks(xtks[::60],labels=times[::60])
ax.set_ylabel("GMSST (degC)")


#%% Save Output

expnames_out    = ["GMSST_ALL","GMSST_MaxIce","GMSST_MeanIce","GMSST_WinterMeanIce"]
dsout           = [aavgs_ds[ex].rename(expnames_out[ex]) for ex in range(4)]

dsout           = xr.merge(dsout)


edict           = proc.make_encoding_dict(dsout)
dsout.to_netcdf(outname,encoding=edict)


#%%

#%%
# Load Ice Mask
ds_mask = dl.load_mask(expname="ERA5").mask
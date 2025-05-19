#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Preprocess Fluxes for CESM2 Hierarchy
as processed by crop_natl_CESM2

Copied upper section of cesm2_hierarchy_v_obs

Created on Mon May 19 16:59:23 2025

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


#%% User Edits

# Indicate Paths (processed output by crop_natl_CESM2.py)
dpath           = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/NAtl/"

# Load GMSST For ERA5 Detrending
detrend_obs_regression = True
dpath_gmsst     = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
nc_gmsst        = "ERA5_GMSST_1979_2024.nc"

# Set Figure Path
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20250523/"
proc.makedir(figpath)

# For simplicity, load ice mask (can compare this later)
dsmask = dl.load_mask(expname='cesm2 pic').mask
dsmask180_cesm = proc.lon360to180_xr(dsmask)


#%% Functions

def calc_stds_sample(aavgs):
    
    aavgs_lp = [proc.lp_butter(aavg,120,6) for aavg in aavgs]
    stds     = np.array([np.nanstd(ss) for ss in aavgs])#np.nanstd(np.array(aavgs),1)
    stds_lp  = np.array([np.nanstd(ss) for ss in aavgs_lp])# np.nanstd(np.array(aavgs_lp),1)
    vratio   = stds_lp/stds * 100
    return aavgs_lp,stds,stds_lp,vratio



#%% Process for each case separately. Load ERA5 (downwards positive) for reference

# =========================================================
#%% First, let's do SOM (need to add everything separately)
# =========================================================

vnames = ["FSNS","FLNS","LHFLX","SHFLX"]

dsall = []
for vv in range(4):
    vname = vnames[vv]
    ncsearch = "%sCESM2_SOM_*%s*.nc" % (dpath,vname)
    nclist   = glob.glob(ncsearch)
    print(nclist[0])
    ds       = xr.open_dataset(nclist[0])[vname]
    dsall.append(ds)
    
#test = [proc.check_flx(ds) for ds in dsall] # Checks for positive upwards...

dsall = xr.merge(dsall)
#%% Plot to check the sign (NTS: FSNS is positive downwards, rest is positive upwards)

fig,axs = plt.subplots(1,4,figsize=(24,4.5),constrained_layout=True)
for vv in range(4):
    ax = axs[vv]
    plotvar = dsall[vnames[vv]]
    plotvar.isel(time=-1).plot(ax=ax,vmin=-120,vmax=120,cmap='cmo.balance')
    
#%% Calculate Qnet

ds_qnet = dsall.FSNS - (dsall.FLNS + dsall.LHFLX + dsall.SHFLX)
ds_qnet = ds_qnet.rename("SHF")
edict = proc.make_encoding_dict(ds_qnet)
outname = nclist[0].replace('SHFLX',"SHF")
ds_qnet.to_netcdf(outname,encoding=edict)


dsall.close()
ds_qnet.close()
#%% Check how this matches with the other simulations


st        = time.time()
cesmnames = ["CESM2_SOM","CESM2_POM3","CESM2_FCM"]
vnames    = [""]
ncesm     = len(cesmnames)
dscesms   = []

for cc in tqdm.tqdm(range(ncesm)):
    
    cname    = cesmnames[cc]
    ncsearch = "%s%s*_SHF_*.nc" % (dpath,cname)
    nclist   = glob.glob(ncsearch)
    print(nclist[0])
    
    ds       = xr.open_dataset(nclist[0]).SHF
    
    if "SOM" in cesmnames[cc]:   # Drop first 60 years
        ds = ds.sel(time=slice('0061-01-01',None))
    elif "POM" in cesmnames[cc]: # Drop first 100 years
        ds = ds.sel(time=slice('0100-01-01',None))
    elif "FCM" in cesmnames[cc]: # Drop first 200 years
        ds = ds.sel(time=slice('0200-01-01',None))
    
    
    dscesms.append(ds.load())
    
    
#%% Load Qnet from ERA5
pathera5    = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
ncera5      = pathera5 + "ERA5_qnet_NAtl_1979to2024.nc"
dsera5      = xr.open_dataset(ncera5).qnet.load()



#%%
flxall = dscesms + [dsera5,]



    
scycle          = [ds.groupby('time.month').mean('time') for ds in flxall]


expnames_short  = ["SOM","PenOM","FCM","ERA5"]
expnames_long   = ["Slab Ocean Model (60-360)","Pencil Ocean Model (100-400)","Fully Coupled Model (200-2000)","ERA5 (1979-2024)"]

nexps = len(expnames_short)
expcols         = ['violet','forestgreen','cornflowerblue','k']
expcols_bar     = ['violet','forestgreen','cornflowerblue','gray']




#%% Look at seasonal cycle at a point

lonf   = -30
latf   = 50
mons3  = proc.get_monstr()
fig,ax = viz.init_monplot(1,1)


for ex in range(nexps):
    plotvar = proc.selpt_ds(scycle[ex],lonf,latf)
    if ex < 3:
        plotvar = np.roll(plotvar.data,0)
                            
    ax.plot(mons3,plotvar,
            c=expcols[ex],label=expnames_long[ex])
    
ax.legend()

# ========================================================
#%% Load TS, CESM2
# ========================================================



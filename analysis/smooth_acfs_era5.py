#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Try applying simple smoothing to the ACFs

- Load ERA5 ACFs
- Apply smoothing
- Visualize differences


Notes
- Copied upper section of viz t2 rei era5


Created on Fri Apr 18 13:32:59 2025

@author: gliu

"""




import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import scipy as sp

import matplotlib.patheffects as PathEffects

import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time

#%% Load custom modules

# local device (currently set to run on Astraeus, customize later)
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import cvd_utils as cvd

#%% Plotting Inputs

# Set Plotting Options
darkmode = False
if darkmode:
    dfcol = "w"
    bgcol = np.array([15,15,15])/256
    sp_alpha = 0.05
    transparent = True
    plt.style.use('dark_background')
    mpl.rcParams['font.family']     = 'Avenir'
else:
    dfcol = "k"
    bgcol = "w"
    sp_alpha = 0.75
    transparent = False
    plt.style.use('default')

bboxplot    = [-80, 0, 20, 65]
mpl.rcParams['font.family'] = 'Avenir'
mons3       = proc.get_monstr(nletters=3)

fsz_tick    = 18
fsz_axis    = 20
fsz_title   = 16

rhocrit     = proc.ttest_rho(0.05, 2, 86)

proj        = ccrs.PlateCarree()

#%% Load some variables for plotting

# Load Sea Ice Masks
dpath_ice = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/reanalysis/proc/NATL_proc_obs/"
nc_masks = dpath_ice + "OISST_ice_masks_1981_2020.nc"
ds_masks = xr.open_dataset(nc_masks).load()

# Load AVISO
dpath_aviso     = dpath_ice + "proc/"
nc_adt          = dpath_aviso + "AVISO_adt_NAtl_1993_2022_clim.nc"
ds_adt          = xr.open_dataset(nc_adt).load()
cints_adt       = np.arange(-100, 110, 10)

#%% Further User Edits (Set Paths, Load other Data)

# Set Paths
figpath         = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20250318/"
sm_output_path  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/"
procpath        = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"



# Indicate Experiment and Comparison Name 
comparename     = "kgm20250417"
expnames        = ["SST_Obs_Pilot_00_Tdcorr0_qnet","ERA5_1979_2024"]
expnames_long   = ["Stochastic Model (with Re-emergence)","ERA5 Reanalysis (1979-2024)"]
expcols         = ["turquoise","k"]
expls           = ["dotted",'solid']

#%% Load ACFs

ncname_acf_obs  = "ERA5_NAtl_1979to2024_lag00to60_ALL_ensALL.nc"
ncname_acf_sm   = "%sSM_%s_lag00to60_ALL_ensALL.nc" % (procpath,expnames[0])
    
# Load Obs. ACFs
ds_obs = xr.open_dataset(procpath + ncname_acf_obs).load().isel(ens=0)
#ds_all.append(ds_obs)

ds_sm  = xr.open_dataset(ncname_acf_sm).load()


#%% Test it out with a point

lonf = -30
latf = 50
kmonth = 1

acfraw = proc.selpt_ds(ds_obs,lonf,latf).squeeze().isel(mons=kmonth).acf

acfsm  = proc.selpt_ds(ds_sm,lonf,latf).squeeze().isel(mons=kmonth).acf.mean('ens')

# def smoother(ts, winlen):
    
#     return np.convolve(ts, np.ones(winlen), 'same') / winlen

acfsmooth = acfraw.rolling(lags=3,center=True,).mean()#smoother(acfraw.data,2)

#acfsmooth2 = proc.selpt_ds(ds_obs_smooth,lonf,latf).squeeze().isel(mons=kmonth).acf

#%%

lags   = np.arange(61)
xtks   = lags[::6]

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))
ax,_   = viz.init_acplot(kmonth,xtks,lags,ax=ax)

ax.plot(lags,acfraw.data,label='raw')
ax.plot(lags,acfsmooth2,label='smooth')

ax.plot(lags,acfsm,label='stochastic model')
ax.legend()

#%% Try applying it to the whole array

ds_obs_smooth = ds_obs.rolling(lags=3,center=True,).mean()


ds_obs_smooth.acf.isel(lags=0) = 1

ds_obs_smooth_new = xr.where(ds_obs_smooth.lags == 0,1,ds_obs_smooth)

acfsmooth2 = proc.selpt_ds(ds_obs_smooth_new,lonf,latf).squeeze().isel(mons=kmonth).acf


#%% Save the new smoothed ACFs for ERA5


ncname_acf_obs_out  = procpath + "ERA5_NAtl_1979to2024_lag00to60_ALL_ensALL_smoothwin03.nc"
acf_out             = ds_obs_smooth_new.acf
edict               = proc.make_encoding_dict(acf_out)
acf_out.to_netcdf(ncname_acf_obs_out,encoding=edict)


#%% Compute the smoothed REI (copied from reemergence/calcualtions/calc_remidx_general)


acf_out = acf_out.rename(dict(lags='lag'))
def calc_rei(x): return proc.calc_remidx_xr(x, return_rei=True)



# Apply looping through basemonth, lon, lat. ('lon', 'lat', 'mon', 'rem_year')
rei_mon = xr.apply_ufunc(
    calc_rei,
    acf_out,
    input_core_dims=[['lag']],
    output_core_dims=[['rem_year',]],
    vectorize=True,
)

print("Function applied in in %.2fs" % (time.time()-st))

# Add numbering based on the re-emergence year
rei_mon['yr'] = np.arange(1, 1+len(rei_mon.yr))

# Formatting to match output of [calc_remidx_CESM1]
#rei_mon = rei_mon.rename({'rem_year': 'yr'})
rei_mon = rei_mon.rename('rei')
rei_mon = rei_mon.squeeze()
rei_mon = rei_mon.rename(dict(mons='mon'))
if 'ens' in dimnames:
    rei_mon = rei_mon.transpose('mon', 'yr', 'ens', 'lat', 'lon')
else:
    rei_mon = rei_mon.transpose('mon', 'yr', 'lat', 'lon')
edict = {'rei': {'zlib': True}}


outpathera5exp  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/ERA5_1979_2024/Metrics/"
outname         = "%sREI_Pointwise_smoothwin03.nc" % outpathera5exp


# Save the output
rei_mon.to_netcdf(outname, encoding=edict)
print("Saved output to: \n\t%s" % (outname))

#%% Visualize the REI

kmonth  = 1
yr      = 5

proj    =    ccrs.PlateCarree()
cmap        = 'cmo.deep_r'
fig,ax,bbb = viz.init_regplot()

plotvar     = rei_mon.isel(mon=kmonth,yr=yr)
pcm         = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                            vmin=0,vmax=1,cmap=cmap)
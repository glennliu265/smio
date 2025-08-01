#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Sanity Check:
    How does GMSST removal impacts conditions at a point
    where the regression to the global mean is strong?
    
Copied upper section of era5_acf_sensitivity analysis

Created on Thu Jul 24 14:56:12 2025

@author: gliu

"""


from tqdm import tqdm
from scipy import signal

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr

import sys
import cmocean
import time
import glob

#%% Import Packages

amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import cvd_utils as cvd

#%% Indicate Paths

figpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20250730/"
datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/ERA5/"
proc.makedir(figpath)

#%%

# Load Sea Ice Masks (ERA5)
dpath_ice   =  "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/ERA5/processed/"
nc_masks    = dpath_ice + "icemask_era5_global180_max005perc.nc"#"/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/reanalysis/proc/NATL_proc_obs/"
ds_masks    = xr.open_dataset(nc_masks).load()

# Load AVISO
dpath_adt   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/reanalysis/proc/NATL_proc_obs/proc/"
dpath_aviso = dpath_adt
nc_adt = dpath_aviso + "AVISO_adt_NAtl_1993_2022_clim.nc"
ds_adt = xr.open_dataset(nc_adt).load()
cints_adt = np.arange(-100, 110, 10)

#%% Load GMSST for Detrending (from common_load)

# Load GMSST 
dpath_gmsst     = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
nc_gmsst        = "ERA5_GMSST_1979_2024.nc"
ds_gmsst        = xr.open_dataset(dpath_gmsst + nc_gmsst).load()#.GMSST_MeanIce.load()
detrend_gm      = lambda ds_in: proc.detrend_by_regression(ds_in,ds_gmsst.Mean_Ice)

# Load GMSST (Older)
nc_gmsst_pre    = "ERA5_GMSST_1940_1978.nc"
ds_gmsst_pre    = xr.open_dataset(dpath_gmsst + nc_gmsst_pre).load()#.GMSST_MeanIce.load()

ds_gmsst_merge = xr.concat([ds_gmsst_pre,ds_gmsst],dim='time')

#%% Load the data

vnames = ["sst","siconc"]
ds_all = []
for vv in range(len(vnames)):
    nc_names = glob.glob(datpath + vnames[vv] + "*.nc")
    ds_load = xr.open_mfdataset(nc_names,concat_dim='valid_time',combine='nested').load()
    ds_all.append(ds_load)

latname     = "latitude"
lonname     = "longitude"
timename    = "valid_time"

ds_all      = [proc.format_ds(ds,latname=latname,lonname=lonname,timename=timename) for ds in ds_all]

ds_all      = xr.merge(ds_all)


ds_sst  = ds_all.sst
mons3   = proc.get_monstr()

#%% Regress to global mean and see where it is strong

# Select Region and Timeperiod
bbox_natl   = [-80,0,0,65]
tstart      = '1979-01-01'
tend        = '2024-12-31'
sst_in      = ds_all.sst.sel(time=slice(tstart,tend))
sst_in      = proc.sel_region_xr(sst_in,bbox_natl)

# Remove Seasonal Mean
ssta        = proc.xrdeseason(sst_in)

ssta.isel(time=0).plot()


# Isolate Global Mean
gmsst_in = ds_gmsst_merge.sel(time=slice(tstart,tend)).GMSST_MaxIce
#gmsst_in = gmsst_in / np.nanstd(gmsst_in)

#%% Detrend and get components

dtout = proc.detrend_by_regression(ssta,gmsst_in)


rpat  = dtout['regression_pattern']

rsig = dtout['sigmask']


#%% Let's look at the regression pattern

proj      = ccrs.PlateCarree()
fig,ax,bb = viz.init_regplot()

lonf      = -73
latf      = 37

# lonf      = -48
# latf      = 43

lonf = -30
latf = 50

locfn,loctitle = proc.make_locstring(lonf,latf)

vmax      = 6
plotvar   = rpat
pcm       = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                          vmin=-vmax,vmax=vmax,cmap='cmo.balance')

ax.plot(lonf,latf,transform=proj,marker="x",markersize=33,c='k')


plotvar = rsig
xx      = viz.plot_mask(plotvar.lon,plotvar.lat,plotvar.T,markersize=.1,
                        proj=proj,geoaxes=True,ax=ax)


cb        = viz.hcbar(pcm,ax=ax)

figname = "%sGMSST_Removal_Check_%s_loc.png" % (figpath,locfn)
plt.savefig(figname,dpi=150)



#%% Plot the histogram


title = "Regression Coeff @ %s: %.2f degC" % (loctitle,regpt)

betasel = proc.selpt_ds(rpat,lonf,latf)
regpt = betasel.data.item()

rpat_in = xr.where(rpat==0.,np.nan,rpat)
rpat_flatten = rpat_in.data.flatten()

fig,ax=plt.subplots(1,1,figsize=(4.5,4.5))

ax.hist(rpat_flatten,bins=25)
ax.set_xlim([-6,6])
#ax.set_xlim([-1.5,1.5])
ax.set_xlabel("GMSST Regression Coefficients (deg C)")

mu = np.nanmean(rpat.data.flatten())

ax.axvline(regpt,color='r')
ax.set_title(title)

figname = "%sGMSST_Removal_Check_%s_hist.png" % (figpath,locfn)
plt.savefig(figname,dpi=150)



#%% Check the regression

fsz_title  = 24
fsz_tick   = 14

plotvars    = [ssta,dtout.sst,dtout.fit]
plotnames   = ["Raw","Detrended","Fit","faketrend"]
plotcols    = ['gray','k','red',"blue"]
plotvarspt  = [proc.selpt_ds(p,lonf,latf) for p in plotvars]


fig,ax = plt.subplots(1,1,figsize=(12.5,4.5))

for ii in range(3):
    plotvar = plotvarspt[ii]
    ax.plot(plotvar.time,plotvar,label=plotnames[ii],c=plotcols[ii])
    
    
ax.set_xlim([plotvar.time[0],plotvar.time[-1]])

# Sanity check to make sure it adds up    
#ax.plot(plotvar.time,plotvarspt[1] + plotvarspt[2],ls='dashed',color='yellow',label="add")

# Also plot GMSST (kinda small... all things considered)
#plotvar = gmsst_in
#ax.plot(plotvar.time,plotvar,ls='dashed',color='blue',label="GMSST")

betasel = proc.selpt_ds(rpat,lonf,latf)
ax.legend(fontsize=fsz_tick,ncol=3)
title = "Trend_Removal @ %s, Regression Coeff=%.2f" % (loctitle,regpt)
ax.set_title(title,fontsize=fsz_title)

ax.set_ylabel("SST [degC]")
ax.tick_params(labelsize=fsz_tick)

ax.axhline([0],ls='dashed',c='gray',lw=.55)

#cb.set_label()


figname = "%sGMSST_Removal_Check_%s_regression.png" % (figpath,locfn)
plt.savefig(figname,dpi=150)

#%% Check impact on re-emergence

lags = np.arange(61)
plotvars_arr = [ds.data for ds in plotvarspt] + [plotvarspt[1].data+faketrend]
mets = scm.compute_sm_metrics(plotvars_arr,lags=lags,nsmooth=2,detrend_acf=False)


#%% Plot ACF

kmonth  = 2
xtks    = lags[::3]

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,4))
ax,_   = viz.init_acplot(kmonth,xtks,lags,ax=ax,title="")

for ii in range(3):
    plotvar = mets['acfs'][kmonth][ii]
    
    ax.plot(lags,plotvar,
            label=plotnames[ii],color=plotcols[ii],lw=2.5)

ax.legend()

figname = "%sGMSST_Removal_Check_%s_acf.png" % (figpath,locfn)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%%



def make_linear_trend(beta,noise,ntime):
    return beta*np.arange(ntime) + np.random.normal(0,noise)

beta  = .001
noise = 0.1
faketrend = make_linear_trend(beta,noise,len(gmsst_in))
    
    
    






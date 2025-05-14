#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check effect of including/excluding Ice cover on NASST/AMV Index

Investigate a few things
- 
- Removing regression of the global mean

Created on Tue May 13 10:37:36 2025

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

dpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"

ncsst = dpath + "ERA5_sst_NAtl_1979to2024.nc"
ncice = dpath + "ERA5_siconc_1940_2024_NATL.nc"


# Load SST
ds_sst = xr.open_dataset(ncsst).sst.load()#.sst


# Load Sea Ice and Crop
ds_ice = xr.open_dataset(ncice)
ds_ice = ds_ice.sel(time=slice('1979-01-01',None)).siconc

# Load Ice Mask
ds_mask = dl.load_mask(expname="ERA5").mask


# Load GMSST
ncgmsst     = "ERA5_GMSST_1979_2024.nc"
ds_gmsst    = xr.open_dataset(dpath + ncgmsst)

# ======================================================
# | Part 1. Setting up sensitivity to mask application |
# ======================================================
#%% Set up Experiments, Apply Mask

bbox_amv    = [-80,0,20,60]
bbname      = "NNNAT"
bbfn,bbstr  = proc.make_locstring_bbox(bbox_amv)
sst_masked  = ds_sst * ds_mask
inssts      = [ds_sst,sst_masked,]
expnames    = ["Raw","with Ice Mask"]

nexps       = len(expnames)

expcols     = ['skyblue','forestgreen']

#%% Calculate AMV Index

natlbox     = [proc.sel_region_xr(sst,bbox_amv) for sst in inssts]
aavgs       = [proc.area_avg_cosweight(sst) for sst in natlbox]


#%% Deseason and do a bunch of additional things 

def xrdetrend(ds,order,return_model=False):
    ntime = len(ds.time)
    x     = np.arange(ntime)
    y     = ds.data
    ydetrended,model=proc.detrend_poly(x,y,order)
    dsout  = xr.DataArray(ydetrended,coords=dict(time=ds.time),dims=dict(time=ds.time),name=ds.name)
    if return_model:
        dsfit = xr.DataArray(model,coords=dict(time=ds.time),dims=dict(time=ds.time),name='fit')
        dsout = xr.merge([dsout,dsfit])
    return dsout
    
    

sstas       = [proc.xrdeseason(sst) for sst in aavgs]
sstas_dt    = [proc.xrdetrend_1d(ds.rename('sst'),3,return_model=True) for ds in sstas]
sstas_dt_lp = [proc.lp_butter(ds.sst,120,6) for ds in sstas_dt]

#%% Plot Detrended Results

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))
for ii in range(2):
    ax.plot(sstas_dt[ii].sst,label="%s Raw" % expnames[ii])
    ax.plot(sstas_dt[ii].fit,label="%s Fit" % expnames[ii])
ax.legend()

#%% Examine Impacts on Variance

instd       = [np.nanstd(ds.sst) for ds in sstas_dt]
instd_lp    = [np.nanstd(ds) for ds in sstas_dt_lp]

#dofs        = []

vratio      = np.array(instd_lp) / np.array(instd) * 100

xlabs       = ["ERA5 %s\n%.2f" % (expnames[ii],vratio[ii])+"%" for ii in range(len(vratio))]
fig,ax      = plt.subplots(1,1,constrained_layout=True,figsize=(6,6))

braw        = ax.bar(np.arange(nexps),instd,color=expcols)
blp         = ax.bar(np.arange(nexps),instd_lp,color='k')

ax.bar_label(braw,fmt="%.04f",c='gray')#color=expcols)#c='gray')
ax.bar_label(blp,fmt="%.04f",c='k')

# Trying this out, but still havent gotten this working
# dofs    = []
# dofs_lp = []

# cfs     = []
# cfs_lp  = []
# for ii in range(nexps):
    
#     invar = sstas_dt[ii].sst
#     ntime = len(invar.data)
#     neff  = proc.calc_dof(invar)
#     dofs.append(neff)
    
#     bnds = proc.calc_confspec(0.05,neff)
#     bnds = np.array(bnds) * instd[ii] #- instd[ii]
#     cfs.append(bnds)
    
#     invar = sstas_dt_lp[ii]
#     ntime = len(invar.data)
#     neff  = proc.calc_dof(invar)
#     dofs_lp.append(neff)
    
#     bnds = proc.calc_confspec(0.05,neff)
#     bnds = np.array(bnds) * instd_lp[ii] #- instd_lp[ii]
#     cfs_lp.append(bnds)

# ax.errorbar(np.arange(nexps),instd,yerr=np.array(cfs).T,barsabove=True,
#             linestyle='none',lw=0.55,c='k')


#ax.errorbar(np.arange(nexps),instd,xerr=np.array(bnds[0]).T)
#ax.bar_label(vratio,fmt="%.2f",c=w,label_type='bottom')

ax.set_xticks(np.arange(nexps),labels=xlabs)
ax.set_ylabel("$\sigma$(SST) [$\degree$C]")
ax.set_ylim([0,1.0])
ax.set_title("%s (%s)" % (bbname,bbstr))

ax.grid(True,ls='dotted',lw=0.55,c='gray')

#%% Detrend using GMSST

insst    = sstas[1] # With Mask
gmsst_in = ds_gmsst.GMSST_MaxIce




varraw = insst.data[None,None,:]

outdict        = proc.regress_ttest(varraw,gmsst_in.data)
beta           = outdict['regression_coeff'].squeeze()
intercept      = outdict['intercept'].squeeze()
globalcomp     = beta * gmsst_in + intercept 
varraw_detrend = insst - globalcomp


#def detrend_ts(in_var,ts)

#%% Check Detrend 

plotvars        = [insst,gmsst_in,globalcomp,varraw_detrend]
plotvars_std    = [np.std(ss) for ss in plotvars]
plotnames       = ["NASST raw","GMSST","NASST Trend","NASST Detrended"]
plotcols        = ['gray','red','orange','cornflowerblue','violet']
fig,ax          = plt.subplots(1,1,constrained_layout=True,figsize=(8,3.5))

for ii in range(4):
    ax.plot(plotvars[ii],c=plotcols[ii],
            label="%s, $\sigma$ = %.3f $\degree C$" % (plotnames[ii],plotvars_std[ii]))

# ax.plot(insst,label="NASST raw",c='gray')
# ax.plot(gmsst_in,label="GMSST",c='red')
# ax.plot(globalcomp,label="NASST Trend",c='orange')
# ax.plot(varraw_detrend,label="NASST Detrended",c='blue')



    


ax.legend()

#%% Plot the Variance Based on Detrending


sstas_dt_experiment = [proc.xrdetrend_1d(insst,ii,return_model=True).detrended_input for ii in [1,2,3,4]]
ssts_trend_test     = sstas_dt_experiment + [varraw_detrend,]

expnames    = ['linear','quadratic','cubic','quartic',"Global Regression"]
nexps       = len(ssts_trend_test)


ssts_trend_test_lp = [proc.lp_butter(ts,120,6) for ts in ssts_trend_test]

stds        = [np.nanstd(sst) for sst in ssts_trend_test]
stds_lp     = [np.nanstd(sst) for sst in ssts_trend_test_lp]


#%% Visualize Barplot


instd       = stds
instd_lp    = stds_lp

vratio      = np.array(instd_lp) / np.array(instd) * 100

xlabs       = ["%s\n%.2f" % (expnames[ii],vratio[ii])+"%" for ii in range(len(vratio))]
fig,ax      = plt.subplots(1,1,constrained_layout=True,figsize=(6,6))

braw        = ax.bar(np.arange(nexps),instd,color=plotcols)
blp         = ax.bar(np.arange(nexps),instd_lp,color='k')

ax.bar_label(braw,fmt="%.04f",c='gray')#color=expcols)#c='gray')
ax.bar_label(blp,fmt="%.04f",c='k')


ax.set_xticks(np.arange(nexps),labels=xlabs)
ax.set_ylabel("$\sigma$(SST) [$\degree$C]")
ax.set_ylim([0,1.0])
ax.set_title("%s (%s)" % (bbname,bbstr))

ax.grid(True,ls='dotted',lw=0.55,c='gray')

#%% Try to Write a function to do this

invar   = natlbox[1]
invards = proc.xrdeseason(invar)
in_ts   = ds_gmsst.GMSST_MeanIce


def detrend_by_regression(invar,in_ts):
    # Given an DataArray [invar] and Timeseries [in_ts]
    # Detrend the timeseries by regression
    
    # Change to [lon x lat x time]
    invar       = invar.transpose('lon','lat','time')
    invar_arr   = invar.data # [lon x lat x time]
    ints_arr    = in_ts.data # [time]
    
    # Perform the regression
    outdict     = proc.regress_ttest(invar_arr,ints_arr)
    beta        = outdict['regression_coeff'] # Lon x Lat
    intercept   = outdict['intercept'] 
    
    # Remove the Trend
    ymodel      = beta[:,:,None] * ints_arr[None,None,:] + intercept[:,:,None]
    ydetrend    = invar_arr - ymodel
    
    # Prepare Output as DataArrays # [(time) x lat x lon]
    coords_full     = dict(time=invar.time,lat=invar.lat,lon=invar.lon)
    coords          = dict(lat=invar.lat,lon=invar.lon)
    da_detrend      = xr.DataArray(ydetrend.transpose(2,1,0),coords=coords_full,dims=coords_full,name=invar.name)
    da_fit          = xr.DataArray(ymodel.transpose(2,1,0),coords=coords_full,dims=coords_full,name='fit')
    da_pattern      = xr.DataArray(beta.T,coords=coords,dims=coords,name='regression_pattern')
    da_intercept    = xr.DataArray(intercept.T,coords=coords,dims=coords,name='intercept')
    da_sig          = xr.DataArray(outdict['sigmask'].T,coords=coords,dims=coords,name='sigmask')
    dsout = xr.merge([da_detrend,da_fit,da_pattern,da_intercept,da_sig])
    
    return dsout

dsdtall = detrend_by_regression(invards.rename('sst'),in_ts)

#%% Check the detrending

lonf    = -40
latf    = 65

yraw    = proc.selpt_ds(invards,lonf,latf,)
ydt     = proc.selpt_ds(dsdtall.sst,lonf,latf,)
yfit    = proc.selpt_ds(dsdtall.fit,lonf,latf,)

fig,ax  = plt.subplots(1,1,figsize=(8,3.5))

ax.plot(yraw,c='gray',label='raw')
ax.plot(ydt,c='k',label='detrended',)
ax.plot(yfit,c='red',label='fit')
ax.legend()

#%% Check Difference if you Remove Global mean Detrending on whole NASST
# Versus the pointwise

aavg_after = proc.sel_region_xr(dsdtall.sst,bbox_amv)
aavg_after = proc.area_avg_cosweight(aavg_after)

plotvars   = [aavg_after,varraw_detzrend]

plotnames  = ["Detrend Pointwise","Detrend After Area-Average"]
plotcols   = ["red","violet"]

stds        = [np.nanstd(sst) for sst in plotvars]
plotvars_lp = [proc.lp_butter(ts.data,120,6) for ts in plotvars]
stds_lp     = [np.nanstd(sst) for sst in plotvars_lp]   
    

instd       = stds
instd_lp    = stds_lp

vratio      = np.array(instd_lp) / np.array(instd) * 100

xlabs       = ["%s\n%.2f" % (plotnames[ii],vratio[ii])+"%" for ii in range(len(vratio))]
fig,ax      = plt.subplots(1,1,constrained_layout=True,figsize=(6,6))

braw        = ax.bar(np.arange(2),instd,color=plotcols)
blp         = ax.bar(np.arange(2),instd_lp,color='k')

ax.bar_label(braw,fmt="%.04f",c='gray')#color=expcols)#c='gray')
ax.bar_label(blp,fmt="%.04f",c='k')

ax.set_xticks(np.arange(2),labels=xlabs)
ax.set_ylabel("$\sigma$(SST) [$\degree$C]")
ax.set_ylim([0,1.0])
ax.set_title("%s (%s)" % (bbname,bbstr))

ax.grid(True,ls='dotted',lw=0.55,c='gray')

#%% 

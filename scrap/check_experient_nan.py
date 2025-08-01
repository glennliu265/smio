#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Check if an experinent has blown up (for run with negative damping...)

Created on Tue Jun  3 08:42:35 2025

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

#%% Load some other things to plot

# # Load Sea Ice Masks
dpath_ice = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/reanalysis/proc/NATL_proc_obs/"
nc_masks  = dpath_ice + "OISST_ice_masks_1981_2020.nc"
ds_masks  = xr.open_dataset(nc_masks).load()

# Load AVISO
dpath_aviso     = dpath_ice + "proc/"
nc_adt          = dpath_aviso + "AVISO_adt_NAtl_1993_2022_clim.nc"
ds_adt          = xr.open_dataset(nc_adt).load()
cints_adt       = np.arange(-100, 110, 10)


# Make a plotting function
def plot_ice_ssh(fsz_ticks=20-2,label_ssh=False):
    # Requires ds_masks and ds_adt to be loaded
    ax = plt.gca()
    
    # # Plot Sea Ice
    plotvar = ds_masks.mask_mon
    cl = ax.contour(plotvar.lon, plotvar.lat,
                    plotvar, colors="cyan",
                    linewidths=2, transform=proj, levels=[0, 1], zorder=1)
    
    # Plot the SSH
    plotvar = ds_adt.mean('time')
    cl = ax.contour(plotvar.lon, plotvar.lat, plotvar.adt*100, colors="k",alpha=0.8,
                    linewidths=0.75, transform=proj, levels=cints_adt)
    if label_ssh:
        ax.clabel(cl,fontsize=fsz_ticks)
    return None


#%% Further User Edits (Set Paths, Load other Data)

# Set Paths
figpath         = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20250702/"
sm_output_path  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/"
proc.makedir(figpath)

# Indicate Region Information
regname = "SPGNE"
bbsel   = [-40,-15,52,62]
locfn,locstring = proc.make_locstring_bbox(bbsel,lon360=True)
bbfn    = "%s_%s" % (regname,locfn)

# # (4) Same as (1) but using the runs including non-negative, insignificant HFFs
# comparename     = "AllFeedbacks"
# expnames        = ["SST_Obs_Pilot_00_Tdcorr0_qnet_AConly_SPGNE","SST_Obs_Pilot_00_Tdcorr0_qnet_noPositive","SST_ERA5_1979_2024"]
# expnames_long   = ["Stochastic Model (All Feedbacks)","Stochastic Mode","ERA5"]
# expnames_short  = ["SM_ALL","SM","ERA5"]
# expcols         = ["midnightblue","goldenrod","k"]
# expls           = ["dotted","dashed",'solid']

# Experiment with no subsurface damping
expname = "SST_Obs_Pilot_00_Tdcorr0_qnet_AConly_SPGNE"
# Experiment with subsurface damping
expname = "SST_Obs_Pilot_00_qnet_AConly_SPGNE_addlbdd"
# Experiment with ORAS5 subsurface damping
expname = "SST_Obs_Pilot_00_qnet_AConly_SPGNE_ORAS5"
# Experiment with En4 damping, all feedback, NO REM
expname = "SST_Obs_Pilot_00_qnet_AConly_SPGNE_noREM"

#%% Load the Data

dsall = dl.load_smoutput(expname,sm_output_path)


#%% Plot the variance

if 'run' not in dsall.coords:
    dsall = dsall.expand_dims(dim={'run':1},axis=-1)
    #dsall    = dsall.isel(run=-1)
    
# Check the max values
dsall.SST.max('time').plot()






#%% Load ERA5 for Comparison... ( copied from cesm2_hierarchy_v_obs)

# Load SST
dpath_era   = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
ncname_era  = dpath_era + "ERA5_sst_NAtl_1979to2024.nc"
ds_era      = xr.open_dataset(ncname_era).sst
ds_era_reg  = proc.sel_region_xr(ds_era,bbsel).load()

# Preprocess
dsa_era = proc.xrdeseason(ds_era_reg)

# Load GMSST For ERA5 Detrending
detrend_obs_regression = True
dpath_gmsst     = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
nc_gmsst        = "ERA5_GMSST_1979_2024.nc"

# Detrend bt Regression
ds_gmsst      = xr.open_dataset(dpath_era + nc_gmsst).GMSST_MeanIce.load()
dtout         = proc.detrend_by_regression(dsa_era,ds_gmsst)
sst_era       = dtout.sst

#%% Check Variance over the region

fsz_tick    = 32
fsz_axis    = 36

# Choose which one =================================
plotvar   = sst_era.std('time')
ename     = "ERA5"

plotvar   = dsall.SST.std('time').max('run')
ename     = "Stochastic Model (All Feedback)"
# ==================================================

proj      = ccrs.PlateCarree()
cints     = np.arange(0.30,1.05,0.05)

fig,ax,bb = viz.init_regplot(regname="SPGE",fontsize=fsz_tick)



pcm       = ax.contourf(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                        levels=cints)

cl        = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                        levels=cints,colors="k",linewidths=0.75)
clbl      = ax.clabel(cl,fontsize=fsz_tick)


cb        = viz.hcbar(pcm,ax=ax,fontsize=fsz_tick)
cb.set_label("%s $\sigma$(SST) [$\degree C$]" % ename,fontsize=fsz_axis)


figname   = "%s%s_SST_Stdev_SPGNE.png" % (figpath,ename)
plt.savefig(figname,dpi=150,bbox_inches='tight')


#%% Check Variance over the region

fsz_tick    = 32
fsz_axis    = 36

# Choose which one =================================
plotvar   = sst_era.std('time')
ename     = "ERA5"

plotvar   = dsall.SST.std('time').mean('run')
ename     = "Stochastic Model (All Feedback)"
# ==================================================

proj      = ccrs.PlateCarree()
cints     = np.arange(0.30,1.05,0.05)

fig,ax,bb = viz.init_regplot(regname="SPGE",fontsize=fsz_tick)



pcm       = ax.contourf(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                        levels=cints)

cl        = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                        levels=cints,colors="k",linewidths=0.75)
clbl      = ax.clabel(cl,fontsize=fsz_tick)


cb        = viz.hcbar(pcm,ax=ax,fontsize=fsz_tick)
cb.set_label("%s $\sigma$(SST) [$\degree C$]" % ename,fontsize=fsz_axis)


figname   = "%s%s_SST_Stdev_SPGNE.png" % (figpath,ename)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Plot where it is above a certain threshold

thres     = 0.94
plotvar   = dsall.SST.std('time').mean('run')
ename     = "Stochastic Model (All Feedback)"

plotvar   = xr.where(plotvar > thres,plotvar,np.nan)


proj      = ccrs.PlateCarree()
cints     = np.arange(0.30,1.05,0.05)

fig,ax,bb = viz.init_regplot(regname="SPGE",fontsize=fsz_tick)



pcm       = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj)

# cl        = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
#                         levels=cints,colors="k",linewidths=0.75)
# clbl      = ax.clabel(cl,fontsize=fsz_tick)


cb        = viz.hcbar(pcm,ax=ax,fontsize=fsz_tick)
cb.set_label("%s $\sigma$(SST) [$\degree C$]" % ename,fontsize=fsz_axis)


#%% Check Max SST

fsz_tick    = 32
fsz_axis    = 36

# Choose which one =================================
plotvar   = sst_era.max('time')
ename     = "ERA5"

plotvar   = dsall.SST.max('time').max('run')
ename     = "Stochastic Model (All Feedback)"
# ==================================================

proj      = ccrs.PlateCarree()
cints     = np.arange(1,3.6,0.1)

fig,ax,bb = viz.init_regplot(regname="SPGE",fontsize=fsz_tick)



pcm       = ax.contourf(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                        levels=cints)

cl        = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                        levels=cints,colors="k",linewidths=0.75)
clbl      = ax.clabel(cl,fontsize=fsz_tick)


cb        = viz.hcbar(pcm,ax=ax,fontsize=fsz_tick)
cb.set_label("%s $MAX$(SST) [$\degree C$]" % ename,fontsize=fsz_axis)


#ax.plot(-39.75,59.75,marker="x",transform=proj,markersize=120,c='blue')
ax.plot(-40,50,marker="x",transform=proj,markersize=120,c='blue')


figname   = "%s%s_SST_MAX_SPGNE.png" % (figpath,ename)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Do some filtering


smstds   = dsall.SST.std('time').max('run')

cutoff   = 5



fig,ax   = plt.subplots(1,1)

binedges = np.arange(0,10.5,0.2)
# Based on https://stackoverflow.com/questions/26218704/matplotlib-histogram-with-collection-bin-for-high-values
ax.hist(np.clip(smstds.data.flatten(),binedges[0],binedges[-1]),bins=binedges)

ax.set_ylim([0,30])

ax.axvline([cutoff])

pct = np.sum(smstds.data.flatten() > cutoff) / len(smstds.data.flatten()) * 100
ax.set_title("Points Above Cutoff: %.2f" % pct + "%")


maskthres = xr.where(smstds > cutoff,np.nan,1)

#cutoff_thres = 

#%% Calculate some quick metrics

if cutoff is not None:
    dsallin = dsall.SST * maskthres
else:
    dsallin = dsall.SST

#
aavg_era = proc.area_avg_cosweight(sst_era)
aavg_sm  = proc.area_avg_cosweight(dsallin)

# Copied from area_avg_output
def reshape_ens2year(ds):
    # Convert Runs to Year
    ds          = ds.transpose('ens','time')
    nens,ntime  = ds.shape
    arr         = ds.data.flatten()
    timedim      = xr.cftime_range(start='0000',periods=nens*ntime,freq="MS",calendar="noleap")
    coords      = dict(time=timedim)
    return xr.DataArray(arr,coords=coords,dims=coords,name=ds.name)

aavg_sm = aavg_sm.rename(dict(run='ens'))
aavg_sm = reshape_ens2year(aavg_sm)

#%%

lags            = np.arange(61)
nsmooths        = [250,4]
ssts            = [aavg_sm.data,aavg_era.data]
metrics_out     = scm.compute_sm_metrics(ssts,nsmooth=nsmooths,lags=lags)


#%%

kmonth = 1
xtks   = lags[::6]
nexps  = len(ssts)
fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,4))
ax,_   = viz.init_acplot(kmonth,xtks,lags,ax=ax)


expnames = ["Stochastic Model","ERA5",]
expnames_long = expnames
expcols  = ['hotpink',"k"]
expls     = ['dashed','solid']

for ex in range(nexps):
    
    plotvar = metrics_out['acfs'][kmonth][ex]
    ax.plot(lags,plotvar,
            label=expnames_long[ex],color=expcols[ex],ls=expls[ex],lw=2.5)
    
ax.legend()


#%%

dtmon_fix       = 60*60*24*30
xper            = np.array([40,10,5,1,0.5])
xper_ticks      = 1 / (xper*12)
    
fig,ax          = plt.subplots(1,1,figsize=(8,4.5),constrained_layout=True)

for ii in range(nexps):
    plotspec        = metrics_out['specs'][ii] / dtmon_fix
    plotfreq        = metrics_out['freqs'][ii] * dtmon_fix
    CCs = metrics_out['CCs'][ii] / dtmon_fix

    ax.loglog(plotfreq,plotspec,lw=2.5,label=expnames_long[ii],c=expcols[ii])
    
    #ax.loglog(plotfreq,CCs[:,0],ls='dotted',lw=0.5,c=expcols[ii])
    #ax.loglog(plotfreq,CCs[:,1],ls='dashed',lw=0.9,c=expcols[ii])

ax.set_xlim([1/1000,0.5])
ax.axvline([1/(6)],label="",ls='dotted',c='gray')
ax.axvline([1/(12)],label="",ls='dotted',c='gray')
ax.axvline([1/(5*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(10*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(40*12)],label="",ls='dotted',c='gray')

ax.set_xlabel("Frequency (1/Month)",fontsize=14)
ax.set_ylabel("Power [$\degree C ^2 cycle \, per \, mon$]")

ax2 = ax.twiny()
ax2.set_xlim([1/1000,0.5])
ax2.set_xscale('log')
ax2.set_xticks(xper_ticks,labels=xper)
ax2.set_xlabel("Period (Years)",fontsize=14)

# Plot Confidence Interval (ERA5)
alpha           = 0.05
cloc_era        = [plotfreq[0],1e-1]
dof_era         = metrics_out['dofs'][-1]
cbnds_era       = proc.calc_confspec(alpha,dof_era)
proc.plot_conflog(cloc_era,cbnds_era,ax=ax,color='k',cflabel=r"95% Confidence") #+r" (dof= %.2f)" % dof_era)

ax.legend()

#%% Locate point with large values/blow up

thres   = 0.1
targvar = dsall.SST.std('time').mean('run')
id_exceed = np.argwhere(np.array(targvar > thres))

values = []
nmaxes = len(id_exceed)
for ii in range(nmaxes):
    yy,xx = id_exceed[ii]
    values.append(targvar.data[yy,xx])

idmax       = np.argmax(np.abs(values))
yymax,xxmax = id_exceed[idmax]
tsout       = dsall.SST.isel(lat=yymax,lon=xxmax)

#%% Plot the timseries at a point

im     = 8
tmax   = 1000
irun   = 0
fig,ax = plt.subplots(1,1,figsize=(12.5,4.5))
ax.plot(tsout.isel(run=irun))
ax.set_xlim([0,tmax])
#ax.set_ylim([-10000,1000])
ax.set_ylim([-4,4])

plotvs = np.arange(im,tmax+im,12)
for ii in range(len(plotvs)):
    ax.axvline(plotvs[ii],c='red',lw=0.75)






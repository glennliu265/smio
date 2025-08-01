#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

For ERA5, check what happens when you compute ACF
using the pre and post-satellite ERA timeperiod

Created on Thu Mar 27 17:29:01 2025

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

figpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20250724/"
datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/ERA5/"
proc.makedir(figpath)
#%%

# Load Sea Ice Masks (ERA5)
dpath_ice =  "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/ERA5/processed/"
nc_masks  = dpath_ice + "icemask_era5_global180_max005perc.nc"#"/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/reanalysis/proc/NATL_proc_obs/"
ds_masks  = xr.open_dataset(nc_masks).load()

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


ds_sst = ds_all.sst
mons3  = proc.get_monstr()
#%% Part (1) Make an Ice Mask

icethres            = 0.05
recalculate_icemask = False

if recalculate_icemask:
    # Make Max Ice Mask
    iceconc_max  = ds_all.siconc.max('time') 
    icemask      = xr.where(iceconc_max > 0.05,np.nan,1).rename('mask')
    
    outname      = "%sprocessed/icemask_era5_global180_max005perc.nc" % datpath
    edict        =  proc.make_encoding_dict(icemask)
    icemask.to_netcdf(outname,encoding=edict)
    
    
    # Make Mean Ice Mask
    iceconc_mean = ds_all.siconc.mean('time') 
    icemask      = xr.where(iceconc_mean > 0.05,np.nan,1).rename('mask')
    
    outname      = "%sprocessed/icemask_era5_global180_mean005perc.nc" % datpath
    edict        =  proc.make_encoding_dict(icemask)
    icemask.to_netcdf(outname,encoding=edict)

outname      = "%sprocessed/icemask_era5_global180_max005perc.nc" % datpath
icemask_max  = xr.open_dataset(outname).load().mask
outname      = "%sprocessed/icemask_era5_global180_mean005perc.nc" % datpath
icemask_mean = xr.open_dataset(outname).load().mask


#%% Make Ice Mask for each period


#%% Check Differences in ice cover between the two periods

cutyear   = '1979'
ds_in     = ds_all.siconc

dscropped = proc.splittime_ds(ds_in,cutyear)

dscropmax = [ds.max('time') for ds in dscropped] 

titles    = ["Pre-Satellite Era (1940-1978)",
             "Satellite Era (1979-2024)"]

#%% --- <0> --- Examine Differences in Sea Ice Coverage

bbice     = [-80,0,40,65]
proj      = ccrs.PlateCarree()
fig,axs,_ = viz.init_orthomap(1,2,bbice,figsize=(18,6.5))

fsz_title = 26
fsz_axis  = 22
fsz_tick  = 18

for ii in range(2):
    
    ax = axs[ii]
    ax = viz.add_coast_grid(ax,bbox=bbice,proj=proj)
    
    plotvar = dscropmax[ii]#dscropped[ii].max('time')
    pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                            cmap='cmo.ice',vmin=0,vmax=1)
    
    cl     = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                             levels=[0.5,],colors="yellow")
    ax.clabel(cl,fontsize=fsz_tick)
    
    # Plot the bounding box of analysis
    bbox_sel = [-50,-10,50,60]
    bbname    = "Yeager2015"
    viz.plot_box(bbox=bbox_sel,ax=ax,color='pink',linewidth=2)
    ax.set_title(titles[ii],fontsize=fsz_title)
    
    
    
cb = viz.hcbar(pcm,ax=axs.flatten())
cb.set_label("Max Sea Ice Concentration (Fraction)",fontsize=fsz_axis)
    
savename = "%sSea_Ice_Concentration_OISST_Satellite_Comparison.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')





#%% Part (2), Examine Behavior over a box

"""
bbname = ["Yeager 2015",
          "SPG West",
          "SPG East",
          "West REI Maxima",
          "East REI Maxima",
          ]
bbsel = ([-50,-10,50,60],
         [-42,-25,50,60],
         [-35,-10,55,63],
         [-50,-40,50,55],
         [-25,-10,50,55]
         )
"""


# Yeager Bpx
#bbox_sel = [-50,-40,50,55]
#bbname   = "REIWest"
#bbox_sel = [-50,-10,50,60]
#bbname    = "Yeager2015"
bbox_sel  = [-40,-15,52,62]
bbname    = "SPGNE"

dsreg    = proc.sel_region_xr(ds_sst,bbox_sel)


#%% 2.1 ) Ice Mask Effects ----------------------------------------------------

dsreg_nomask   = dsreg.copy()
dsreg_meanmask = dsreg * icemask_mean 
dsreg_maxmask  = dsreg * icemask_max



    


expnames    = ["No Mask","Mean Mask","Max Mask"] 
expcols     = ['k','cornflowerblue','orange']
expls       = ['solid','dashed','dotted']
in_ssts     = [dsreg_nomask, dsreg_meanmask, dsreg_maxmask]


in_ssts     = [proc.area_avg_cosweight(ds) for ds in in_ssts]

#%% Preprocess, Do Calcuations

lags        = np.arange(61)
sstas       = [proc.xrdeseason(ds) for ds in in_ssts]
sstasdt     = [signal.detrend(ds) for ds in sstas]
metrics_out = scm.compute_sm_metrics(sstasdt,lags=lags,detrend_acf=False)

#%% --- <0> --- Visualize ACF Differences
xticks  = lags[::6]

kmonth      = 1 
fig,ax      = plt.subplots(1,1,constrained_layout=True,figsize=(8,4.5))
ax,_      = viz.init_acplot(kmonth,xticks,lags,ax=ax)

for ii in range(3):
    plotacf = metrics_out['acfs'][kmonth][ii]
    ax.plot(lags,plotacf,label=expnames[ii],c=expcols[ii],ls=expls[ii],lw=2.5)

ax.legend()
ax.set_title("%s ACF for %s" % (mons3[kmonth],bbname),fontsize=12)
savename = "%sIcemask_Comparison_ACF_%s_Avg_mon%02i.png" % (figpath,bbname,kmonth+1)
plt.savefig(savename,dpi=150)

#%% 2.2) Test Time Period -----------------------------------------------------

dsreg         = proc.sel_region_xr(ds_sst,bbox_sel)
dsreg_maxmask = dsreg * icemask_max

# Lets use the max mask
aavg_masked = proc.area_avg_cosweight(dsreg_maxmask)
 
presat      = aavg_masked.sel(time=slice('1940-01-01','1978-12-31'))
postsat     = aavg_masked.sel(time=slice('1979-01-01','2024-12-31'))

in_ssts     = [aavg_masked,presat,postsat]
expnames    = ["All","Pre-Satellite (1940-1978)","Post-Satellite (1979-2024)"] 
expcols     = ['k','cornflowerblue','orange']
expls       = ['solid','dashed','dotted']

#%% Preprocess, Do Calcuations

nsmooth     = 5
lags        = np.arange(61)
sstas       = [proc.xrdeseason(ds) for ds in in_ssts]
sstasdt     = [signal.detrend(ds) for ds in sstas]

metrics_out = scm.compute_sm_metrics(sstasdt,lags=lags,nsmooth=nsmooth,detrend_acf=False)


#%% --- <0> --- Visualize ACF Differences
xticks  = lags[::6]

kmonth  = 1
fig,ax  = plt.subplots(1,1,constrained_layout=True,figsize=(8,4.5))
ax,_      = viz.init_acplot(kmonth,xticks,lags,ax=ax)

for ii in range(3):
    plotacf = metrics_out['acfs'][kmonth][ii]
    ax.plot(lags,plotacf,label=expnames[ii],c=expcols[ii],ls=expls[ii],lw=2.5)

ax.legend()
ax.set_title("%s ACF for %s" % (mons3[kmonth],bbname),fontsize=12)
savename = "%sPeriod_Comparison_ACF_%s_Avg_mon%02i.png" % (figpath,bbname,kmonth+1)
plt.savefig(savename,dpi=150)

#%% --- <0> --- Plot the timeseries (No Detrending)
tplot  = np.arange(len(aavg_masked))
times  = aavg_masked.time

times_str = [str(times.data[t])[:7] for t in range(len(times))]
times_yr  = [t[:4] for t in times_str]


fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,4.5))

# Plot Raw Timeseries
plotvar = sstas[0]
ax.plot(tplot,plotvar,label='Raw (Not Detrended)',lw=.75,c='gray')

# Plot Rolling Mean
meanwindow = 60
plotvar_smooth = plotvar.rolling(time=meanwindow,center=True).mean()
ax.plot(tplot,plotvar_smooth,label="smoothed (%i-month window)" % meanwindow,
        lw=3,c='navy')

ax.legend()

ax.set_xlabel("Time")
ax.set_ylabel("SST [$\degree C$]")
ax.axvline(0+(1979-1940)*12,c="k",lw=2.5)
ax.set_xticks(tplot[::120],labels=times_yr[::120])
ax.set_xlim([0,1020])

ax.set_title("SST Timeseries Undetrended, %s" % bbname)
savename = "%sSST_Timeseries_Undetrended_%s.png" % (figpath,bbname)
plt.savefig(savename,dpi=150)


#%% --- <0> --- 

id1978 = 0+(1979-1940)*12


tpers = [np.arange(len(times)),np.arange(id1978),np.arange(id1978,len(times))]
fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(10,4.5))

# Plot Raw Timeseries
plotvar = sstas[0]
ax.plot(tplot,plotvar,label='Raw (Not Detrended)',lw=.75,c='gray')

# Plot Rolling Mean
meanwindow = 60
plotvar_smooth = plotvar.rolling(time=meanwindow,center=True).mean()
ax.plot(tplot,plotvar_smooth,label="smoothed (%i-month window)" % meanwindow,
        lw=3,c='red')



# Plot Detrended version of each
for ii in range(3):
    plotvar        = sstasdt[ii]
    plotvar_smooth = np.convolve(plotvar,np.ones(meanwindow),'same')/meanwindow#plotvar.rolling(time=meanwindow,center=True).mean()
    ax.plot(tpers[ii],plotvar_smooth,label=expnames[ii] + '(detrended)',
            c=expcols[ii],lw=3)


ax.legend()

ax.set_xlabel("Time")
ax.set_ylabel("SST [$\degree C$]")
ax.axvline(0+(1979-1940)*12,c="k",lw=2.5)
ax.set_xticks(tplot[::120],labels=times_yr[::120])
ax.set_xlim([0,1020])

ax.set_title("SST Timeseries, %s" % bbname)
savename = "%sSST_Timeseries_%s.png" % (figpath,bbname)
plt.savefig(savename,dpi=150)


#%% --- <0> --- Compare Power Spectra

dtmon_fix       = 60*60*24*30
xper            = np.array([40,10,5,1,0.5])
xper_ticks      = 1 / (xper*12)
    

fig,ax      = plt.subplots(1,1,figsize=(8,4.5),constrained_layout=True)

for ii in range(3):
    plotspec        = metrics_out['specs'][ii] / dtmon_fix
    plotfreq        = metrics_out['freqs'][ii] * dtmon_fix
    CCs = metrics_out['CCs'][ii] / dtmon_fix

    ax.loglog(plotfreq,plotspec,lw=2.5,label=expnames[ii],c=expcols[ii])
    
    ax.loglog(plotfreq,CCs[:,0],ls='dotted',lw=0.5,c=expcols[ii])
    ax.loglog(plotfreq,CCs[:,1],ls='dashed',lw=0.9,c=expcols[ii])

ax.set_xlim([1/1000,0.5])
ax.axvline([1/(6)],label="",ls='dotted',c='gray')
ax.axvline([1/(12)],label="",ls='dotted',c='gray')
ax.axvline([1/(5*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(10*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(40*12)],label="",ls='dotted',c='gray')
ax.legend()
ax.set_xlabel("Frequency (1/Month)",fontsize=14)
ax.set_ylabel("Power [$\degree C ^2 cycle \, per \, mon$]")

ax2 = ax.twiny()
ax2.set_xlim([1/1000,0.5])
ax2.set_xscale('log')
ax2.set_xticks(xper_ticks,labels=xper)
ax2.set_xlabel("Period (Years)",fontsize=14)

ax.set_title("Power Spectra (%s)" % (bbname))

savename = "%sSpectra_ERA5_Period_Comparison_%s.png" % (figpath,bbname)
plt.savefig(savename,dpi=150,bbox_inches='tight')
    

# ====================================================================================
#%% 3) Trying to understand the differences

ds_in          = ds_sst
dscrop         = proc.splittime_ds(ds_in,cutyear)

# Deseason
dscropds       = [proc.xrdeseason(ds) for ds in dscrop]
dscropdsdt     = [proc.xrdetrend(ds) for ds in dscropds]

#%% 4) Visualize differences in standard deviation

bbice     = [-80,0,40,65]
proj      = ccrs.PlateCarree()
fig,axs,_ = viz.init_orthomap(1,2,bbice,figsize=(18,6.5))

fsz_title = 26
fsz_axis  = 22
fsz_tick  = 18

for ii in range(2):
    
    ax = axs[ii]
    ax = viz.add_coast_grid(ax,bbox=bbice,proj=proj)
    
    plotvar = dscropdsdt[ii].std('time')#dscropmax[ii]#dscropped[ii].max('time')
    pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                            cmap='cmo.thermal',vmin=0,vmax=2.5)
    
    # cl     = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
    #                          levels=[0.5,],colors="yellow")
    #ax.clabel(cl,fontsize=fsz_tick)
    
    # Plot the bounding box of analysis
    bbox_sel = [-50,-10,50,60]
    bbname    = "Yeager2015"
    viz.plot_box(bbox=bbox_sel,ax=ax,color='pink',linewidth=2)
    ax.set_title(titles[ii],fontsize=fsz_title)
    
    
    
cb = viz.hcbar(pcm,ax=axs.flatten())
cb.set_label("Standard Deviation (SST) [$\degree$C]",fontsize=fsz_axis)
    
savename = "%sSST_OISST_StdDev_Satellite_Comparison.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Look at variance differences

fig,ax,_ = viz.init_orthomap(1,1,bbice,figsize=(12,6.5))
ax       = viz.add_coast_grid(ax,bbox=bbice,proj=proj)

plotvar  = np.log(dscropdsdt[1].std('time') / dscropdsdt[0].std('time'))
pcm      = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                        cmap='cmo.balance',vmin=-1,vmax=1)

cb = viz.hcbar(pcm,ax=ax)
cb.set_label(r"Log ($\sigma_{pre}/\sigma_{post}$)",fontsize=fsz_axis)

savename = "%sSST_OISST_StdDev_Ratio_Satellite_Comparison.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Compare Pointwise acf between the different regions
acf_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"

fns_acf = [
    "ERA5_NAtl_1940to1978_lag00to60_ALL_ensALL.nc",
    "ERA5_NAtl_1979to2024_lag00to60_ALL_ensALL.nc"
    ]

ds_acfs = [acf_path + fn for fn in fns_acf]
ds_acfs = [xr.open_dataset(fn).load() for fn in ds_acfs]
ds_acfs = [ds.acf.squeeze() for ds in ds_acfs]

t2s     = [proc.calc_T2(ds,axis=-1,ds=True) for ds in ds_acfs]

#%% Visualize difference in persistence

kmonth    = 2

for kmonth in range(12):
    lon = ds_acfs[0].lon
    lat = ds_acfs[0].lat
    
    titles    = ["Pre-Satellite Era (1940-1978)",
                 "Satellite Era (1979-2024)",
                 "Pre - Post"]
    
    fig,axs,_ = viz.init_orthomap(1,3,bbice,figsize=(18,6.5))
    
    fsz_title = 26
    fsz_axis  = 22
    fsz_ticks  = 18
    
    for ii in range(3):
        
        ax = axs[ii]
        ax = viz.add_coast_grid(ax,bbox=bbice,proj=proj)
        
        if ii  < 2:
            plotvar = t2s[ii][...,kmonth]
            cmap    = 'cmo.dense'
            vlms    = [0,24]
        else:
            plotvar = t2s[1][...,kmonth] - t2s[0][...,kmonth]
            cmap    = 'cmo.balance'
            vlms    = [-24,24]
            
        ax.set_title(titles[ii],fontsize=fsz_title)
        pcm     = ax.pcolormesh(lon,lat,plotvar.T,transform=proj,
                                cmap=cmap,vmin=vlms[0],vmax=vlms[1])
        
        cb = viz.hcbar(pcm,ax=ax)
        cb.set_label("$T^2$ (Months)",fontsize=fsz_axis)
        
        # # Plot Sea Ice
        plotvar = ds_masks.mask
        plotvar = xr.where(np.isnan(plotvar),0,plotvar)
        cl = ax.contour(plotvar.lon, plotvar.lat,
                        plotvar, colors="yellow",
                        linewidths=2, transform=proj, levels=[0,1], zorder=2)
        
        # Plot the SSH
        plotvar = ds_adt.mean('time')
        cl = ax.contour(plotvar.lon, plotvar.lat, plotvar.adt*100, colors="k",alpha=0.8,
                        linewidths=0.75, transform=proj, levels=cints_adt)
        ax.clabel(cl,fontsize=fsz_ticks-2)
    
    savename = "%sT2_ERA5_Period_Comparison_mon%02i.png" % (figpath,kmonth+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
        

#%% 4) Additional Sensitivity Analysis

#%% Part (1) Try different forms of detrending (each period)


in_aavg    = aavg_masked#postsat#sstas[2]#postsat
inssta     = proc.xrdeseason(in_aavg)




xdim = np.arange(len(inssta))

# Linear Detrend
inssta_lin,model1 = proc.detrend_poly(xdim,inssta.data,1) #signal.detrend(inssta,type='linear')
#inssta_lin = proc.detrend_dim(inssta,0)

# Quadratic Detrend
inssta2,model2 = proc.detrend_poly(xdim,inssta.data,2)

# Cubic Detrend
inssta3,model3 = proc.detrend_poly(xdim,inssta.data,3)

dtnames   = ["Raw","Linear","Quadratic","Cubic"]
dttest_in = [inssta.data,inssta_lin,inssta2,inssta3]
dtmods    = [None,model1,model2,model3]

dtcols    = ['gray','blue','orange','forestgreen']

metrics_dt = scm.compute_sm_metrics(dttest_in,lags=lags,detrend_acf=False)


#%% Plot the timeseries

fig,ax = plt.subplots(1,1,figsize=(12.5,4.5),constrained_layout=True)
for ii in range(4):
    
    if ii > 0:
        ls= 'dashed'
        ax.plot(xdim,dtmods[ii],c=dtcols[ii],ls=ls)
    else:
        ls = 'solid'
        
        
        
    
    ax.plot(xdim,dttest_in[ii],label=dtnames[ii],c=dtcols[ii],lw=1.5,ls=ls)
ax.set_xlim([0,xdim[-1]])
ax.legend()
#plt.plot(postsat)

#%% --- <0> --- Visualize Detrending Effects on ACF

xtks = lags[::6]

fig,ax = plt.subplots(1,1,figsize=(12.5,4.5),constrained_layout=True)
ax,_ = viz.init_acplot(kmonth,xtks,lags,ax=ax)

acfs = []
for ii in range(4):
    
    if ii > 0:
        ls= 'dashed'
    else:
        ls = 'solid'
    plotvar = metrics_dt['acfs'][kmonth][ii]
    
    ax.plot(lags,plotvar,label=dtnames[ii],c=dtcols[ii],lw=2.5,ls=ls)
    
    
    acfs.append(plotvar)
    
ax.legend()

#%% 4) New Section (Compare Time Period with GMSST Detrending)

def preproc_ds(ds,gmsst_in):
    dsa   = proc.xrdeseason(ds)
    if gmsst_in is None:
        print("Doing Linear Detrend")
        dsadt = proc.xrdetrend(dsa)
    else:
        dsadt = proc.detrend_by_regression(dsa.rename('sst'),gmsst_in)
        dsadt = dsadt['sst']
    return dsadt


detrend_type = ''

presat      = dsreg_meanmask.sel(time=slice('1940-01-01','1978-12-31'))
postsat     = dsreg_meanmask.sel(time=slice('1979-01-01','2024-12-31'))

in_ssts     = [dsreg_meanmask,presat,postsat]
expnames    = ["All","Pre-Satellite (1940-1978)","Post-Satellite (1979-2024)"] 
expcols     = ['k','cornflowerblue','orange']
expls       = ['solid','dashed','dotted']


in_gmssts   = [ds_gmsst_merge.GMSST_MaxIce,
               ds_gmsst_pre.GMSST_MaxIce,
               ds_gmsst.GMSST_MaxIce]

# Preprocess (Deseason and Detrend)
if detrend_type == 'linear':
    in_sstas    = [preproc_ds(in_ssts[ii],None) for ii in range(3)]
else:
    in_sstas    = [preproc_ds(in_ssts[ii],in_gmssts[ii]) for ii in range(3)]


aavgs       = [proc.area_avg_cosweight(proc.xrdeseason(ds)).data for ds in in_ssts]
aavgs_dt    = [proc.area_avg_cosweight(ds).data for ds in in_sstas]

#%% Compute ACFs

nsmooth = 5
lags    = np.arange(61)
met_dt  = scm.compute_sm_metrics(aavgs_dt,lags=lags,nsmooth=nsmooth,detrend_acf=False)
met     = scm.compute_sm_metrics(aavgs,lags=lags,nsmooth=nsmooth,detrend_acf=False)

#%% Compare ACFs

xticks      = lags[::6]
ytks        = np.arange(-.25,1.25,.25)

kmonth      = 2
fig,ax      = plt.subplots(1,1,constrained_layout=True,figsize=(8,4.5))
ax,_        = viz.init_acplot(kmonth,xticks,lags,ax=ax)

for aa in range(2):
    if aa == 0:
        metrics_out = met
        loopalpha   = 0.25
        labeladd    = ""
        lsloop          = 'dotted'
    else:
        metrics_out = met_dt
        loopalpha   = 1
        labeladd    = ", GMSST Removed"
        lsloop          = 'solid'
    
    # Loop by Experiment
    for ii in range(3):
        plotacf = metrics_out['acfs'][kmonth][ii]
        label = expnames[ii] + labeladd
        ax.plot(lags,plotacf,label=label,c=expcols[ii],ls=lsloop,lw=2.5,alpha=loopalpha)

ax.set_yticks(ytks)
ax.set_ylim([-.25,1])
ax.legend()
ax.set_title("%s ACF for %s" % (mons3[kmonth],bbname),fontsize=12)
savename = "%sPeriod_Comparison_ACF_Detrend_%s_%s_Avg_mon%02i.png" % (figpath,detrend_type,bbname,kmonth+1)
plt.savefig(savename,dpi=150)

#%% Plot the timeseries

fig,ax = plt.subplots(1,1,figsize=(12.5,4.5))

times_in = [ds.time for ds in in_gmssts]
for aa in range(2):
    if aa == 0:
        metrics_out = aavgs
        loopalpha   = 0.25
        labeladd    = ""
        lsloop          = 'dotted'
    else:
        metrics_out = aavgs_dt
        loopalpha   = 1
        labeladd    = " detrended"
        lsloop      = 'solid'
    
    # Loop by Experiment
    
    for ii in range(3):
        
        plotvar = metrics_out[ii]
        label   = expnames[ii] + labeladd
        ax.plot(times_in[ii],plotvar,label=label,c=expcols[ii],ls=lsloop,lw=2.5,alpha=loopalpha)

#ax.set_ylim([-.2,.2])

#%% Check How different timeseries is with removal of linear trend for each month

import scipy as sp




def detrend_bymon(ts):
    ntime      = len(ts)
    nyr        = int(ntime/12)
    ts_yrmon   = ts.reshape(nyr,12)
    ts_detrend = sp.signal.detrend(ts_yrmon,axis=0,type='linear')
    return ts_detrend.flatten()

def detrend_bymon_detail(ts,returnall=False):
    ntime      = len(ts)
    nyr        = int(ntime/12)
    ts_yrmon   = ts.reshape(nyr,12)
    
    coeffs    = []
    newmodel  = []
    residuals = []
    for im in range(12):
        c,m,r = proc.polyfit_1d(np.arange(nyr),ts_yrmon[:,im],1)
        coeffs.append(c)
        newmodel.append(m)
        residuals.append(r)
    
    coeffs      = np.array(coeffs)
    newmodel    = np.array(newmodel)
    residuals   = np.array(residuals)
    
    ts_detrend = ts_yrmon - newmodel.T
    
    if returnall:
        return ts_detrend,ts_yrmon,coeffs,newmodel,residuals
    return ts_detrend.flatten()


ii       = 2
tsraw    = aavgs_dt[ii]
tsdt     = detrend_bymon_detail(tsraw)
pfit_out = detrend_bymon_detail(tsraw,returnall=True)

fig,ax = plt.subplots(1,1,figsize=(12.5,4.5))


ax.plot(times_in[ii],tsraw,label="Original")
ax.plot(times_in[ii],tsdt,label="Detrend by Month")
ax.legend()


#%% Check Seasonal Trends

ts_detrend,ts_yrmon,coeffs,newmodel,residuals = pfit_out

im = 0

if ii == 0:
    xplot = np.arange(1940,2025)
elif ii == 1:
    xplot = np.arange(1940,1979)
else:
    xplot = np.arange(1979,2025)


plotmons = np.roll(np.arange(12),1)
fig,axs = plt.subplots(4,3,figsize=(10,12),constrained_layout=True)

for aa in range(12):
    ax = axs.flatten()[aa]
    
    im = plotmons[aa]
    ax.plot(xplot,ts_yrmon[:,im],c="k",lw=3,label="Original")
    ax.plot(xplot,ts_detrend[:,im],c="red",lw=3,ls='dashed',label="Detrended")
    ax.plot(xplot,newmodel[im,:],c='magenta',lw=2,ls='dashed',label='Fit')
    ax.set_xlim([xplot[0],xplot[-1]])
    ax.set_title(mons3[im])
    
    ax.set_ylim([-1.5,1.5])
    ax.axhline([0],c='k',lw=.55)
    if aa == 0:
        ax.legend()
plt.suptitle("Monthy Linear Trends for %s" % (expnames[ii]))

#%% Check Coefficient by Season


interceptsmon = coeffs[:,0]
betasmon      = coeffs[:,1]

fig,ax=viz.init_monplot(1,1,)
#ax.plot(mons3,betasmon,c=expcols[ii],lw=2.5,label='intercept')
ax.plot(mons3,interceptsmon,c=expcols[ii],ls='dotted',label="slope")
ax.set_title("Linear Fit Slopes for %s" % expnames[ii])
ax.legend()
ax.set_ylim([-.03,.03])
ax.axhline([0],c='k',lw=.55)

ax.set_ylabel("Degree C")


#%% ==========================================================================
    
    #ts = ts.reshape()
    #times = ds.time.data



#%% 5) Verify Differences with mean/max ice detrending


in_sst      = dsreg_meanmask
in_gmssts   = [None,ds_gmsst_merge.GMSST_MeanIce,ds_gmsst_merge.GMSST_MaxIce]
in_sstas    = [preproc_ds(in_sst,in_gmssts[ii]) for ii in range(3)]


expnames    = ["Linear","Mean Ice","Max Ice"] 
expcols     = ['red','cornflowerblue','midnightblue']
expls       = ['solid','dashed','dotted']

aavgs_dt    = [proc.area_avg_cosweight(ds).data for ds in in_sstas]
metrics_out  = scm.compute_sm_metrics(aavgs_dt,lags=lags,nsmooth=nsmooth,detrend_acf=False)

#%%
kmonth      = 3
fig,ax      = plt.subplots(1,1,constrained_layout=True,figsize=(8,4.5))
ax,_      = viz.init_acplot(kmonth,xticks,lags,ax=ax)

for ii in range(3):
    plotacf = metrics_out['acfs'][kmonth][ii]
    ax.plot(lags,plotacf,label=expnames[ii],c=expcols[ii],ls=expls[ii],lw=2.5)

ax.legend()
ax.set_title("%s ACF for %s" % (mons3[kmonth],bbname),fontsize=12)
savename = "%sIcemask_Detrend_Comparison_ACF_%s_Avg_mon%02i.png" % (figpath,bbname,kmonth+1)
plt.savefig(savename,dpi=150)




# Part (1) Try 40-year chunks






# for ii in range(2):
    
#     dsin = ds_acfs[ii]
    
#     reis = []
#     for im in range(12):
        
#         rei     = proc.calc_remidx_xr(dsin.isel(mons=im),return_rei=True)
#         reis.append(rei)

#%% 





#ds_sst,ds_sicon

#nc_siconc = glob.glob(datpath + "sst*.nc")


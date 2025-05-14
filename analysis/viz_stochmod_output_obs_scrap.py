#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 08:39:21 2025

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
dpath_ice   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/reanalysis/proc/NATL_proc_obs/"
nc_masks    = dpath_ice + "OISST_ice_masks_1981_2020.nc"
ds_masks    = xr.open_dataset(nc_masks).load()
ds_ice_era5 = dl.load_mask("ERA5").mask
dsice_era5_plot = xr.where(np.isnan(ds_ice_era5),0,1)

# Load AVISO
dpath_aviso = dpath_ice + "proc/"
nc_adt      = dpath_aviso + "AVISO_adt_NAtl_1993_2022_clim.nc"
ds_adt      = xr.open_dataset(nc_adt).load()
cints_adt   = np.arange(-100, 110, 10)

# Make a plotting function
def plot_ice_ssh(fsz_ticks=20-2,label_ssh=False):
    # Requires ds_masks and ds_adt to be loaded
    ax = plt.gca()
    
    # # Plot Sea Ice
    plotvar = dsice_era5_plot#ds_masks.mask_mon
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
figpath         = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20250507/"
sm_output_path  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/"
proc.makedir(figpath)

# Indicate Observations
append_obs      = True
dpath_obs       = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/reanalysis/proc/NATL_proc_obs/"
ncname_obs      = "ERA5_sst_NAtl_1979to2024.nc" 
obsname         = "ERA5"
obsname_long    = "ERA5 Reanalysis (1979-2024)"

detrend_obs_regression = True
dpath_gmsst     = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
nc_gmsst        = "ERA5_GMSST_1979_2024.nc"

# Indicate Experiment and Comparison Name 

# ---

# (1) Comparison for Paper Outline (with and w/o Re-emergence vs Obs, Qnet HFF/Forcing)
comparename     = "paperoutline"
expnames        = ["SST_Obs_Pilot_00_Tdcorr0_qnet","SST_Obs_Pilot_00_Tdcorr1_qnet","SST_ERA5_1979_2024"]
expnames_long   = ["Stochastic Model (with re-emergence)","Stochastic Model","ERA5"]
expcols         = ["turquoise","goldenrod","k"]
expls           = ["dotted","dashed",'solid']

# ---

# (2) Include Insignificant HFF Points
comparename     = "hffsigtest"
expnames        = ["SST_Obs_Pilot_00_Tdcorr0_qnet","SST_Obs_Pilot_00_Tdcorr0_qnet_noPositive","SST_ERA5_1979_2024"]
expnames_long   = ["Stochastic Model","Stochastic Model (Include Insignificant HFF)","ERA5"]
expnames_short  = ["SM","SM (Insig HFF)","ERA5"]
expcols         = ["turquoise","cornflowerblue","k"]
expls           = ["dotted","dashed",'solid']


# ---
# (3) Same as (1) but using the runs including non-negative, insignificant HFFs
comparename     = "IncludeInsig"
expnames        = ["SST_Obs_Pilot_00_Tdcorr0_qnet_noPositive","SST_Obs_Pilot_00_Tdcorr1_qnet_noPositive_SPGNE","SST_ERA5_1979_2024"]
expnames_long   = ["Stochastic Model (with re-emergence)","Stochastic Model","ERA5"]
expnames_short  = ["SM_REM","SM","ERA5"]
expcols         = ["turquoise","goldenrod","k"]
expls           = ["dotted","dashed",'solid']



#%% Load stochastic model output

st     = time.time()
ds_all = []
nexps  = len(expnames)
for ex in range(nexps):
    
    expname = expnames[ex]
    if (detrend_obs_regression) and ("ERA5" in expname):
        print("Skipping ERA5, loading separately")
        continue
    print("Loading output for %s" % expname)    
    
    # Note: Only loads the first run...
    ds = dl.load_smoutput(expname,sm_output_path,runids=[0,])
    ds_all.append(ds)
print("Loaded all output in %.2fs" % (time.time()-st))


#%% Anomalize and preprocess

st      = time.time()
in_ssts = [ds.SST for ds in ds_all]
sstas   = [proc.xrdeseason(ds) for ds in in_ssts]
sstasdt = [proc.xrdetrend(ds) for ds in sstas]
print("Completed preprocessing in %.2fs" % (time.time()-st))

#%% Load ERA5 separately, if option is set

if detrend_obs_regression:
    ds_era5      = xr.open_dataset(dpath_obs + ncname_obs).sst.load()
    ds_gmsst     = xr.open_dataset(dpath_gmsst + nc_gmsst).GMSST_MeanIce.load()
    
    # Deseasonalize
    ssta_era5    = proc.xrdeseason(ds_era5)
    # Remove Trend
    dsdtera5     = proc.detrend_by_regression(ssta_era5,ds_gmsst)
    ssta_dt_era5 = dsdtera5.sst
    
    sstasdt.append(ssta_dt_era5)

#%% Some Basinwide Plots...

# Set Plotting Parameters
fsz_title = 24
fsz_axis  = 22
fsz_ticks = 20

#%% Plot Stdev for each simulation

for ex in range(nexps):
    
    fig,ax,_ = viz.init_orthomap(1,1,bboxplot,figsize=(18,10))
    ax       = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)
    plotvar  = sstasdt[ex].std('time')
    pcm      = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                            vmin=0,vmax=2,cmap='cmo.thermal')
    
    ax.set_title(expnames_long[ex],fontsize=fsz_title)
    
    cb = viz.hcbar(pcm,ax=ax)
    cb.set_label("$\sigma$(SST) [$\degree$C]",fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_ticks)
    
    plot_ice_ssh(label_ssh=False)
    
    figname = "%sStdev_SST_%s.png" % (figpath,expnames[ex])
    plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Plot the Log ratio

cints           = np.log(np.array([0.25,0.5,1,2,4]))
clabs           = ["0.25x","0.5x","1x","2x","4x"]

refid           = -1
mon             = 1
for mon in range(1,13,1):
    
    for expid in range(2):
        
        fig,ax,_        = viz.init_orthomap(1,1,bboxplot,figsize=(16,10))
        ax              = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)
        
        if mon == "Annual":
            sstin = [ds for ds in sstasdt]
        else:
            sstin = [ds.sel(time=ds.time.dt.month.isin(mon)) for ds in sstasdt]
        
        plotvar         = np.log((sstin[expid].std('time')) / (sstin[refid].std('time')))  # Stochastic Model / ERA5
        pcm             = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                                vmin=-1,vmax=1,cmap='cmo.balance')
        
        
        cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                                levels=cints,colors="k",linewidths=0.75)
        fmt= {}
        for l, s in zip(cl.levels, clabs):
            fmt[l] = s
            
        ax.clabel(cl,fmt=fmt,fontsize=fsz_ticks)
        ax.set_title( mons3[mon-1] + " $\sigma_{SST}$ Ratio (" + expnames_long[expid] + " / " + expnames_long[refid] + ")",fontsize=fsz_title)
        
        
        plot_ice_ssh(label_ssh=False)
        
        cb = viz.hcbar(pcm,ax=ax)
        cb.set_label(r"log($\frac{Stochastic \, Model}{ERA5}$)",fontsize=fsz_axis)
        cb.ax.tick_params(labelsize=fsz_ticks)
        
        savename = "%sStdev_Ratio_Stochmod_%s_ERA5_Mon%s.png" % (figpath,expnames[expid],mon)
        plt.savefig(savename,dpi=150,bbox_inches='tight')

# =======================================================
#%% Part 2: Take Area Average over selected bounding box
# =======================================================

# Set Bounding Box and smoothing amoung, lags
bbsel       = [-40,-15,52,62]
nsmooths    = [50,50,4]
lags        = np.arange(61)

dsreg       = [proc.sel_region_xr(ds,bbsel) for ds in sstasdt]
ds_aavg     = [proc.area_avg_cosweight(ds) for ds in dsreg]
aavg        = [ds.data for ds in ds_aavg]

metrics_pt  = scm.compute_sm_metrics(aavg,nsmooths,lags=lags)
metrics_byreg.append(metrics_pt)

# Compute area average prior to detrending for visualization
dsreg_undt  = [proc.sel_region_xr(ds,bbsel) for ds in sstas]
aavg_undt   = [proc.area_avg_cosweight(ds) for ds in dsreg_undt]
aavg_byreg.append(aavg_undt)

# Compute Stdevs and LPF Dstdevs
stdevs      = np.array([np.nanstd(ts) for ts in aavg])
aavg_lp     = [proc.lp_butter(ts,120,6) for ts in aavg]
stdevs_lp   = np.array([np.nanstd(ts) for ts in aavg_lp])

    
#%% Plot the autocorrelation function

#% Plot ACF --------------------------------------------------------
kmonth     = 1
for kmonth in range(12):
    xtks       = np.arange(0,61,6)
    #lags       = np.arange(37)
    fig,ax     = plt.subplots(1,1,constrained_layout=True,figsize=(8,4))
    
    ax,_ = viz.init_acplot(kmonth,xtks,lags,ax=ax,title=None)
    
    for ex in range(nexps):
        plotvar = metrics_pt['acfs'][kmonth][ex]
        ax.plot(lags,plotvar,label=expnames_long[ex],c=expcols[ex],lw=2.5)
        
    ax.legend()
    ax.set_title("%s Autocorrelation" % (mons3[kmonth],))
    ax.set_ylim([-.5,1.25])
    
    savename = "%sACF_%s_mon%02i.png" % (figpath,comparename,kmonth+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
    
    
#%% Plot ACF Map

pmesh            = True
xtks             = lags[::6]
acfmaps          = np.array(metrics_pt['acfs']) # [Basemonth x Experiment x Lag]
vlms             = [0,1]

cints            = np.arange(0,1.1,0.1)#np.arange(-1,1,0.2)

for ex in range(nexps):
    fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(20,6.5))
    
    ax.set_xticks(xtks)
    ax.tick_params(labelsize=fsz_tick)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    ax.set_ylabel("Base Month of\nSST Anomaly",fontsize=fsz_axis)
    plotvar   = acfmaps[:,ex,:] # [Kmonth x run x lag] --> [kmonth x lag]
    if pmesh:
        pcm = ax.pcolormesh(lags,mons3,plotvar,
                            cmap='cmo.amp',vmin=vlms[0],vmax=vlms[1],
                            edgecolors="lightgray")
    else:
        pcm = ax.contourf(lags,mons3,plotvar,
                            cmap='cmo.amp',levels=cints,
                            edgecolors="lightgray")
    
    cl = ax.contour(lags,mons3,plotvar,
                    colors="k",levels=cints,)
    clbl = ax.clabel(cl,fontsize=fsz_ticks)
    viz.add_fontborder(clbl)
    
    cb = viz.hcbar(pcm,ax=ax,fraction=0.045)
    cb.ax.tick_params(labelsize=fsz_tick)
    cb.set_label("Correlation with Base Month SST",fontsize=fsz_axis)
    
    ax.set_title("%s" % expnames_long[ex],fontsize=fsz_title)
    
    savename = "%sACFMap_%s_%s.png" % (figpath,comparename,expnames[ex])
    plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% TRying to add lines along feb (need to deal with this further)


# kmonth = 1 
# nyears = int(len(lags)/12)
# for yy in range(nyears):
#     lagsplot   = yy*12 +  np.arange(0,12,1)
#     imon       = np.arange(0,12,1)
#     ycoord     = imon + kmonth + 1
#     ax.plot(lagsplot,np.flip(ycoord),c='yellow',ls='dotted',lw=2)
    
#for ii in range(len())


#%% Plot the Spectra

metrics_out     = metrics_pt

dtmon_fix       = 60*60*24*30
xper            = np.array([40,10,5,1,0.5])
xper_ticks      = 1 / (xper*12)

    
fig,ax      = plt.subplots(1,1,figsize=(8,4.5),constrained_layout=True)

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

savename = "%sSpectra_%s.png" % (figpath,comparename)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Make Barplot for Standard Deviation Over Subpolar Gyre Region
    
instd       = stdevs
instd_lp    = stdevs_lp

vratio      = np.array(instd_lp) / np.array(instd) * 100


xlabs       = ["%s\n%.2f" % (expnames_short[ii],vratio[ii])+"%" for ii in range(len(vratio))]
fig,ax      = plt.subplots(1,1,constrained_layout=True,figsize=(6,6))


braw        = ax.bar(np.arange(nexps),instd,color='gray')
blp         = ax.bar(np.arange(nexps),instd_lp,color='k')


ax.bar_label(braw,fmt="%.04f",c='gray')
ax.bar_label(blp,fmt="%.04f",c='k')

#ax.bar_label(vratio,fmt="%.2f",c=w,label_type='bottom')

ax.set_xticks(np.arange(nexps),labels=xlabs)
ax.set_ylabel("$\sigma$(SST) [$\degree$C]")
ax.set_ylim([0,1.0])

#%% Monthly variance, barplot version



for ex in range(nexps):
    
    N          = 12
    theta      = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    plot_theta = np.roll(theta,-3)  # Introduce offset to start at 12:00
    
    # Intoduce a Flip to make values increase clockwise
    plotvar      = np.flip(metrics_pt['monvars'][ex])
    theta_labels = np.flip(mons3)
    
    
    width      = 2*np.pi / len(plotvar)
    
    #bcolors = 
    fig,ax         = plt.subplots(1,1,subplot_kw={'projection':'polar'})
    bars = ax.bar(plot_theta, plotvar, width=width,bottom=0.2,color=expcols[ex],alpha=0.99,edgecolor="w")
    # blb = ax.bar_label(bars,fmt="%.02f",c='k',)
    # viz.add_fontborder(blb,w=3)
    
    
    #ax.plot(theta,plotvar)
    ax.set_xticks(plot_theta,labels=theta_labels)
    ax.set_rlim([0,1])
    ax.grid(True,c='k',ls='dotted',lw=0.55)
    ax.set_title(r"%s, Monthly Variance ([$\degree$C$^2$])" % expnames_long[ex])
    
    

#%% Non-Rewritten Scripts Below # ---------------------------------------------





#%% Plot Standard Deviation for each simulation

    
    
    
    
    
#%% Plot the log ratio between the two


#%% Let's check the autocorrelation (at a point)

lonf       = -30
latf       = 55

locfn,loctitle = proc.make_locstring(lonf,latf,fancy=True)

nsmooths   = [1000,40]

dspt       = [proc.selpt_ds(ds,lonf,latf) for ds in sstasdt]
sstpt      = [ds.data for ds in dspt]

metrics_pt = scm.compute_sm_metrics(sstpt,nsmooths)

#% Plot ACF

kmonth     = 1
xtks       = np.arange(0,37,3)
lags       = np.arange(37)
fig,ax     = plt.subplots(1,1,constrained_layout=True,figsize=(8,4))

ax,_ = viz.init_acplot(kmonth,xtks,lags,ax=ax,title=None)

for ii in range(3):
    plotvar = metrics_pt['acfs'][kmonth][ii]
    ax.plot(lags,plotvar,label=expnames[ii],c=expcols[ii],lw=2.5)
    
ax.legend()
ax.set_title("%s Autocorrelation (%s)" % (mons3[kmonth],loctitle))
ax.set_ylim([-.5,1.25])


#%% Plot Bounding Boxes

cints           = np.log(np.array([0.25,0.5,1,2,4]))
clabs           = ["0.25x","0.5x","1x","2x","4x"]

#AnnRatio        = np.log((sstasdt[0].std('time')) / (sstasdt[1].std('time')))  # Stochastic Model / ERA5
AnnRatio        = np.log((sstasdt[1].std('time')) / (sstasdt[2].std('time')))  # Stochastic Model / ERA5
#mon = "Annual"

#%%
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
bbcol = ('k',
         'orange',
         'violet',
         'navy',
         'firebrick')

bbsty = (
    'solid',
    'dashed',
    'solid',
    'dashed',
    'dotted'
    )
nregs = len(bbname)



#%% Plot Map of ratios
bboxspg         = [-70,-10,35,65]

fig,ax,_        = viz.init_orthomap(1,1,bboxspg,figsize=(16,10))
ax              = viz.add_coast_grid(ax,bbox=bboxspg,fill_color="lightgray",fontsize=fsz_tick)

plotvar         = AnnRatio  # Stochastic Model / ERA5
pcm             = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                        vmin=-1,vmax=1,cmap='cmo.balance')


cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                        levels=cints,colors="k",linewidths=0.75)
fmt= {}
for l, s in zip(cl.levels, clabs):
    fmt[l] = s
    
ax.clabel(cl,fmt=fmt,fontsize=fsz_ticks)

ax.set_title( "$\sigma_{SST}$ Ratio (" + expnames[0] + " / " + expnames[1] + ")",fontsize=fsz_title)


# # Plot Sea Ice
plotvar = ds_masks.mask_mon
cl = ax.contour(plotvar.lon, plotvar.lat,
                plotvar, colors="yellow",
                linewidths=2, transform=proj, levels=[0, 1], zorder=1)

# Plot the SSH
plotvar = ds_adt.mean('time')
cl = ax.contour(plotvar.lon, plotvar.lat, plotvar.adt*100, colors="dimgray",alpha=0.8,
                linewidths=0.75, transform=proj, levels=cints_adt)
ax.clabel(cl,fontsize=fsz_ticks-2)


cb = viz.hcbar(pcm,ax=ax)
cb.set_label(r"log($\frac{Stochastic \, Model}{ERA5}$)",fontsize=fsz_axis)
cb.ax.tick_params(labelsize=fsz_ticks)


# Plot some boxes
for rr in range(nregs):
    viz.plot_box(bbsel[rr],ax=ax,proj=proj,color=bbcol[rr],leglab=bbname[rr],linewidth=4,linestyle=bbsty[rr])


    
savename = "%sStdev_Ratio_Stochmod_ERA5_BBox.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Compute Metrics for each region
nsmooths      = [30,30,30,3]
metrics_byreg = []
aavg_byreg    = []
lags          = np.arange(61)

for rr in range(nregs):
    bbin = bbsel[rr]
    dsreg       = [proc.sel_region_xr(ds,bbin) for ds in sstasdt]
    ds_aavg     = [proc.area_avg_cosweight(ds) for ds in dsreg]
    aavg        = [ds.data for ds in ds_aavg]
    
    metrics_pt = scm.compute_sm_metrics(aavg,nsmooths,lags=lags)
    metrics_byreg.append(metrics_pt)
    
    # Compute area average prior to detrending for visualization
    dsreg_undt  = [proc.sel_region_xr(ds,bbin) for ds in sstas]
    aavg_undt   = [proc.area_avg_cosweight(ds) for ds in dsreg_undt]
    aavg_byreg.append(aavg_undt)

#%% Examine the timeseries for each simulation

meanwindow  = 60
rr          = 3

for rr in range(nregs):
    fig,axs     = plt.subplots(4,1,constrained_layout=True,figsize=(12,10))
    
    for ii in range(4):
        ax = axs[ii]
        plotvar = aavg_byreg[rr][ii]
        xplot   = np.arange(len(plotvar))
        
        # Plot the unfiltered timeseries
        ax.plot(xplot,plotvar,label="raw",c='gray',lw=1)
        
        # Plot the filtered timeseries
        plotvar_smooth = plotvar.rolling(time=meanwindow,center=True).mean()
        
        ax.plot(xplot,plotvar_smooth,label="smoothed (%i-month window)" % meanwindow,lw=3,c='goldenrod')
        
        if ii == 0:
            ax.legend(ncol=2,fontsize=fsz_tick)
            
        #viz.label_sp(expnames[ii])
        ax.set_title(expnames[ii],fontsize=fsz_title)
    
        # Add some other things
        ax.set_xlim([xplot[0],xplot[-1]])
        ax.axhline([0],c='k',lw=0.75,ls='dashed')
        ax.tick_params(labelsize=fsz_tick)
    
    plt.suptitle(bbname[rr],fontsize=fsz_title)
    
    savename = "%sTimeseries_%s_meanwindow%03i.png" % (figpath,bbname[rr],meanwindow)
    plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Plot ACF

rr = 2

for rr in range(nregs):
    metrics = metrics_byreg[rr]
    
    
    #% Plot ACF --------------------------------------------------------
    kmonth     = 1
    xtks       = np.arange(0,61,6)
    #lags       = np.arange(37)
    fig,ax     = plt.subplots(1,1,constrained_layout=True,figsize=(8,4))
    
    ax,_ = viz.init_acplot(kmonth,xtks,lags,ax=ax,title=None)
    
    for ii in range(4):
        plotvar = metrics['acfs'][kmonth][ii]
        ax.plot(lags,plotvar,label=expnames[ii],c=expcols[ii],lw=2.5)
        
    ax.legend()
    ax.set_title("%s Autocorrelation (%s)" % (mons3[kmonth],bbname[rr]))
    ax.set_ylim([-.5,1.25])
    savename = "%sACF_%s_mon%02i.png" % (figpath,bbname[rr],kmonth+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')

# ====================================
#%% Try plotting polar plot of the ACF
# ====================================

# N          = 12
# theta      = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
# plot_theta = np.roll(theta,-3)  # Introduce offset to start at 12:00

# # Intoduce a Flip to make values increase clockwise
# plotvar      = np.flip(metrics_pt['monvars'][ex])
# theta_labels = np.flip(mons3)


# width      = 2*np.pi / len(plotvar)

# #bcolors = 
# fig,ax         = plt.subplots(1,1,subplot_kw={'projection':'polar'})
# bars = ax.bar(plot_theta, plotvar, width=width,bottom=0.2,color=expcols[ex],alpha=0.99,edgecolor="w")
# # blb = ax.bar_label(bars,fmt="%.02f",c='k',)
# # viz.add_fontborder(blb,w=3)


# #ax.plot(theta,plotvar)
# ax.set_xticks(plot_theta,labels=theta_labels)
# ax.set_rlim([0,1])
# ax.grid(True,c='k',ls='dotted',lw=0.55)
# ax.set_title(r"%s, Monthly Variance ([$\degree$C$^2$])" % expnames_long[ex])



ii        = 0

kmonth    = 1

for kmonth in range(12):
    acfin     = metrics_pt['acfs'][kmonth][ii]
    
    # Get Loop
    nyrs       = int(len(lags)/12) # calculate number of years, based on lag
    montile    = np.tile(np.arange(1,13,1),nyrs) # Tile months
    rmontile   = np.append(np.roll(montile,-kmonth),kmonth+1) # Tile, then add month (to account for lag zero)
    
    
    N           = 12
    theta       = np.linspace(0.0, 2 * np.pi, N, endpoint=False) 
    theta_tile  = np.tile(theta,nyrs)
    rtheta_tile = np.append(np.roll(theta_tile,-kmonth),theta_tile[kmonth])
    
    fig,ax      = plt.subplots(1,1,subplot_kw={'projection':'polar'})
    
    ax.plot(rtheta_tile,acfin)
    sc = ax.scatter(rtheta_tile,acfin,c=lags)
    #bars        = ax.bar(plot_theta, plotvar, width=width,bottom=0.2,color=expcols[ex],alpha=0.99,edgecolor="w")
    
    ax.set_xticks(theta,labels=np.arange(1,13,1))
    
    ax.set_title("%s ACF (%s)" % (mons3[kmonth],expnames_long[ii]))
    
    ax.set_rlim([0,1])
    
    cb = viz.hcbar(sc,ax=ax,pad=0.05)
    cb.set_label("Lags (Month)")
    savename = "%sPolar_ACF_Plot_%s_mon%02i.png" % (figpath,expnames[ii],kmonth+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
# blb = ax.bar_label(bars,fmt="%.02f",c='k',)
# viz.add_fontborder(blb,w=3)

#plot_theta = np.roll(theta,-3) 


#%% Plot monvar

for rr in range(nregs):
    fig,ax = viz.init_monplot(1,1,figsize=(8,4))
    
    
    for ii in range(2):
        plotvar = metrics_byreg[rr]['monvars'][ii]
        ax.plot(mons3,plotvar,label=expnames[ii])
    ax.legend()
    ax.set_title("Monthly Variance (%s)" % (bbname[rr]))
    
    ax.set_ylim([0,1.5])
    ax.set_ylabel("Interannual Standard Deviation [$\degree$C]")
    
    savename = "%sMonvar_%s.png" % (figpath,bbname[rr])
    plt.savefig(savename,dpi=150,bbox_inches='tight')
    
#%% Plot Spectra

#expcols         = ['blue','orange']

dtmon_fix       = 60*60*24*30
xper            = np.array([40,10,5,1,0.5])
xper_ticks      = 1 / (xper*12)

for rr in range(nregs):
    
    metrics_out     = metrics_byreg[rr]
    
    
    fig,ax      = plt.subplots(1,1,figsize=(8,4.5),constrained_layout=True)
    
    for ii in range(4):
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
    
    ax.set_title("Power Spectra (%s)" % (bbname[rr]))
    
    
    savename = "%sSpectra_%s.png" % (figpath,bbname[rr])
    plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Look at plot in other forms... Linear plot first

dtmon_fix       = 60*60*24*30
xper            = np.array([40,10,5,1,0.5])
xper_ticks      = 1 / (xper*12)

xlims_lin = [1/1000,0.1]

for rr in range(nregs):
        
    metrics_out     = metrics_byreg[rr]
    
    
    fig,ax      = plt.subplots(1,1,figsize=(8,4.5),constrained_layout=True)
    
    for ii in range(4):
        plotspec        = metrics_out['specs'][ii] / dtmon_fix
        plotfreq        = metrics_out['freqs'][ii] * dtmon_fix
        CCs             = metrics_out['CCs'][ii] / dtmon_fix
    
        ax.plot(plotfreq,plotspec,lw=2.5,label=expnames[ii],c=expcols[ii])
        
        ax.plot(plotfreq,CCs[:,0],ls='dotted',lw=0.5,c=expcols[ii])
        ax.plot(plotfreq,CCs[:,1],ls='dashed',lw=0.9,c=expcols[ii])
    
    if rr == 2:
        ax.set_ylim([0,40])
    
    ax.set_xticks(xper_ticks)
    ax.set_xlim(xlims_lin)
    ax.axvline([1/(6)],label="",ls='dotted',c='gray')
    ax.axvline([1/(12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(5*12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(10*12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(40*12)],label="",ls='dotted',c='gray')
    ax.legend()
    
    ax.set_xlabel("Frequency (1/Month)",fontsize=14)
    ax.set_ylabel("Power [$\degree C ^2 cycle \, per \, mon$]")
    
    ax2 = ax.twiny()
    
    #ax2.set_xscale('log')
    ax2.set_xticks(xper_ticks,labels=xper)
    ax2.set_xlim(xlims_lin)
    ax2.set_xlabel("Period (Years)",fontsize=14)
    
    ax.set_title("Power Spectra (%s)" % (bbname[rr]))
    
    
    savename = "%sSpectra_Linear_%s.png" % (figpath,bbname[rr])
    plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Look at plot in other forms... Variance Preserving

dtmon_fix       = 60*60*24*30
xper            = np.array([40,10,5,1,0.5])
xper_ticks      = 1 / (xper*12)

xlims_lin = [1/1000,0.5]

for rr in range(nregs):
        
    metrics_out     = metrics_byreg[rr]
    
    
    fig,ax      = plt.subplots(1,1,figsize=(8,4.5),constrained_layout=True)
    
    for ii in range(4):
        
        plotfreq        = metrics_out['freqs'][ii] * dtmon_fix
        plotspec        = metrics_out['specs'][ii] * plotfreq #* dtmon_fix
        CCs             = metrics_out['CCs'][ii] * plotfreq[:,None]#* dtmon_fix
    
        ax.semilogx(plotfreq,plotspec,lw=2.5,label=expnames[ii],c=expcols[ii])
        
        ax.semilogx(plotfreq,CCs[:,0],ls='dotted',lw=0.5,c=expcols[ii])
        ax.semilogx(plotfreq,CCs[:,1],ls='dashed',lw=0.9,c=expcols[ii])
    
    
    
    ax.set_xticks(xper_ticks)
    ax.set_xlim(xlims_lin)
    ax.axvline([1/(6)],label="",ls='dotted',c='gray')
    ax.axvline([1/(12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(5*12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(10*12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(40*12)],label="",ls='dotted',c='gray')
    ax.legend()
    
    ax.set_xlabel("Frequency (1/Month)",fontsize=14)
    ax.set_ylabel("Power [$\degree C ^2 cycle \, per \, mon$]")
    
    ax2 = ax.twiny()
    
    ax2.set_xscale('log')
    
    ax2.set_xticks(xper_ticks,labels=xper)
    ax2.set_xlim(xlims_lin)
    
    ax2.set_xlabel("Period (Years)",fontsize=14)
    ax.set_title("Power Spectra (%s)" % (bbname[rr]))
    
    
    savename = "%sSpectra_FreqxPower_%s.png" % (figpath,bbname[rr])
    plt.savefig(savename,dpi=150,bbox_inches='tight')



    
#%%
#%% Random Plot, plot map of ACF at a specific leadtime

bboxspg2 = [-80,0,30,70]

fig,axs,_ = viz.init_orthomap(1,3,figsize=(28,14),bboxplot=bboxspg2)

for ex in range(3):
    ax = axs[ex]
    ax = viz.add_coast_grid(ax,bbox=bboxspg2,fill_color="lightgray",fontsize=fsz_tick)
    
    ax.set_title(expnames[ex],fontsize=fsz_title)
    
    plotvar = 

    

#%% Test (1): Check how different forms of detrending impact 



#%% Compare Pointwise ACF...

#

ncpath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
ncfile      = "ERA5_NAtl_1979to2025_lag00to60_ALL_ensALL.nc"
ds_acf      = xr.open_dataset(ncpath + ncfile).load()
acf_era5    = ds_acf.acf.isel(thres=0,ens=0)

ncfile2     = "SM_SST_Obs_Pilot_00_Tdcorr0_lag00to60_ALL_ensALL.nc"
ds_acf_sm2  = xr.open_dataset(ncpath + ncfile2).load() 
acf_sm      = ds_acf_sm2.acf.isel(thres=0,ens=0)


#%% Plot for ERA5 or SM experiment indicated above
plotnow   = "SM"
bboxspg   = [-80,0,30,65]
fsz_ticks = 16
fzs_axis  = 18
fsz_title = 24
cints     = np.arange(-1,1.25,0.25)

if plotnow == "ERA5":
    plotexp = "Era5_1979_2024"
    acf_in  = acf_era5
elif plotnow == "SM":
    plotexp = "SM_SST_Obs_Pilot_00_Tdcorr0"
    acf_in  = acf_sm
    

kmonth = 1
ilag   = 2

for ilag in range(61):
    plotvar = acf_in.isel(mons=kmonth,lags=ilag)
    
    fig,ax  = viz.init_regplot(bboxin=bboxspg)
    
    
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar.T,vmin=-1,vmax=1,
                        transform=proj,cmap='cmo.balance')
    
    cl  = ax.contour(plotvar.lon,plotvar.lat,plotvar.T,levels=cints,
                        transform=proj,colors='w',linewidths=0.75)
    #ax.clabel(cl,fontsize=fsz_ticks)
    
    ilagmon = int( (kmonth+ilag)%12 )
    ax.set_title("Lag %02i (%s)" % (ilag,mons3[ilagmon]),fontsize=fsz_title)
    
    # plot the currents
    plotvar = ds_adt.isel(time=ilagmon)
    cl = ax.contour(plotvar.lon, plotvar.lat, plotvar.adt*100, colors="dimgray",alpha=0.8,
                    linewidths=0.75, transform=proj, levels=cints_adt)
    #ax.clabel(cl,fontsize=fsz_ticks-2)s
    
    # # Plot Sea Ice
    plotvar = ds_masks.mask_mon
    cl = ax.contour(plotvar.lon, plotvar.lat,
                    plotvar, colors="yellow",
                    linewidths=2, transform=proj, levels=[0, 1], zorder=1)
    
    
    
    cb = viz.hcbar(pcm)
    cb.ax.tick_params(labelsize=fsz_ticks)
    cb.set_label("Correlation with %s Anomalies" % (mons3[kmonth]),fontsize=fsz_axis)
    
    outname = "%s%s_ACF_Map_Month%02i_lag%02i.png" % (figpath,plotexp,kmonth+1,ilag)
    plt.savefig(outname,dpi=150,bbox_inches='tight')
    
    
#%% Plot Corresponding ACF as Above for marking time


rr         = 3
kmonth     = 1

metrics = metrics_byreg[rr]

for ilag in range(60):
    ilagmon = int( (kmonth+ilag)%12 )
    print("Lag %i is %s" % (ilag,mons3[ilagmon]))

    #% Plot ACF --------------------------------------------------------
    
    fig,ax     = plt.subplots(1,1,constrained_layout=True,figsize=(8,4))
    
    ax,_ = viz.init_acplot(kmonth,xtks,lags,ax=ax,title=None)
    
    for ii in range(4):
        plotvar = metrics['acfs'][kmonth][ii]
        ax.plot(lags,plotvar,label=expnames[ii],c=expcols[ii],lw=2.5)
        
        ax.plot(lags[ilag],plotvar[ilag],c=expcols[ii],marker="o",
                markerfacecolor='None',markersize=15)
    
    
    
    
    
    ax.axvline([ilag],c='red',lw=2.5)
    ax.legend()
    ax.set_title("%s Autocorrelation (%s)" % (mons3[kmonth],bbname[rr]))
    ax.set_ylim([-.5,1.25])
    savename = "%sACF_%s_mon%02i_frame%02i.png" % (figpath,bbname[rr],kmonth+1,ilag)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
        
    


#%%
# #%% Old Script Below ==========================================================

# #%% Load and append Observational set

# # if append_obs:
# #     # Load ERA5
# #     ncera           = dpath_era + 
# #     dsera           = xr.open_dataset(ncera).load().sst

# #%% Load the data
# # Load stochastic model data (1kyr run, Td Mistake)
# # dpath_sm    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/SST_Obs_Pilot_00/Output/"
# # ncsm        = dpath_sm + "SST_runidrun00.nc"
# # dssm        = xr.open_dataset(ncsm).load()

# # Load stochastic model data (1kyr run, with entrainment forcing) -------------

# splitrun    = False
# # <Option 1> Load 2 500yr runs for SPG subregion -
# if splitrun:
#     dpath_sm    = sm_output_path + "SST_Obs_Pilot_SPG_Short/Output/"
#     ncsms       = [dpath_sm + "SST_runidrun00.nc",dpath_sm + "SST_runidrun01.nc"]
#     dssm        = xr.open_mfdataset(ncsms,concat_dim='run',combine='nested').load()
#     sstout      = dssm.SST.data
#     sstout      = sstout.reshape(2*6000,121,241)
#     faketime    = xr.cftime_range(start='0000',periods=12000,freq="MS",calendar="noleap")
#     dims        = dict(time=faketime,lat=dssm.lat,lon=dssm.lon)
#     dssm        = xr.DataArray(sstout,name="SST",dims=dims,coords=dims)
# # <Option 2> Load 1000-yr run for NAtl
# else:
    
#     dpath_sm   = sm_output_path +  "SST_Obs_Pilot_00_TdCorr0/Output/"
#     ncsm       = dpath_sm + "SST_runidrun00.nc"
#     dssm       = xr.open_dataset(ncsm).load().SST

    

# # Load stochastic model data (1kyr run, NO entrainment forcing)
# dpath_sm2 = sm_output_path + "SST_Obs_Pilot_00/Output/"
# ncsm2     = dpath_sm2 + "SST_runidrun00.nc"
# dssm2     = xr.open_dataset(ncsm2).load().SST


# # Load stochastic model (1kyr run with Qnet Damping/Forcing)
# dpath_sm3  = sm_output_path + "SST_Obs_Pilot_00_TdCorr0_qnet/Output/"
# ncsm3      = dpath_sm3 + "SST_runidrun00.nc"
# dssm3      = xr.open_dataset(ncsm3).load().SST



# #%% (1) Anomalize and preprocess ---------------------------------------------

# in_ssts     = [dssm2,dssm,dssm3,dsera]

# expcols     = ["turquoise",
#                "goldenrod",
#                'hotpink',
#                "k"]
# expnames    = ["Stochastic Model",
#                "Stochastic Model (with Entrainment)",
#                "Stochastic Model (Qnet)",
#                "ERA5"]
# sstas       = [proc.xrdeseason(ds) for ds in in_ssts]
# sstasdt     = [proc.xrdetrend(ds) for ds in sstas]


# nexps       = len(in_ssts)



# #%%
# # ----
# mldnc = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/MIMOC_regridERA5_h_pilot.nc"
# dsmld = xr.open_dataset(mldnc).load()

# hpt = proc.selpt_ds(dsmld,lonf,latf)
# hpt.plot()

# # ----

# # import xarray as xr

# # import numpy as np
# # import xarray as xr
# # import sys
# # import time
# # import matplotlib.pyplot as plt

# # import cartopy.crs as ccrs
# # #%% Import Custom Modules

# # # stormtrack
# # amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
# # scmpath = "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/" # scm module

# # sys.path.append(amvpath)
# # sys.path.append(scmpath)

# # from amv import proc,viz
# # import scm
# # import amv.loaders as dl

# # fname = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/sm_experiments/SST_Obs_Pilot_00/Output/SST_runidrun00.nc"
# # ds = xr.open_dataset(fname).load()

# # sstvar = ds.SST.var('time')
# # sststd = ds.SST.std('time')

# # #%%

# # proj = ccrs.PlateCarree()

# # fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj},constrained_layout=True)
# # ax.coastlines()


#%% Compare standard deviation

#plt.plot(sstasdt[0].sel(lon=lonf,lat=latf,method='nearest'))
lonf  =-50
latf  = 54
plt.plot(sstasdt[0].sel(lon=lonf,lat=latf,method='nearest'))

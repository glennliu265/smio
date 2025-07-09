#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Area Average Metrics
using timeseries preprocessed by area_average_output

Copied upper section of viz_stochmod_output_obs_Scrap.py on 2025.05.07
Copied visualization sections from area_average sensitivity 
Created on Wed May  7 09:29:32 2025

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
nc_masks = dpath_ice + "OISST_ice_masks_1981_2020.nc"
ds_masks = xr.open_dataset(nc_masks).load()

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
figpath         = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20250711/"
sm_output_path  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/"
proc.makedir(figpath)

# Indicate Region Information
regname = "SPGNE"
bbsel   = [-40,-15,52,62]
locfn,locstring = proc.make_locstring_bbox(bbsel,lon360=True)
bbfn    = "%s_%s" % (regname,locfn)

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
expnames        = ["SST_Obs_Pilot_00_Tdcorr1_qnet_noPositive_SPGNE","SST_Obs_Pilot_00_Tdcorr0_qnet_noPositive","SST_ERA5_1979_2024"]
expnames_long   = ["Stochastic Model","Stochastic Model (with re-emergence)","ERA5"]
expnames_short  = ["SM","SM_REM","ERA5"]
expcols         = ["goldenrod","turquoise","k"]
expls           = ["dashed","dotted",'solid']

# ---
# (4) Try Different Significance lEvels
comparename     = "SigLevelTest"
expnames        = ["SST_Obs_Pilot_00_Tdcorr1_qnet_noPositive_SPGNE",
                   "SST_Obs_Pilot_00_Tdcorr0_qnet_noPositive",
                   "SST_Obs_Pilot_00_Tdcorr0_qnet",
                   "SST_Obs_Pilot_00_Tdcorr0_qnet_p10_SPGNE",
                   "SST_Obs_Pilot_00_Tdcorr0_qnet_p20_SPGNE",
                   "SST_ERA5_1979_2024",
                   ]
expnames_long   = ["Remove Negative (no entrainment forcing)",
                   "Remove Negative",
                   "5% Significance",
                   "10% Significance",
                   "20% Significance",
                   "ERA5"
                   ]
expnames_short  = ["NoNeg0","NoNeg1","5","10","20","ERA5"]
expcols         = ["goldenrod","turquoise",'midnightblue','blue','cornflowerblue',"k"]
expls           = ["dotted","dotted",'dashed','dashed','dashed','solid']      

# (5) Add Subsurface Damping EN4
comparename     = "LbddEN4"
expnames        = [
                   "SST_Obs_Pilot_00_qnet_AConly_SPGNE_addlbdd",
                   #"SST_Obs_Pilot_00_qnet_AConly_SPG_addlbdd",
                   "SST_Obs_Pilot_00_Tdcorr0_qnet_noPositive",
                   "SST_Obs_Pilot_00_Tdcorr0_qnet_p10_SPGNE",
                   "SST_ERA5_1979_2024",
                   ]
expnames_long   = ["Add Subsurface Damping",
                   #"Add Subsurface Damping (with updated forcing)",
                   "No Positive Feedback",
                   "10% Significance",
                   "ERA5"
                   ]
expnames_short  = ["Lbdd","10","ERA5"] # "LbddUpdate",
expcols         = ["turquoise",'midnightblue','blue','cornflowerblue',"k"]
expls           = ["dotted",'dashed','dashed','solid']                  


# (6) Updated Versionw ith EN4 Damping and All-Feedback Forcing
comparename     = "OutlineInProg"
expnames        = ["SST_Obs_Pilot_00_qnet_AConly_SPGNE_noREM","SST_Obs_Pilot_00_qnet_AConly_SPG_addlbdd","SST_ERA5_1979_2024"]
expnames_long   = ["Stochastic Model","Stochastic Model (with re-emergence)","ERA5"]
expnames_short  = ["SM","SM_REM","ERA5"]
expcols         = ["goldenrod","turquoise","k"]
expls           = ["dashed","dotted",'solid']

# (6) Updated Versionw ith EN4 Damping and All-Feedback Forcing
comparename     = "OutlineORAS5"
expnames        = ["SST_Obs_Pilot_00_qnet_AConly_SPGNE_noREM","SST_Obs_Pilot_00_qnet_AConly_SPGNE_ORAS5","SST_ERA5_1979_2024"]
expnames_long   = ["Stochastic Model","Stochastic Model (with re-emergence)","ERA5"]
expnames_short  = ["SM","SM (+REM)","ERA5"]
expcols         = ["goldenrod","turquoise","k"]
expls           = ["dashed","dotted",'solid']

#%% Load information for each region

dsall = []
nexps = len(expnames)
for ex in tqdm.tqdm(range(nexps)):
    expname = expnames[ex]
    ncname  = "%s%s/Metrics/Area_Avg_%s.nc" % (sm_output_path,expname,bbfn)
    
    ds      = xr.open_dataset(ncname).load()
    dsall.append(ds)

    
#%% Preprocessing

dsa      = [proc.xrdeseason(ds) for ds in dsall]
ssts     = [sp.signal.detrend(ds.SST.data) for ds in dsa]
ssts_ds  = [proc.xrdetrend(ds.SST) for ds in dsa]

#%% Detrend ERA5 using Global Mean Regression Approach
if (detrend_obs_regression):
    
    # 
    sst_era = dsa[-1].SST
    sst_era = sst_era.expand_dims(dim=dict(lon=1,lat=1))
    
    # Add dummy lat lon
    #sst_era['lon'] = 1
    #sst_era['lat'] = 1
    
    # Load GMSST
    ds_gmsst     = xr.open_dataset(dpath_gmsst + nc_gmsst).GMSST_MeanIce.load()
    dsdtera5     = proc.detrend_by_regression(sst_era,ds_gmsst)
    
    sst_era_dt = dsdtera5.SST.squeeze()
    
    ssts_ds[-1] = sst_era_dt
    ssts[-1] = sst_era_dt.data
    #print("\nSkipping ERA5, loading separately")



#%% Compute basic metrics

ssts         = [ds.data for ds in ssts_ds]
lags         = np.arange(61)
nsmooths     = [250,] * (nexps-1) + [4,]
 #nsmooths     = [250,250,4]
metrics_out  = scm.compute_sm_metrics(ssts,nsmooth=nsmooths,lags=lags)

#%% Compute some additional metrics

stds     = np.array([ds.std('time').data.item() for ds in ssts_ds])
ssts_lp  = [proc.lp_butter(ts,120,6) for ts in ssts_ds]
stds_lp  = np.array([np.std(ds) for ds in ssts_lp])
vratio   = (stds_lp  / stds) * 100

#%% Plot ACF
kmonth = 1
xtks   = lags[::6]
conf   = 0.95

for kmonth in range(12):
    fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,4))
    ax,_   = viz.init_acplot(kmonth,xtks,lags,ax=ax)
    
    for ex in range(nexps):
        
        if ex == nexps-1:
            col_in = dfcol
        else:
            col_in = expcols[ex]
        
        plotvar = metrics_out['acfs'][kmonth][ex]
        
        
        
        ax.plot(lags,plotvar,
                label=expnames_long[ex],color=col_in,ls=expls[ex],lw=2.5)
        
        
        # Calcualate Confidence Interval
        #neff  = proc.calc_dof()
        
        cflag = proc.calc_conflag(plotvar,conf,2,len(ssts[ex])/12)
        if ex == 2:
            if darkmode:
                alpha = 0.15
            else:
                alpha = 0.05
        else:
            alpha = 0.15
        ax.fill_between(lags,cflag[:,0],cflag[:,1],alpha=alpha,color=col_in,zorder=3)
        
    ax.legend()
    
    ax.set_title("")
    #ax.set_title("%s SST Autocorrelation" % mons3[kmonth])
    ax.set_xlabel("Lag from %s (Months)" % mons3[kmonth])
    ax.set_ylabel("Correlation with %s. Anomalies" % (mons3[kmonth]))
    
    
    figname = "%sACF_%s_mon%02i.png" % (figpath,comparename,kmonth+1)
    if darkmode:
        figname = proc.darkname(figname)
        #figname = proc.addstrtoext(figname,"_darkmode")
    plt.savefig(figname,dpi=150,transparent=transparent)



#%% Plot Spectra
decadal_focus = False

dtmon_fix       = 60*60*24*30

if decadal_focus:
    xper            = np.array([10,5,1,0.5])
else:
    xper            = np.array([40,10,5,1,0.5])
xper_ticks      = 1 / (xper*12)

fig,ax      = plt.subplots(1,1,figsize=(8,4.5),constrained_layout=True)

for ii in range(nexps):
    if ii == nexps-1:
        col_in = dfcol
    else:
        col_in = expcols[ii]
    plotspec        = metrics_out['specs'][ii] / dtmon_fix
    plotfreq        = metrics_out['freqs'][ii] * dtmon_fix
    CCs = metrics_out['CCs'][ii] / dtmon_fix

    ax.loglog(plotfreq,plotspec,lw=2.5,label=expnames_long[ii],c=col_in)
    
    #ax.loglog(plotfreq,CCs[:,0],ls='dotted',lw=0.5,c=expcols[ii])
    #ax.loglog(plotfreq,CCs[:,1],ls='dashed',lw=0.9,c=expcols[ii])

ax.set_xlim([xper_ticks[0],0.5])
ax.axvline([1/(6)],label="",ls='dotted',c='gray')
ax.axvline([1/(12)],label="",ls='dotted',c='gray')
ax.axvline([1/(5*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(10*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(40*12)],label="",ls='dotted',c='gray')

ax.set_xlabel("Frequency (1/Month)",fontsize=14)
ax.set_ylabel("Power [$\degree C ^2 cycle \, per \, mon$]")

ax2 = ax.twiny()
ax2.set_xlim([xper_ticks[0],0.5])
ax2.set_xscale('log')
ax2.set_xticks(xper_ticks,labels=xper)
ax2.set_xlabel("Period (Years)",fontsize=14)


# Plot Confidence Interval (ERA5)
alpha           = 0.05
cloc_era        = [plotfreq[0],1e-1]
dof_era         = metrics_out['dofs'][-1]
cbnds_era       = proc.calc_confspec(alpha,dof_era)
proc.plot_conflog(cloc_era,cbnds_era,ax=ax,color=dfcol,cflabel=r"95% Confidence") #+r" (dof= %.2f)" % dof_era)

ax.legend()

figname = "%sSpectra_LogLog_%s.png" % (figpath,comparename)
if darkmode:
    figname = proc.addstrtoext(figname,"_darkmode")
plt.savefig(figname,dpi=150,bbox_inches='tight',transparent=transparent)
    
#%% Plot ACF Map

fsz_ticks = 14

pmesh            = True
xtks             = lags[::6]
acfmaps          = np.array(metrics_out['acfs']) # [Basemonth x Experiment x Lag]
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

#%% Make Barplot

remove_topright = True
expcols_bar     = np.array(expcols).copy()
expcols_bar[-1] = 'gray'
label_vratio    = False

fsz_axis        = 18
fsz_tick        = 16

instd           = stds
instd_lp        = stds_lp

if label_vratio:
    xlabs           = ["%s\n%.2f" % (expnames_long[ii],vratio[ii])+"%" for ii in range(len(vratio))]
else:
    xlabs  = expnames_short
fig,ax          = plt.subplots(1,1,constrained_layout=True,figsize=(6,6))

braw            = ax.bar(np.arange(nexps),instd,color=expcols_bar)
blp             = ax.bar(np.arange(nexps),instd_lp,color=dfcol)

if darkmode:
    upperlbl_col = 'lightgray'
else:
    upperlbl_col = 'gray'

ax.bar_label(braw,fmt="%.02f",c=upperlbl_col,fontsize=fsz_axis)
ax.bar_label(blp,fmt="%.02f",c=dfcol,fontsize=fsz_axis)

#ax.bar_label(vratio,fmt="%.2f",c=w,label_type='bottom')

ax.set_xticks(np.arange(nexps),labels=xlabs,fontsize=fsz_tick,rotation=45)
ax.set_ylabel("$\sigma$(SST) [$\degree$C]",fontsize=fsz_axis)
ax.set_ylim([0,1.0])
ax.tick_params(labelsize=fsz_tick)

if remove_topright:
    ax.spines[['right', 'top']].set_visible(False)
    
savename = "%sBarplots_%s_%s.png" % (figpath,comparename,expnames[ex])
if darkmode:
    figname = proc.addstrtoext(figname,"_darkmode")
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=transparent)

#%% Plot the Monthly Variance

fsz_legend=12

fig,ax = viz.init_monplot(1,1,figsize=(8,4.5))

for ex in range(nexps):
    
    col_in = expcols[ex]
    if darkmode and col_in =="k":
        col_in = dfcol
        
    plotvar = metrics_out['monvars'][ex]
    
    ax.plot(mons3,plotvar,label=expnames_long[ex],c=col_in,lw=2.5,ls=expls[ex])


ax.set_ylim([-.2,.5])
ax.set_ylabel("SST Variance [$\degree C^2$]",fontsize=fsz_axis)
ax.tick_params(labelsize=fsz_tick)
ax.legend(fontsize=fsz_legend,ncol=2)

savename = "%sMonvar_%s.png" % (figpath,comparename)
if darkmode:
    figname = proc.addstrtoext(figname,"_darkmode")
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=transparent)


# ====================================================
#%% Plot ACF for Winter and Summer (For Paper Outline)
# ====================================================

fsz_title = 26
fsz_ticks = 14
plotkmons = [1,6]
fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(8,6.5))

for ii in range(2):
    ax = axs[ii]
    kmonth = plotkmons[ii]
    
    ax,_   = viz.init_acplot(kmonth,xtks,lags,ax=ax,title="")
    for ex in range(nexps):
        
        if ex == nexps-1:
            col_in = dfcol
        else:
            col_in = expcols[ex]
        
        plotvar = metrics_out['acfs'][kmonth][ex]
        
        ax.plot(lags,plotvar,
                label=expnames_long[ex],color=col_in,ls=expls[ex],lw=2.5)
        
        cflag = proc.calc_conflag(plotvar,conf,2,len(ssts[ex])/12)
        if ex == 2:
            if darkmode:
                alpha = 0.15
            else:
                alpha = 0.05
        else:
            alpha = 0.15
        ax.fill_between(lags,cflag[:,0],cflag[:,1],alpha=alpha,color=col_in,zorder=3)
        
    if ii == 0:
        ax.set_xlabel("")
    else:
        ax.legend(framealpha=0,fontsize=fsz_ticks,ncol=2)
        
        
    ax.set_ylim([-.25,1])
    
    ax.set_title("")
    ax.set_ylabel("Correlation with \n %s. Anomalies" % (mons3[kmonth]),fontsize=fsz_tick)
    
    viz.label_sp(ii,alpha=0.15,ax=ax,fontsize=fsz_title,y=1.28,x=-.15,
                 fontcolor=dfcol)
    
    
    ax.tick_params(labelsize=fsz_ticks)
    
    
figname = "%sACF_%s_PaperOutline.png" % (figpath,comparename)
if darkmode:
    figname = proc.darkname(figname)
plt.savefig(figname,dpi=150,transparent=transparent)

# ---------------------------------
#%% Combine the Spectra and Barplot
# ---------------------------------


fig       = plt.figure(figsize=(14,4))
gs        = gridspec.GridSpec(4,12)

# --------------------------------- # Barplot
ax11            = fig.add_subplot(gs[:,:3],)
ax              = ax11

remove_topright = True
expcols_bar     = np.array(expcols).copy()
expcols_bar[-1] = 'gray'
label_vratio    = False

fsz_axis         = 18
fsz_ticks        = 16
fsz_legend       = 14

instd           = stds
instd_lp        = stds_lp

if label_vratio:
    xlabs           = ["%s\n%.2f" % (expnames_long[ii],vratio[ii])+"%" for ii in range(len(vratio))]
else:
    xlabs  = expnames_short

braw            = ax.bar(np.arange(nexps),instd,color=expcols_bar)
blp             = ax.bar(np.arange(nexps),instd_lp,color=dfcol)

if darkmode:
    upperlbl_col = 'lightgray'
else:
    upperlbl_col = 'gray'

ax.bar_label(braw,fmt="%.02f",c=upperlbl_col,fontsize=fsz_axis)
ax.bar_label(blp,fmt="%.02f",c=dfcol,fontsize=fsz_axis)

# --- Make Fake Legend'
colorsf = {'Raw':'gray','10-year Low-pass':'k',}         
labels = list(colorsf.keys())
handles = [plt.Rectangle((0,0),1,1, color=colorsf[label]) for label in labels]
ax.legend(handles, labels,fontsize=fsz_legend,framealpha=0)
#ax.legend()
#ax.bar_label(vratio,fmt="%.2f",c=w,label_type='bottom')

ax.set_xticks(np.arange(nexps),labels=xlabs,fontsize=fsz_tick,rotation=45)
ax.set_ylabel("$\sigma$(SST) [$\degree$C]",fontsize=fsz_axis)
ax.set_ylim([0,1.0])
ax.tick_params(labelsize=fsz_tick)

if remove_topright:
    ax.spines[['right', 'top']].set_visible(False)

viz.label_sp(0,alpha=0.15,ax=ax,fontsize=fsz_title,y=1.10,x=-.45,
             fontcolor=dfcol)



# --------------------------------- # Power Spectra
ax22       = fig.add_subplot(gs[:,4:])

ax         = ax22

decadal_focus = False

dtmon_fix       = 60*60*24*30

if decadal_focus:
    xper            = np.array([10,5,1,0.5])
else:
    xper            = np.array([40,10,5,1,0.5])
xper_ticks      = 1 / (xper*12)

for ii in range(nexps):
    if ii == nexps-1:
        col_in = dfcol
    else:
        col_in = expcols[ii]
    plotspec        = metrics_out['specs'][ii] / dtmon_fix
    plotfreq        = metrics_out['freqs'][ii] * dtmon_fix
    CCs = metrics_out['CCs'][ii] / dtmon_fix

    ax.loglog(plotfreq,plotspec,lw=2.5,label=expnames_long[ii],c=col_in)
    
    #ax.loglog(plotfreq,CCs[:,0],ls='dotted',lw=0.5,c=expcols[ii])
    #ax.loglog(plotfreq,CCs[:,1],ls='dashed',lw=0.9,c=expcols[ii])

ax.set_xlim([xper_ticks[0],0.5])
ax.axvline([1/(6)],label="",ls='dotted',c='gray')
ax.axvline([1/(12)],label="",ls='dotted',c='gray')
ax.axvline([1/(5*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(10*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(40*12)],label="",ls='dotted',c='gray')

ax.set_xlabel("Frequency (1/month)",fontsize=fsz_axis)
ax.set_ylabel("Power [$\degree C ^2 / cycle \, per \, mon$]",fontsize=fsz_axis)

ax2 = ax.twiny()
ax2.set_xlim([xper_ticks[0],0.5])
ax2.set_xscale('log')
ax2.set_xticks(xper_ticks,labels=xper)
ax2.set_xlabel("Period (Years)",fontsize=fsz_axis)

# Plot Confidence Interval (ERA5)
alpha           = 0.05
cloc_era        = [6e-2,5e-2]
dof_era         = metrics_out['dofs'][-1]
cbnds_era       = proc.calc_confspec(alpha,dof_era)
proc.plot_conflog(cloc_era,cbnds_era,ax=ax,color=dfcol,cflabel=r"95% Confidence") #+r" (dof= %.2f)" % dof_era)
ax.legend(fontsize=fsz_legend,framealpha=0.5,edgecolor='none')

for ax in [ax,ax2]:
    ax.tick_params(labelsize=fsz_ticks)
    
viz.label_sp(1,alpha=0.15,ax=ax,fontsize=fsz_title,y=1.10,x=-.1,
             fontcolor=dfcol)

figname = "%sVariance_%s_PaperOutline.png" % (figpath,comparename)
if darkmode:
    figname = proc.darkname(figname)
plt.savefig(figname,dpi=150,transparent=transparent,bbox_inches='tight')
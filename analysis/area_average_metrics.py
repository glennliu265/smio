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
figpath         = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20251106/"
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
expnames_short  = ["SM_NoREM","SM_REM","ERA5"]
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
expnames        = ["SST_Obs_Pilot_00_qnet_AConly_SPGNE_noREM","SST_ORAS5_avg","SST_ERA5_1979_2024"]
expnames_long   = ["Stochastic Model","Stochastic Model (with re-emergence)","ERA5"]
expnames_short  = ["SM","SM (+REM)","ERA5"]
expcols         = ["goldenrod","turquoise","k"]
expls           = ["dashed","dotted",'solid']

# # (7) Compare different subsurface damping
# comparename     = "CompareSubsurface"
# expnames        = ["SST_Obs_Pilot_00_qnet_AConly_SPG_addlbdd","SST_Obs_Pilot_00_qnet_AConly_SPGNE_ORAS5","SST_ORAS5_avg","SST_ERA5_1979_2024"]
# expnames_long   = ["EN4","ORAS5 (opa0)","ORAS5 (Ens. Avg.)","ERA5"]
# expnames_short  = ["EN4","ORAS5_opa0","ORAS5_avg","ERA5"]
# expcols         = ["red","cornflowerblue","navy","k"]
# expls           = ["dashed","dotted","dotted",'solid']


# # (8) ORAS5 MLD Comparison
# comparename     = "MLDComparison"
# expnames        = ["SST_ORAS5_avg","SST_ORAS5_avg_mld003","SST_ERA5_1979_2024"]
# expnames_long   = ["MIMOC MLD","ORAS5 MLD","ERA5"]
# expnames_short  = ["MIMOC","ORAS5","ERA5"]
# expcols         = ["hotpink","navy","k"]
# expls           = ["dotted","dashed",'solid']


# (9) Forcing EOF vs Std Comparison
comparename     = "ForcingComparison"
expnames        = ["SST_ORAS5_avg","SST_ORAS5_avg_EOF","SST_ORAS5_avg_GMSSTmon_EOF","SST_ERA5_1979_2024"]
expnames_long   = ["stdev(F') Forcing","EOF-based Forcing","EOF-based Forcing with GMSSTmon Detrend","ERA5"]
expnames_short  = ["Fstd","EOF","EOF_GMSSTmon","ERA5"]
expcols         = ["hotpink","navy","cornflowerblue","k"]
expls           = ["dotted","dashed","dashed",'solid']
detect_blowup   = True

# (10) Linear Detrend (All Month) vs GMSST Removal (Sep Month)
comparename     = "DetrendingEffect"
expnames        = ["SST_ORAS5_avg","SST_ORAS5_avg_GMSST","SST_ORAS5_avg_GMSSTmon","SST_ERA5_1979_2024"]
expnames_long   = ["LinearDetrend","GMSST","GMSSTmon","ERA5"]
expnames_short  = ["LinearDetrend","GMSST","GMSSTmon","ERA5"]
expcols         = ["hotpink","cornflowerblue","navy","k"]
expls           = ["dotted",'solid',"dashed",'solid']
detect_blowup   = True


# # (11) Forcing Repair
# comparename     = "ForcingComparison"
# expnames        = ["SST_ORAS5_avg","SST_ORAS5_avg_GMSSTmon_EOF","SST_ORAS5_avg_GMSSTmon_EOF_usevar","SST_ERA5_1979_2024"]
# expnames_long   = ["stdev(F') Forcing","EOF-based Forcing","EOF-based Forcing (corrected)","ERA5"]
# expnames_short  = ["Fstd","EOF","EOF_corrected","ERA5"]
# expcols         = ["hotpink","navy","cornflowerblue","k"]
# expls           = ["dotted","dashed","dashed",'solid']
# detect_blowup   = True


# (12) Draft 2 Edition (using case with GMSST Mon Detrend)
comparename     = "Draft02"
expnames        = ["SST_ORAS5_avg_GMSST_EOFmon_usevar_NoRem_NATL","SST_ORAS5_avg_GMSST_EOFmon_usevar_NATL","SST_ERA5_1979_2024"]
expnames_long   = ["Stochastic Model (no re-emergence)","Stochastic Model (with re-emergence)","ERA5"]
expnames_short  = ["SM_NoREM","SM_REM","ERA5"]
expcols         = ["goldenrod","turquoise","k"]
expls           = ["dashed","dotted",'solid']
detect_blowup   = True

# (13) Draft 3 Edition (using case with GMSST Mon Detrend)
comparename     = "Draft03_FullHierachy"
expnames        = ["SST_ORAS5_avg_GMSST_EOFmon_usevar_SOM_NATL",
                   "SST_ORAS5_avg_GMSST_EOFmon_usevar_SOM_NATL_MLDvar",
                   "SST_ORAS5_avg_GMSST_EOFmon_usevar_NoRem_NATL",
                   "SST_ORAS5_avg_GMSST_EOFmon_usevar_NATL",
                   "SST_ERA5_1979_2024"]

# expnames_long   = ["Stochastic Model 0 (SOM)",
#                    "Stochastic Model 1 (Add seasonal MLD)",
#                    "Stochastic Model 2 (Add entrain damping)",
#                    "Stochastic Model 3 (Add entrain forcing)",
#                    "ERA5"]
# expnames_short  = ["SM_SOM",
#                    "SM_MLDvar",
#                    "SM_NoREM",
#                    "SM_REM",
#                    "ERA5"]

expnames_long   = ["Level 1",
                   "Leve 1.5 (Seasonal MLD only)",
                   "Level 2 (Entrainment Damping)",
                   "Level 3 (Entrainment Damping + Re-emergence)",
                   "Observations (ERA5)"]
expnames_short  = ["Level 1",
                   "Level 1.5",
                   "Level 2",
                   "Level 3",
                   "Obs."]
expcols         = ["salmon",
                   "darkviolet",
                   "goldenrod",
                   "turquoise",
                   "k"]
expls           = ['dotted',
                   'dotted',
                   'dashed',
                   'dashed',
                   'solid',
                   'solid']

# # (13) Draft 3 Edition (Add SOM)
# comparename     = "Draft03_AddSOM"
# expnames        = ["SST_ORAS5_avg_GMSST_EOFmon_usevar_SOM_NATL",
#                    "SST_ORAS5_avg_GMSST_EOFmon_usevar_NoRem_NATL",
#                    "SST_ORAS5_avg_GMSST_EOFmon_usevar_NATL",
#                    "SST_ERA5_1979_2024"]

# expnames_long   = ["Level 1",
#                    "Level 2 (Entrainment Damping)",
#                    "Level 3 (Entrainment Damping + Re-emergence)",
#                    "Observations (ERA5)"]
# expnames_short  = ["Level 1",
#                    "Level 2",
#                    "Level 3",
#                    "Obs."]
# expcols         = ["salmon",
#                    "goldenrod",
#                    "turquoise",
#                    "k"]
# expls           = ['dotted',
#                    'dashed',
#                    'dashed',
#                    'solid']

# detect_blowup   = True




#%% Load information for each region

dsall = []
nexps = len(expnames)
for ex in tqdm.tqdm(range(nexps)):
    expname = expnames[ex]
    ncname  = "%s%s/Metrics/Area_Avg_%s.nc" % (sm_output_path,expname,bbfn)
    
    ds      = xr.open_dataset(ncname).load()
    dsall.append(ds)
    
#%% Temp Fix, remove chunks where the model blows up...

def detect_blowup(ds,vname,chunk,thres):
    ntime = ds[vname].shape[0]
    niter = int(ntime/chunk)
    
    for ii in range(niter):
        
        # Get Chunk
        id0 = ii*chunk
        id1 = (ii+1) * chunk
        
        #ids = np.arange(ii*chunk,(ii+1)*chunk)
        dschunk = ds[vname].isel(time=slice(id0,id1))
        
        if np.any(np.abs(dschunk) > thres):
            
            print("Exceeded threshold in chunk %i to %i" % (id0,id1))
            dsbefore = ds.isel(time=slice(None,id0))
            dsafter  = ds.isel(time=slice(id1,None))
            ds = xr.concat([dsbefore,dsafter],dim='time')

    return ds

if detect_blowup:
    dsall = [detect_blowup(ds,'SST',12000,10) for ds in dsall]
    

    
#%% Preprocessing (Deseason Only)

dsa      = [proc.xrdeseason(ds) for ds in dsall]
ssts     = [sp.signal.detrend(ds.SST.data) for ds in dsa] # Detrend for Stochastic Model
ssts_ds  = [proc.xrdetrend(ds.SST) for ds in dsa] # Detrend for Stochastic Model

#%% Detrend ERA5 using Global Mean Regression Approach
if (detrend_obs_regression):
    
    sst_era = dsa[-1].SST # Take undetrended ERA5 SST
    sst_era = sst_era.expand_dims(dim=dict(lon=1,lat=1))
    
    # Add dummy lat lon
    #sst_era['lon'] = 1
    #sst_era['lat'] = 1
    
    # Load GMSST
    ds_gmsst     = xr.open_dataset(dpath_gmsst + nc_gmsst).GMSST_MaxIce.load()
    dsdtera5     = proc.detrend_by_regression(sst_era,ds_gmsst,regress_monthly=True)
    
    sst_era_dt = dsdtera5.SST.squeeze()
    
    ssts_ds[-1] = sst_era_dt
    ssts[-1]    = sst_era_dt.data
    #print("\nSkipping ERA5, loading separately")
    
#%% Compute basic metrics

ssts         = [ds.data for ds in ssts_ds]
lags         = np.arange(61)
nsmooths     = [250,] * (nexps-1) + [4,]
 #nsmooths     = [250,250,4]
metrics_out  = scm.compute_sm_metrics(ssts,nsmooth=nsmooths,lags=lags,detrend_acf=False)

monstds      = [ss.groupby('time.month').std('time') for ss in ssts_ds]

#%% Compute some additional metrics

# Calculate Standard Deviations and Low-Pass Filtered Variance
stds     = np.array([ds.std('time').data.item() for ds in ssts_ds])
ssts_lp  = [proc.lp_butter(ts,120,6) for ts in ssts_ds]
stds_lp  = np.array([np.std(ds) for ds in ssts_lp])
vratio   = (stds_lp  / stds) * 100

# Calculate High Pass Filtered variance as well
ssts_hp  = [ssts_ds[ii] - ssts_lp[ii] for ii in range(nexps)]
stds_hp  = np.array([np.std(ds) for ds in ssts_hp])

#%% Check Distribution of Anomalies

kmonth  = None

for kmonth in range(13):
    
    if kmonth == 12:
        kmonth = None
    hbins   = np.arange(-2,2.1,0.1)
    fig,axs = plt.subplots(4,1,figsize=(6,10),constrained_layout=True)
    
    for ex in range(nexps):
        
        ax      = axs[ex]
        
        plotvar = ssts[ex]
        
        if kmonth is not None:
            plotvar = plotvar[kmonth::12]
            
        
        mu      = plotvar.mean()
        sigma   = plotvar.std()
        
        #print(plotvar.mean())
        ax.hist(plotvar,bins=hbins,edgecolor="w",color=expcols[ex],alpha=0.75,density=True)
        
        ax.axvline(mu,lw=0.75,c='k',label="$\mu$=%.2e" % mu)
        ax.axvline(-sigma,lw=0.75,c='k',ls='dashed')
        ax.axvline(-2*sigma,lw=0.75,c='k',ls='dotted')
        ax.axvline(sigma,lw=0.75,c='k',ls='dashed',label="$1\sigma$=%.2f" % sigma)
        ax.axvline(2*sigma,lw=0.75,c='k',ls='dotted',label="$2\sigma$=%.2f" % (2*sigma))
        
        
        ax.set_xlim([-2,2])
        ax.set_title("%s, Skew=%.2f" % (expnames_long[ex],sp.stats.skew(plotvar)))
        
        pdffit = sp.stats.norm.pdf(hbins,mu,sigma)
        ax.plot(hbins,pdffit,label="PDF Fit",color='gray',lw=.55)
        
        ax.legend()
        
        if ex == 2:
            if kmonth is not None:
                ax.set_xlabel("%s SST Anomaly [$\degree C$]" % mons3[kmonth])
            else:
                ax.set_xlabel("SST Anomaly [$\degree C$]")
        if ex == 1:
            ax.set_ylabel("Frequency")
        
        viz.label_sp(ex,ax=ax,fig=fig)
    
    if kmonth is not None:
        figname = "%sSM_vs_ERA5_Normality_Check_%s_mon%02i.png" % (figpath,comparename,kmonth+1)
    else:
        figname = "%sSM_vs_ERA5_Normality_Check_%s.png" % (figpath,comparename)
    plt.savefig(figname,dpi=150,bbox_inches='tight')





#%% Plot ACF
kmonth = 1
xtks   = lags[::6]
conf   = 0.95
use_neff = True
dofs_eff = np.zeros((nexps,12)) * np.nan

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
        if use_neff:
            plotvar_mon = ssts[ex][kmonth::12]
            dof_in      = proc.calc_dof(plotvar_mon,calc_r1=True,r1_in=None)
            dofs_eff[ex,kmonth] = dof_in
        else:
            dof_in = len(ssts[ex])/12
        
        cflag = proc.calc_conflag(plotvar,conf,2,dof_in)
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
    
    figname = "%sACF_%s_neff%i_mon%02i.png" % (figpath,comparename,use_neff,kmonth+1)
    if darkmode:
        figname = proc.darkname(figname)
        #figname = proc.addstrtoext(figname,"_darkmode")
    plt.savefig(figname,dpi=150,transparent=transparent)
    
#%% Plot ACF minigrid

plot_im  = np.roll(np.arange(12),1)
fig,axs = plt.subplots(4,3,constrained_layout=True,figsize=(10,8))

alphalist = list(map(chr, range(97, 123)))
alphalist_upper = [s.upper() for s in alphalist]

for mm in range(12):
    
    ax     = axs.flatten()[mm]
    kmonth = plot_im[mm]
    lab    = "%s) %s" % (alphalist_upper[mm],mons3[kmonth])
    viz.label_sp(lab,ax=ax,labelstyle="%s",usenumber=True)
    if mm == 10:
        ax.set_xlabel("Lag [month]")
    
    
    # Plot ACFs (copied from above) ===========================================
    for ex in range(nexps):
        if ex == nexps-1:
            col_in = dfcol
        else:
            col_in = expcols[ex]
        
        plotvar = metrics_out['acfs'][kmonth][ex]
        ax.plot(lags,plotvar,
                label=expnames_long[ex],color=col_in,ls=expls[ex],lw=2.5)
        # Calcualate Confidence Interval
        if use_neff:
            plotvar_mon = ssts[ex][kmonth::12]
            dof_in      = proc.calc_dof(plotvar_mon,calc_r1=True,r1_in=None)
            dofs_eff[ex,kmonth] = dof_in
        else:
            dof_in = len(ssts[ex])/12
        cflag = proc.calc_conflag(plotvar,conf,2,dof_in)
        if ex == 2:
            if darkmode:
                alpha = 0.15
            else:
                alpha = 0.05
        else:
            alpha = 0.15
        ax.fill_between(lags,cflag[:,0],cflag[:,1],alpha=alpha,color=col_in,zorder=3)
        
    # =========================================================================
    
    ax.set_xlim([0,60])
    ax.set_ylim([-0.25,1.25])
    ax.set_xticks(np.arange(0,66,6))
    ax.set_yticks(np.arange(0,1.25,0.25))
    ax.axhline([0],ls='dashed',c='k',lw=0.55)
    if mm == 0:
        ax.legend(fontsize=7,framealpha=.0)
    
    
figname = "%sACF_%s_neff%i_AllMonths.png" % (figpath,comparename,use_neff)
plt.savefig(figname,dpi=150,transparent=transparent) 

#%%
#viz.add_ylabel("Correlation",ax=ax)#%% Plot Effective DOF

fig,axs =viz.init_monplot(3,1,figsize=(6,8.5))

for ex in range(nexps):
    ax = axs[ex]
    blb = ax.bar(mons3,dofs_eff[ex,:],label=expnames_long[ex],alpha=1,color=expcols[ex],)
    ax.bar_label(blb,fmt="%i")
    
    if ex == 1:
        ax.set_ylabel("Degrees of Freedom")
        
    if ex < 2:
        ax.set_ylim([0,10000])
    else:
        ax.set_ylim([0,50])
    
    ax.set_xlim([-1,12])



#%% Plot Spectra
decadal_focus = True

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
instd_hp        = stds_hp

if label_vratio:
    xlabs           = ["%s\n%.2f" % (expnames_long[ii],vratio[ii])+"%" for ii in range(len(vratio))]
else:
    xlabs       = expnames_short
fig,ax          = plt.subplots(1,1,constrained_layout=True,figsize=(6,6))

braw            = ax.bar(np.arange(nexps),instd,color=expcols_bar)
blp             = ax.bar(np.arange(nexps),instd_lp,color=dfcol)
#bhp             = ax.bar(np.arange(nexps),instd_hp,color='w')

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

#%% Make Barplot with high pass comparison


# Abandoned project ot make grouped barplot
import pandas as pd
import seaborn as sns
stds_all = np.zeros((nexps,3)) * np.nan
stds_all[:,0] = np.array(stds)
stds_all[:,1] = np.array(stds_lp)
stds_all[:,2] = np.array(stds_hp)

metric_names =["std","std_lp","std_hp"]
coords       = dict(exp=expnames_short,metric=metric_names)
stds_ds      = xr.DataArray(stds_all,coords=coords,dims=coords)

stds_metrics = {
    'std':stds_all[:,0],
    'std_lp':stds_all[:,1],
    'std_hp':stds_all[:,2],
    }

x = np.arange(nexps)  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig,ax = plt.subplots(1,1,layout='constrained')

for attribute, measurement in stds_metrics.items():
    offset = width * multiplier
    
    edc = 'none'
    if multiplier == 0:
        color_in = expcols_bar
    elif multiplier == 1:
        color_in = "k"
    else:
        color_in = 'w',#"lightgray"
        edc = 'k'
    
    rects  = ax.bar(x + offset, measurement, width, label=attribute,
                    color=color_in,edgecolor=edc,linewidth=0.5)
    
    ax.set_xticks(x,expnames_short)
    ax.bar_label(rects, padding=3,fmt="%.02f")
    multiplier += 1
    
ax.legend()







#%% Plot the Monthly Variance

fsz_legend= 10
fsz_tick  = 18
fig,ax = viz.init_monplot(1,1,figsize=(10,4.5))

for ex in range(nexps):
    
    col_in = expcols[ex]
    if darkmode and col_in =="k":
        col_in = dfcol
        
    plotvar = metrics_out['monvars'][ex]
    
    ax.plot(mons3,plotvar,label=expnames_long[ex],c=col_in,lw=3,ls=expls[ex])

ax.set_ylim([-.2,.5])
ax.set_ylabel("SST Variance [$\degree C^2$]",fontsize=fsz_axis)
ax.tick_params(labelsize=fsz_tick)
ax.legend(fontsize=fsz_legend,ncol=2)

savename = "%sMonvar_%s.png" % (figpath,comparename)
if darkmode:
    figname = proc.addstrtoext(figname,"_darkmode")
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=transparent)




#%% Plot the monthly variance as a percentage

ex      = 1
pct_exp       = (metrics_out['monvars'][ex] / metrics_out['monvars'][-1])*100
pct_exp_norem = (metrics_out['monvars'][0] / metrics_out['monvars'][-1])*100

fig,ax  = viz.init_monplot(1,1,figsize=(8,4.5))
bbar        = ax.bar(mons3,pct_exp,color=expcols[ex])
ax.bar_label(bbar,fmt="%.01f",c=dfcol,fontsize=12)

bbar_norem  = ax.bar(mons3,pct_exp_norem,color=expcols[0],alpha=0.90)
ax.bar_label(bbar_norem,fmt="%.01f",c=dfcol,fontsize=12)

ax.set_xlim([-1,12])
ax.set_ylim([0,100])
ax.set_title("Percent Variance Explained by %s" % (expnames_long[ex]))

savename = "%sPercent_Variance_ERA5_Explained_by_%s_%s.png" % (figpath,expnames[ex],comparename)

if darkmode:
    figname = proc.addstrtoext(figname,"_darkmode")
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=transparent)

#%% Try to compute the exponential fit

kmonth = 4

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,4))
ax,_   = viz.init_acplot(kmonth,xtks,lags,ax=ax)

for ex in range(nexps):
    
    if ex == nexps-1:
        col_in = dfcol
    else:
        col_in = expcols[ex]
    
    plotvar = metrics_out['acfs'][kmonth][ex]
    
    
    #expfit_out = proc.expfit(plotvar[[0,11,23]],lags[[0,11,23]],60)
    
    expfit_out = proc.expfit(plotvar[::12],lags[::12],60)
    lagstheo       = np.linspace(0,60,100)
    acftheo        = np.exp(expfit_out['tau_inv']*lagstheo) 
    
    
    
    
    ax.plot(lags,plotvar,
            label=expnames_long[ex],color=col_in,ls=expls[ex],lw=2.5)
    
    
    ax.plot(lagstheo,acftheo,
            label=expnames_long[ex] + ", e-folding=%.2f months" % (-1*1/expfit_out['tau_inv']),color=col_in,ls='solid',lw=1)
    
    # ax.plot(lags[::12],expfit_out['acf_fit'],
    #         label=expnames_long[ex] + "Efold=%.2f months" % (-1*1/expfit_out['tau_inv']),color=col_in,ls='solid',lw=1)
    
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

#%% Repeat computation of expfit for each month

sellags_fit = lags[::12]

taus = np.zeros((nexps,12)) * np.nan

for kmonth in range(12):
    for ex in range(nexps):
        
        if ex == nexps-1:
            col_in = dfcol
        else:
            col_in = expcols[ex]
        
        plotvar = metrics_out['acfs'][kmonth][ex]
        expfit_out = proc.expfit(plotvar[sellags_fit],lags[sellags_fit],60)
        taus[ex,kmonth] = 1/expfit_out['tau_inv'] * -1
        
        #expfit_out = proc.expfit(plotvar[[0,11,23]],lags[[0,11,23]],60)
        #lagstheo       = np.linspace(0,60,100)
        #acftheo        = np.exp(expfit_out['tau_inv']*lagstheo) 
        
fig,ax = viz.init_monplot(1,1,figsize=(8.5,4.5))
for ex in range(nexps):
    ax.plot(mons3,taus[ex,:],color=expcols[ex],ls=expls[ex],lw=2.5,label=expnames_long[ex])
    
ax.legend()
ax.set_xlabel("Base Month")
ax.set_ylabel("E-folding Timescale (Months)")
ax.set_yticks(np.arange(0,66,6))
ax.set_title("Monthly E-Folding Timescale (5-year fit)")

# =============================================================================
#%% Bootstrap Spectra
# =============================================================================
# Based on tomoki's suggestion, try bootstrapping 

nsmooth = 4
mciter  = 10000
ex      = 0
pct     = 0.10

eraspec_dict = scm.quick_spectrum([ssts[-1],],[nsmooth,],pct,return_dict=True)

stochmod_conts     = []
mc_specdicts       = []

for ex in range(nexps-1):
    stochmod_ts       = ssts[ex]
    
    ntime_era         = len(ssts[-1])
    mcdict            = proc.mcsampler(stochmod_ts,ntime_era,mciter)
    stochmod_samples  = [mcdict['samples'][ii,:] for ii in range(mciter)]
    
    stochmod_specdict = scm.quick_spectrum(stochmod_samples,[nsmooth,]*mciter,pct,return_dict=True,make_arr=True)
    specdict_cont     = scm.quick_spectrum([stochmod_ts,],[250,]*mciter,pct,return_dict=True)
    stochmod_conts.append(specdict_cont)
    mc_specdicts.append(stochmod_specdict)

#%% =====  make the plot

ex                = 1
stochmod_specdict = mc_specdicts[ex]
stochmod_cont     = stochmod_conts[ex]

def init_specplot():
    
    
    
    fig,ax          = plt.subplots(1,1,figsize=(12.5,4.5))
    
    decadal_focus   = True
    obs_cutoff      = 10 # in years
    obs_cutoff      = 1/(obs_cutoff*12)
    
    dtmon_fix       = 60*60*24*30
    
    if decadal_focus:
        xper            = np.array([20,10,5,1,0.5])
    else:
        xper            = np.array([40,10,5,1,0.5])
    xper_ticks      = 1 / (xper*12)
    
    ax.set_xlim([xper_ticks[0],0.5])
    ax.axvline([1/(6)],label="",ls='dotted',c='gray')
    ax.axvline([1/(12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(5*12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(10*12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(20*12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(40*12)],label="",ls='dotted',c='gray')
    
    ax.set_xlabel("Frequency (1/month)",fontsize=fsz_axis)
    ax.set_ylabel("Power [$\degree C ^2 / cycle \, per \, mon$]",fontsize=fsz_axis)
    
    ax2 = ax.twiny()
    ax2.set_xlim([xper_ticks[0],0.5])
    ax2.set_xscale('log')
    ax2.set_xticks(xper_ticks,labels=xper)
    ax2.set_xlabel("Period (Years)",fontsize=fsz_axis)
    
    return fig,ax

fig,ax = init_specplot()

# Plot ERA
plotspec  = eraspec_dict['specs'][0] / dtmon_fix
plotfreq  = eraspec_dict['freqs'][0] *dtmon_fix
ax.loglog(plotfreq,plotspec,marker="o",markersize=5,color="k",label="ERA5")

# Plot Stochastic Model
muspec    = stochmod_specdict['specs'].mean(0) / dtmon_fix
plotfreq  = stochmod_specdict['freqs'][0,:] * dtmon_fix
ax.loglog(plotfreq,muspec,marker="x",color="blue",label="Stochastic Model, %i-Sample Mean" % mciter)

for mc in range(mciter):

    sampspec    = stochmod_specdict['specs'][mc,:] / dtmon_fix
    ax.loglog(plotfreq,sampspec,c=expcols[ex],alpha=0.01,zorder=1,label="")
    

bnds = np.quantile(stochmod_specdict['specs'] /dtmon_fix ,[0.025,0.975],axis=0)
ax.loglog(plotfreq,bnds[0],ls='dotted',color='blue',label="95% Conf.")
ax.loglog(plotfreq,bnds[1],ls='dotted',color='blue')

# Plot Stochastic Model (Full Timeseries)
plotspec  = specdict_cont ['specs'][0] / dtmon_fix
plotfreq  = specdict_cont ['freqs'][0] *dtmon_fix
ax.loglog(plotfreq,plotspec,marker=".",markersize=1,color="violet",label="Stochastic Model, Continous Timeseries")

ax.legend()

figname = "%sSpectra_Confidence_test_mciter%i_%s.png" % (figpath,mciter,expnames[ex])
plt.savefig(figname,dpi=150,bbox_inches='tight')


#%% Check Diferent smoothings and impact on ERA5 SSTs


nsmooths_test = [1,2,3,4,5,10,25]

era_ts = ssts[-1]


erasmooth_test = scm.quick_spectrum([era_ts,]*len(nsmooths_test),nsmooths_test,pct,return_dict=True,make_arr=True)

fig,ax         = init_specplot()

for nn in range(len(nsmooths_test)):
    
    # Plot ERA
    plotspec  = erasmooth_test['specs'][nn,:] / dtmon_fix
    plotfreq  = erasmooth_test['freqs'][nn,:] * dtmon_fix
    ax.loglog(plotfreq,plotspec,marker="o",markersize=5,label="nsmooth=%i" % nsmooths_test[nn],alpha=0.75)
ax.legend()
    



figname = "%sSpectra_Smoothing_Test_ERA5.png" % (figpath)
plt.savefig(figname,dpi=150,bbox_inches='tight')

    
#%% Bootstrapping the standard deviations and monthly variance

mcstds      = []
mcstds_lp   = []

monstds_sample = []

for ex in tqdm.tqdm(range(nexps-1)):
    stochmod_ts         = ssts[ex]
    
    ntime_era           = len(ssts[-1])
    mcdict              = proc.mcsampler(stochmod_ts,ntime_era,mciter)
    stochmod_samples    = [mcdict['samples'][ii,:] for ii in range(mciter)]
    
    stochmod_samples_lp = [proc.lp_butter(ts,120,6) for ts in stochmod_samples]
    
    # Reshape to mon x year then take standard deviation
    monstd_mc = np.array(stochmod_samples).reshape(mciter,int(ntime_era/12),12).std(1)
    
    
    mcstds.append( np.nanstd(np.array(stochmod_samples),1) )
    mcstds_lp.append( np.nanstd(np.array(stochmod_samples_lp),1) )
    monstds_sample.append(monstd_mc)
    

#%% Check Distribution of Variance

bins    = np.arange(0,0.61,0.01)

if "Draft03" in comparename: # Draft 3 Version
    fig,axs = plt.subplots(3,2,constrained_layout=True,figsize=(10,6))

    iilab = [0,4,1,5,2,]
    ii    = 0
    for ex in range(3):
        
        for vv in range(2):
            
            if vv == 0: # Raw
                mcstds_in = mcstds
                stds_in   = stds
                xlm       = [0.25,0.60]
                xlab      = "$\sigma(SST)$"
                
            else:       # LP Filter
                mcstds_in = mcstds_lp
                stds_in   = stds_lp
                xlm       = [0,0.50]
                xlab      = "10-year LP Filtered $\sigma(SST)$"
            
        
            
            ax = axs[ex,vv]
            
            ax.hist(mcstds_in[ex],bins=bins,color=expcols[ex],edgecolor='w',density=True)
            
            bnds = np.quantile(mcstds_in[ex],[0.025,0.975])
            mu   = np.nanmean(mcstds_in[ex])
            
            ax.axvline(stds_in[-1],color="k",label="Obs. = %.2f" % stds_in[-1])
            
            
            
            ax.axvline(stds_in[ex],color="blue",label="$\mu$ (Full Timeseries) = %.2f" % stds_in[ex])
            ax.axvline(mu,label="$\mu$ (Samples)= %.2f" % mu,ls='solid',color='gray')
            cflab = r"95%% Bounds: [%.2f, %.2f]" % (bnds[0],bnds[1])
            ax.axvline(bnds[0],label=cflab,ls='dashed',color="gray")
            ax.axvline(bnds[1],label="",ls='dashed',color="gray")
            
            ax.set_xlim(xlm)
            ax.set_ylim([0,25])
            ax.legend()
            
            # csfit   = sp.stats.chi2.fit(mcstds[1])
            # pdftheo = sp.stats.chi2.pdf(bins,df=csfit[0])
            # ax.plot(bins,pdftheo)
            if ex == 1:
                ax.set_xlabel("%s [$\degree$ C]" % xlab)
            
            if vv == 0:
                ax.set_ylabel("Frequency\n%s" % expnames_short[ex])
        
            viz.label_sp(ii,ax=ax,fig=fig)
            ii += 1

    figname = "%sMC_Test_Stochastic_Model_Stdev_%s.png" % (figpath,comparename)
    plt.savefig(figname,dpi=150,bbox_inches='tight')
else:
    fig,axs = plt.subplots(2,2,constrained_layout=True,figsize=(10,10))
    
    iilab = [0,2,1,3]
    ii    = 0
    for vv in range(2):
        
        if vv == 0: # Raw
            mcstds_in = mcstds
            stds_in    = stds
            xlm       = [0.25,0.60]
            xlab      = "$\sigma(SST)$"
            
        else:       # LP Filter
            mcstds_in = mcstds_lp
            stds_in   = stds_lp
            xlm       = [0,0.50]
            xlab      = "10-year LP Filtered $\sigma(SST)$"
            
        for ex in range(2):
            
            ax = axs[ex,vv]
            ax.hist(mcstds_in[ex],bins=bins,color=expcols[ex],edgecolor='w',density=True)
            
            bnds = np.quantile(mcstds_in[ex],[0.025,0.975])
            mu   = np.nanmean(mcstds_in[ex])
            
            ax.axvline(stds_in[-1],color="k",label="ERA5 = %.2f" % stds_in[-1])
            
            ax.axvline(stds_in[ex],color="blue",label="$\mu$ (Full Timeseries) = %.2f" % stds_in[ex])
            ax.axvline(mu,label="$\mu$ (Samples)= %.2f" % mu,ls='solid',color='gray')
            cflab = r"95% Bounds: [%.2f, %.2f]" % (bnds[0],bnds[1])
            ax.axvline(bnds[0],label=cflab,ls='dashed',color="gray")
            ax.axvline(bnds[1],label="",ls='dashed',color="gray")

            
            
            
            ax.set_xlim(xlm)
            ax.set_ylim([0,25])
            ax.legend()
            
            # csfit   = sp.stats.chi2.fit(mcstds[1])
            # pdftheo = sp.stats.chi2.pdf(bins,df=csfit[0])
            # ax.plot(bins,pdftheo)
            if ex == 1:
                ax.set_xlabel("%s [$\degree$ C]" % xlab)
            
            if vv == 0:
                ax.set_ylabel("Frequency\n%s" % expnames_long[ex])
        
            viz.label_sp(iilab[ii],ax=ax,fig=fig)
            ii += 1
    
    figname = "%sMC_Test_Stochastic_Model_Stdev_%s.png" % (figpath,comparename)
    plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Try getting confidence interval for ERA5

ntime_era       = len(ssts[-1])
n_eff           = proc.calc_dof(ssts[-1],) # calculate effective dof

# Get theoretical chi^2 PDF using n_eff-1

xvariances      = np.linspace(0,1,100)
era5var_pdf     = sp.stats.chi2.pdf(xvariances,n_eff-1,)#loc=stds[-1]**2)

plt.plot(xvariances,era5var_pdf)

nu      = n_eff -1 
alpha   = 0.05
upperv  = sp.stats.chi2.isf(1-alpha/2,nu)
lowerv  = sp.stats.chi2.isf(alpha/2,nu)

lower   = nu / lowerv
upper   = nu / upperv

lowerbnd_era5 = np.sqrt(lower)
upperbnd_era5 = np.sqrt(upper)

#%% Setup for barplot

errbar_var      = np.zeros((2,nexps))
errbar_var_lp   = np.zeros((2,nexps))

mcstds_arr      = np.array(mcstds) # [exp,sample]
mcstds_lp_arr   = np.array(mcstds_lp)

lowervar        = np.abs(np.quantile(mcstds,0.025,axis=1) - stds[:(nexps-1)]) # Can't be negative
uppervar        = np.quantile(mcstds,0.975,axis=1) - stds[:(nexps-1)]

errbar_var[0,:] = np.hstack([lowervar,[None,]])
errbar_var[1,:] = np.hstack([uppervar,[None,]])

lowervar_lp = np.abs(np.quantile(mcstds_lp,0.025,axis=1) - stds_lp[:(nexps-1)]) # Can't be negative
uppervar_lp = np.quantile(mcstds_lp,0.975,axis=1) - stds_lp[:(nexps-1)]

errbar_var_lp[0,:] = np.hstack([lowervar_lp,[None,]])
errbar_var_lp[1,:] = np.hstack([uppervar_lp,[None,]])

# ====================================================
#%% Plot ACF for Winter and Summer (For Paper Outline)
# ====================================================

fsz_title = 26
fsz_ticks = 14
plotkmons = [2,6]
use_neff  = True
conf      = 0.95
alpha     = 0.15 #0.15

if nexps == 3:
    fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(8,6.5))
else:
    fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(8,7))

for ii in range(2):
    ax     = axs[ii]
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
        
        if use_neff:
            dof_in = proc.calc_dof(ssts[ex][kmonth::12])
        else:
            dof_in = len(ssts[ex])/12
        
        print("%s for mon %i, DOF In = %.2f" % (expnames_long[ex],kmonth+1,dof_in))
        
        cflag = proc.calc_conflag(plotvar,conf,2,dof_in)
        if ex == (nexps-1):
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
        ax.set_xlabel("Lag [months]")
        if nexps == 3:
            ax.legend(framealpha=0,fontsize=fsz_ticks,ncol=2)
        else:
            ax.legend(framealpha=0,fontsize=12,ncol=1)
            
    
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


fig             = plt.figure(figsize=(14,4))
gs              = gridspec.GridSpec(4,12)

# --------------------------------- # Barplot
ax11            = fig.add_subplot(gs[:,:3],)
ax              = ax11

remove_topright = True
expcols_bar     = np.array(expcols).copy()
expcols_bar[-1] = 'gray'
label_vratio    = False
label_stds      = True

fsz_axis         = 18
fsz_ticks        = 16
fsz_legend       = 14
ytks_var         = np.arange(0,1.2,0.2)

instd            = stds
instd_lp         = stds_lp

if label_vratio:
    xlabs           = ["%s\n%.2f" % (expnames_long[ii],vratio[ii])+"%" for ii in range(len(vratio))]
else:
    xlabs  = expnames_short

braw            = ax.bar(np.arange(nexps),instd,color=expcols_bar,yerr=errbar_var,
                         error_kw=dict(ecolor='darkgray',
                                       barsabove=True,
                                       capsize=5,marker="o",markersize=25,mfc='None',
                                       ))
blp             = ax.bar(np.arange(nexps),instd_lp,color=dfcol,yerr=errbar_var_lp,
                         error_kw=dict(ecolor='w',
                                       barsabove=True,
                                       capsize=5,marker="d",markersize=25,mfc='None',))

if darkmode:
    upperlbl_col = 'lightgray'
else:
    upperlbl_col = 'gray'

if label_stds:
    ax.bar_label(braw,fmt="%.02f",c=upperlbl_col,fontsize=fsz_axis)
    ax.bar_label(blp,fmt="%.02f",c='w',fontsize=fsz_axis,label_type='center')

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
ax.set_yticks(ytks_var)
ax.tick_params(labelsize=fsz_tick)

if remove_topright:
    ax.spines[['right', 'top']].set_visible(False)

viz.label_sp(0,alpha=0.15,ax=ax,fontsize=fsz_title,y=1.10,x=-.45,
             fontcolor=dfcol)



# --------------------------------- # Power Spectra
ax22       = fig.add_subplot(gs[:,4:])

ax         = ax22

decadal_focus = True
obs_cutoff = 10 # in years
obs_cutoff = 1/(obs_cutoff*12)

dtmon_fix       = 60*60*24*30

if decadal_focus:
    xper            = np.array([20,10,5,2,1,0.5])
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
    
    # Plot Cut Off Section for Obs
    if ii == (nexps-1):
        iplot_hifreq = np.where(plotfreq > obs_cutoff)[0]
        ax.loglog(plotfreq,plotspec,label="",c=col_in,ls='dashed',lw=1.5)
        hiplotfreq     = plotfreq[iplot_hifreq]
        hiplotspec     = plotspec[iplot_hifreq]
        ax.loglog(hiplotfreq,hiplotspec,lw=2.5,label=expnames_long[ii],c=col_in,zorder=2)
        
    else:
        ax.loglog(plotfreq,plotspec,lw=2.5,label=expnames_long[ii],c=col_in,zorder=2)
        
    # Plot the 95% Confidence Interval (for stochastic model output)
    if ii < (nexps-1):
        plotspec1 = mc_specdicts[ii]['specs'] / dtmon_fix
        plotfreq1 = mc_specdicts[ii]['freqs'][0,:] * dtmon_fix
        
        bnds = np.quantile( plotspec1 ,[0.025,0.975],axis=0)
        
        ax.fill_between(plotfreq1,bnds[0],bnds[1],color=expcols[ii],alpha=0.15,zorder=1)
        #ax.loglog(plotfreq,bnds[0],ls='dotted',color='blue',label="95% Conf.")
        #ax.loglog(plotfreq,bnds[1],ls='dotted',color='blue')
    else:
        
        # Plot Confidence Interval (ERA5)
        alpha           = 0.05
        cloc_era        = [2e-2,6]
        dof_era         = metrics_out['dofs'][-1]
        cbnds_era       = proc.calc_confspec(alpha,dof_era)
        ax.fill_between(plotfreq,cbnds_era[0]*plotspec,cbnds_era[1]*plotspec,color=expcols[ii],alpha=0.05,zorder=1)
        #proc.plot_conflog(cloc_era,cbnds_era,ax=ax,color=dfcol,cflabel=r"95% Confidence (ERA5)") #+r" (dof= %.2f)" % dof_era)
    
    #ax.loglog(plotfreq,CCs[:,0],ls='dotted',lw=0.5,c=expcols[ii])
    #ax.loglog(plotfreq,CCs[:,1],ls='dashed',lw=0.9,c=expcols[ii])

ax.set_xlim([xper_ticks[0],0.5])
ax.axvline([1/(6)],label="",ls='dotted',c='gray')
ax.axvline([1/(12)],label="",ls='dotted',c='gray')
ax.axvline([1/(5*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(2*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(10*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(20*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(40*12)],label="",ls='dotted',c='gray')

ax.set_xlabel("Frequency (1/month)",fontsize=fsz_axis)
ax.set_ylabel("Power [$\degree C ^2 / cycle \, per \, mon$]",fontsize=fsz_axis)

ax2 = ax.twiny()
ax2.set_xlim([xper_ticks[0],0.5])
ax2.set_xscale('log')
ax2.set_xticks(xper_ticks,labels=xper)
ax2.set_xlabel("Period (Years)",fontsize=fsz_axis)

ax.legend(fontsize=fsz_legend,framealpha=0.5,edgecolor='none')

for ax in [ax,ax2]:
    ax.tick_params(labelsize=fsz_ticks)
    
viz.label_sp(1,alpha=0.15,ax=ax,fontsize=fsz_title,y=1.10,x=-.1,
             fontcolor=dfcol)

figname = "%sVariance_%s_PaperOutline.png" % (figpath,comparename)
if darkmode:
    figname = proc.darkname(figname)
plt.savefig(figname,dpi=150,transparent=transparent,bbox_inches='tight')

# ===============
#%% Draft 3 Version, with Monthly Variance Added
# ===============

fig             = plt.figure(figsize=(14,10))
gs              = gridspec.GridSpec(8,12)

ax11            = fig.add_subplot(gs[:3,:3],) # Barplot
ax22            = fig.add_subplot(gs[:3,4:11])  # Month Std
ax33            = fig.add_subplot(gs[4:,:11]) # Spectra

# --------------------------------- # Barplot

ax               = ax11
remove_topright  = True
expcols_bar      = np.array(expcols).copy()
expcols_bar[-1]  = 'gray'
label_vratio     = False
label_stds       = True

fsz_axis         = 18
fsz_ticks        = 16
fsz_legend       = 14
fsz_legend_spectra = 16
ytks_var         = np.arange(0,1.2,0.2)

instd            = stds
instd_lp         = stds_lp

if label_vratio:
    xlabs           = ["%s\n%.2f" % (expnames_long[ii],vratio[ii])+"%" for ii in range(len(vratio))]
else:
    xlabs  = expnames_short

braw            = ax.bar(np.arange(nexps),instd,color=expcols_bar,yerr=errbar_var,
                         error_kw=dict(ecolor='darkgray',
                                       barsabove=True,
                                       capsize=5,marker="o",markersize=25,mfc='None',
                                       ))
blp             = ax.bar(np.arange(nexps),instd_lp,color=dfcol,yerr=errbar_var_lp,
                         error_kw=dict(ecolor='w',
                                       barsabove=True,
                                       capsize=5,marker="d",markersize=25,mfc='None',))

if darkmode:
    upperlbl_col = 'lightgray'
else:
    upperlbl_col = 'gray'

if label_stds:
    ax.bar_label(braw,fmt="%.02f",c=upperlbl_col,fontsize=fsz_axis)
    ax.bar_label(blp,fmt="%.02f",c='w',fontsize=fsz_axis,label_type='center')

# --- Make Fake Legend'
colorsf = {'Raw':'gray','10-year Low-pass':'k',}         
labels = list(colorsf.keys())
handles = [plt.Rectangle((0,0),1,1, color=colorsf[label]) for label in labels]
ax.legend(handles, labels,fontsize=fsz_legend,framealpha=0,
          bbox_to_anchor=(0.04, 0.82, 1., .102))

#ax.legend()
#ax.bar_label(vratio,fmt="%.2f",c=w,label_type='bottom')

ax.set_xticks(np.arange(nexps),labels=xlabs,fontsize=fsz_tick,rotation=45)
ax.set_ylabel("$\sigma$(SST) [$\degree$C]",fontsize=fsz_axis)
ax.set_ylim([0,1.0])
ax.set_yticks(ytks_var)
ax.tick_params(labelsize=fsz_tick)

if remove_topright:
    ax.spines[['right', 'top']].set_visible(False)

viz.label_sp(0,alpha=0.15,ax=ax,fontsize=fsz_title,y=1.10,x=-.45,
             fontcolor=dfcol)




# --------------------------------- # Monthly Variance

ax              = ax22

viz.label_sp(1,alpha=0.15,ax=ax,fontsize=fsz_title,y=1.10,x=-.15,
             fontcolor=dfcol)


for ex in range(nexps):
    plotvar = monstds[ex]
    ax.plot(mons3,plotvar,label=expnames_long[ex],
            color=expcols[ex],lw=2.5,ls=expls[ex],marker="o",zorder=1)
    
    # Plot Confidence INterval
    if ex < (nexps-1):
        plotmc = monstds_sample[ex]
        bnds   = np.quantile(plotmc,[0.025,0.95],axis=0)
        ax.fill_between(mons3,bnds[0],bnds[1],color=expcols[ex],alpha=0.10,zorder=5)


ax.set_ylabel("Monthly $\sigma(SST)$ [$\degree$C]",fontsize=fsz_axis)
#ax.set_xticks(np.arange(1,13))
ax.set_xticklabels(mons3)
ax.set_xlim([0,11])
ax.set_ylim([0,0.75])
ax.set_yticks(np.arange(0.2,0.81,0.2))
ax.tick_params(labelsize=fsz_ticks)

if remove_topright:
    ax.spines[['right', 'top']].set_visible(False)

# --------------------------------- # Power Spectra

ax = ax33



decadal_focus = True
obs_cutoff = 10 # in years
obs_cutoff = 1/(obs_cutoff*12)

dtmon_fix       = 60*60*24*30

if decadal_focus:
    xper            = np.array([20,10,5,2,1,0.5])
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
    
    # Plot Cut Off Section for Obs
    if ii == (nexps-1):
        iplot_hifreq = np.where(plotfreq > obs_cutoff)[0]
        ax.loglog(plotfreq,plotspec,label="",c=col_in,ls='dashed',lw=1.5)
        hiplotfreq     = plotfreq[iplot_hifreq]
        hiplotspec     = plotspec[iplot_hifreq]
        ax.loglog(hiplotfreq,hiplotspec,lw=2.5,label=expnames_long[ii],c=col_in,zorder=2)
        
    else:
        ax.loglog(plotfreq,plotspec,lw=2.5,label=expnames_long[ii],c=col_in,zorder=2)
        
    # Plot the 95% Confidence Interval (for stochastic model output)
    if ii < (nexps-1):
        plotspec1 = mc_specdicts[ii]['specs'] / dtmon_fix
        plotfreq1 = mc_specdicts[ii]['freqs'][0,:] * dtmon_fix
        
        bnds = np.quantile( plotspec1 ,[0.025,0.975],axis=0)
        
        ax.fill_between(plotfreq1,bnds[0],bnds[1],color=expcols[ii],alpha=0.15,zorder=1)
        #ax.loglog(plotfreq,bnds[0],ls='dotted',color='blue',label="95% Conf.")
        #ax.loglog(plotfreq,bnds[1],ls='dotted',color='blue')
    else:
        
        # Plot Confidence Interval (ERA5)
        alpha           = 0.05
        cloc_era        = [2e-2,6]
        dof_era         = metrics_out['dofs'][-1]
        cbnds_era       = proc.calc_confspec(alpha,dof_era)
        ax.fill_between(plotfreq,cbnds_era[0]*plotspec,cbnds_era[1]*plotspec,color=expcols[ii],alpha=0.05,zorder=1)
        #proc.plot_conflog(cloc_era,cbnds_era,ax=ax,color=dfcol,cflabel=r"95% Confidence (ERA5)") #+r" (dof= %.2f)" % dof_era)
    
    #ax.loglog(plotfreq,CCs[:,0],ls='dotted',lw=0.5,c=expcols[ii])
    #ax.loglog(plotfreq,CCs[:,1],ls='dashed',lw=0.9,c=expcols[ii])

ax.set_xlim([xper_ticks[0],0.5])
ax.axvline([1/(6)],label="",ls='dotted',c='gray')
ax.axvline([1/(12)],label="",ls='dotted',c='gray')
ax.axvline([1/(5*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(2*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(10*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(20*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(40*12)],label="",ls='dotted',c='gray')

ax.set_xlabel("Frequency (1/month)",fontsize=fsz_axis)
ax.set_ylabel("Power [$\degree C ^2 / cycle \, per \, mon$]",fontsize=fsz_axis)

ax2 = ax.twiny()
ax2.set_xlim([xper_ticks[0],0.5])
ax2.set_xscale('log')
ax2.set_xticks(xper_ticks,labels=xper)
ax2.set_xlabel("Period (Years)",fontsize=fsz_axis)

ax.legend(fontsize=fsz_legend_spectra,framealpha=0.5,edgecolor='none')

for ax in [ax,ax2]:
    ax.tick_params(labelsize=fsz_ticks)
    
viz.label_sp(2,alpha=0.15,ax=ax,fontsize=fsz_title,y=1.10,x=-.1,
             fontcolor=dfcol)

figname = "%sVariance_%s_Draft3Ver.png" % (figpath,comparename)
if darkmode:
    figname = proc.darkname(figname)
plt.savefig(figname,dpi=150,transparent=transparent,bbox_inches='tight')



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare Estimated Heat Flux Feedback from CESM2 Hierarchy

- Copied upper section of calc_Fprime_CESM2_hierarchy (2025.07.14)


Created on Mon Jul 14 11:26:54 2025

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

#%% Do the calculations

# Calculation Options
expnames        = ["SOM","MCOM","FOM"]
sstnames        = ["TS","TS","TS"]
flxnames        = ["SHF","SHF","SHF"]
expnames_long   = ["Slab","Multi-Column","Full"]
expyears        = ["60-360","100-500","200-1800"]
dpath           = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/hff/"

tstarts         = ['0060-01-01','0100-01-01','0200-01-01']
tends           = ['0360-12-31','0500-12-31','2000-12-31']

figpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20250716/"
proc.makedir(figpath)


#%% Load Estimated Damping

nexps  = len(expnames)
hffall = []
for ex in range(nexps):
    
    ncsearch = "%sCESM2_%s_%s_damping*.nc" % (dpath,expnames[ex],flxnames[ex])
    nc       = glob.glob(ncsearch)[0]
    ds_hff   = xr.open_dataset(nc).load()
    hffall.append(ds_hff)
    
    print(nc)


#%% Load CESM2 PiControl Ice Mask
maskpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/masks/"
masknc   = "cesm2_pic_limask_0.3p_0.05p_0200to2000.nc"
dsmask   = xr.open_dataset(maskpath+masknc).load()
dsmask180 = proc.lon360to180_xr(dsmask)
# bbreg     = proc.get_bbox(ds_sst)
# dsmask180 = proc.sel_region_xr(dsmask180,bbreg).mask


#%% Compare Damping Estimates over the northeastern SPG

bboxSPGNE   = [-40,-15,52,62]
bbsel       = [-50,-10,45,65]
fsz_tick    = 24
fsz_axis    = 24
fsz_title   = 28
proj        = ccrs.PlateCarree()
mons3       = proc.get_monstr()

ilag        = 0
imon        = 1

cints       = np.arange(-39,42,3)

for imon in range(12):
    fig,axs,_ = viz.init_orthomap(1,3,bboxplot=bbsel,centlon=-30,figsize=(24,12.5))
    
    
    for ex in range(3):
        
        ax              = axs[ex]
        ax              = viz.add_coast_grid(ax,bbox=bbsel,fill_color='lightgray',fontsize=fsz_tick)
        
        vname_damping   = "%s_damping" % flxnames[ex]
        plotvar         = hffall[ex][vname_damping].isel(lag=ilag,month=imon) * -1
        
        pcm             = ax.contourf(plotvar.lon,plotvar.lat,plotvar,cmap='cmo.balance',transform=proj,zorder=-1,levels=cints)
        cl             = ax.contour(plotvar.lon,plotvar.lat,plotvar,colors="k",linewidths=0.55,transform=proj,zorder=-1,levels=cints)
        
        
        clbl=ax.clabel(cl,fontsize=fsz_tick)
        viz.add_fontborder(clbl)
        
        ax.set_title( "%s (Years %s)" % (expnames_long[ex],expyears[ex]),fontsize=fsz_title)
        
        viz.plot_box(bboxSPGNE,ax=ax,color='purple',proj=proj,linewidth=2.5)
        
        
        # plotvar   = logratio
        # #pcm       = ax.contourf(lon,lat,plotvar,cmap='cmo.balance',transform=proj,zorder=-1,levels=cints)
        # pcm       = ax.pcolormesh(lon,lat,plotvar,cmap='cmo.balance',transform=proj,zorder=-1,vmin=-.5,vmax=.5)
        
        # Plot the Sea Ice
        plotvar = xr.where(np.isnan(dsmask180).mask,0,1)
        cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                                colors="cyan",levels=[0,1,],
                                transform=proj,linewidths=2.5)
        
    cb        = viz.hcbar(pcm,ax=axs.flatten(),fontsize=fsz_tick,fraction=0.04)
    cb.set_label(r"%s Lag %i Heat Flux Damping [$\frac{W}{m^2 \degree C}$]" % (mons3[imon],ilag+1),fontsize=fsz_axis)
    
    figname = "%sCESM2_Hierarchy_%s_lag%02i_mon%02i.png" % (figpath,vname_damping,ilag+1,imon+1)
    plt.savefig(figname,dpi=150,bbox_inches='tight')
    
    

#%% Compare Damping Estimates over the northeastern SPG

ilag        = 0
imon        = 1
vname_plot  = "cov"#"sst_flx_crosscorr" # "sst_autocorr"#


if vname_plot == "SHF_damping":
    cints       = np.arange(-39,42,3)
elif vname_plot == "cov":
    cints       = np.arange(-5,5.5,0.5)
    


for imon in range(12):
    fig,axs,_ = viz.init_orthomap(1,3,bboxplot=bbsel,centlon=-30,figsize=(24,12.5))
    
    
    for ex in range(3):
        
        ax              = axs[ex]
        ax              = viz.add_coast_grid(ax,bbox=bbsel,fill_color='lightgray',fontsize=fsz_tick)
        
        vname_damping   = "%s_damping" % flxnames[ex]
        plotvar         = hffall[ex][vname_plot].isel(lag=ilag,month=imon) #* -1
        
        if vname_plot == "sst_autocorr":
            cints       = np.arange(0.80,1.02,0.02)
            cmap        = 'cmo.amp'
        else:
            cints       = np.arange(-.5,0.55,0.05)
            cmap        = 'cmo.balance'
        
        pcm             = ax.contourf(plotvar.lon,plotvar.lat,plotvar,cmap=cmap,transform=proj,zorder=-1,levels=cints)
        cl             = ax.contour(plotvar.lon,plotvar.lat,plotvar,colors="k",linewidths=0.55,transform=proj,zorder=-1,levels=cints)
        
        clbl=ax.clabel(cl,fontsize=fsz_tick)
        viz.add_fontborder(clbl)
        
        ax.set_title( "%s (Years %s)" % (expnames_long[ex],expyears[ex]),fontsize=fsz_title)
        
        viz.plot_box(bboxSPGNE,ax=ax,color='purple',proj=proj,linewidth=2.5)
        
        
        # plotvar   = logratio
        # #pcm       = ax.contourf(lon,lat,plotvar,cmap='cmo.balance',transform=proj,zorder=-1,levels=cints)
        # pcm       = ax.pcolormesh(lon,lat,plotvar,cmap='cmo.balance',transform=proj,zorder=-1,vmin=-.5,vmax=.5)
        
        # Plot the Sea Ice
        plotvar = xr.where(np.isnan(dsmask180).mask,0,1)
        cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                                colors="cyan",levels=[0,1,],
                                transform=proj,linewidths=2.5)
        
    cb        = viz.hcbar(pcm,ax=axs.flatten(),fontsize=fsz_tick,fraction=0.04)
    cb.set_label(r"%s Lag %i %s" % (mons3[imon],ilag+1,vname_plot),fontsize=fsz_axis)
    
    figname = "%sCESM2_Hierarchy_%s_lag%02i_mon%02i.png" % (figpath,vname_plot,ilag+1,imon+1)
    plt.savefig(figname,dpi=150,bbox_inches='tight')
    
#%% Load and also plot the HFF for observations

ncobs       = "ERA5_qnet_hfdamping_NAtl_1979to2024_ensorem1_detrend1.nc"
#dpath_era   = ""
ds_era      = xr.open_dataset(dpath + ncobs).load()

#%% Plot era5 values

vname_damping =  "%s_damping" % flxnames[0]

imon       = 0
ilag       = 0
for imon in range(12):
    fig,axs,_ = viz.init_orthomap(1,4,bboxplot=bbsel,centlon=-30,figsize=(24,12.5))
    cints       = np.arange(-39,42,3)
    cints_line = np.arange(-52,55,3)
    
    for ex in range(4):
        
        ax              = axs[ex]
        ax              = viz.add_coast_grid(ax,bbox=bbsel,fill_color='lightgray',fontsize=fsz_tick)
        
        
        if ex == 3:
            plotvar         = ds_era.qnet_damping.isel(lag=ilag,month=imon) * -1
            
        else:
            plotvar         = hffall[ex][vname_damping].isel(lag=ilag,month=imon) * -1
            vname_damping   = "%s_damping" % flxnames[ex]
        
        pcm             = ax.contourf(plotvar.lon,plotvar.lat,plotvar,cmap='cmo.balance',transform=proj,zorder=-1,levels=cints,)
        cl              = ax.contour(plotvar.lon,plotvar.lat,plotvar,colors="k",linewidths=0.55,transform=proj,zorder=-1,levels=cints_line)
        
        if ex == 3:
            clbl=ax.clabel(cl,fontsize=10,levels=cints_line[::2])
        else:
            clbl=ax.clabel(cl,fontsize=fsz_tick)
        viz.add_fontborder(clbl)
        
        if ex < 3:
            ax.set_title( "%s (Years %s)" % (expnames_long[ex],expyears[ex]),fontsize=fsz_title)
        else:
            ax.set_title("ERA5 (1979-2024)")
        
        viz.plot_box(bboxSPGNE,ax=ax,color='purple',proj=proj,linewidth=2.5)
        
        # plotvar   = logratio
        # #pcm       = ax.contourf(lon,lat,plotvar,cmap='cmo.balance',transform=proj,zorder=-1,levels=cints)
        # pcm       = ax.pcolormesh(lon,lat,plotvar,cmap='cmo.balance',transform=proj,zorder=-1,vmin=-.5,vmax=.5)
        
        # Plot the Sea Ice
        if ex < 3:
            plotvar = xr.where(np.isnan(dsmask180).mask,0,1)
            cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                                    colors="cyan",levels=[0,1,],
                                    transform=proj,linewidths=2.5)
    
    # Plot ERA5 
    cb        = viz.hcbar(pcm,ax=axs.flatten(),fontsize=fsz_tick,fraction=0.04)
    cb.set_label(r"%s Lag %i Heat Flux Damping [$\frac{W}{m^2 \degree C}$]" % (mons3[imon],ilag+1),fontsize=fsz_axis)
    figname = "%sCESM2_Hierarchy_vERA5_%s_lag%02i_mon%02i.png" % (figpath,vname_damping,ilag+1,imon+1)
    plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Plot Area Average Damping

damping_reg  = [proc.sel_region_xr(ds["SHF_damping"],bboxSPGNE) for ds in hffall]
damping_aavg = [proc.area_avg_cosweight(ds) for ds in damping_reg]

era5_reg     = proc.sel_region_xr(ds_era.qnet_damping,bboxSPGNE)
era5_aavg    = proc.area_avg_cosweight(era5_reg)

#%% Plot area-average estimate values

expcols     = ['violet','forestgreen','cornflowerblue','k']
ilag = 0
fig,ax = viz.init_monplot(1,1,figsize=(8,4))

for ex in range(3):
    plotvar = damping_aavg[ex].isel(lag=ilag) * -1
    ax.plot(mons3,plotvar,label=expnames[ex],c=expcols[ex],lw=2.5,marker="o")
    
ax.plot(mons3,era5_aavg.isel(lag=ilag)*-1,color="k",label="ERA5",lw=2.5,marker="o")
ax.legend()

ax.set_title("SPGNE-Averaged Lag %i HFF Estimates" % (ilag+1))
ax.set_ylabel("Heat Flux Damping (W/m2/degC)")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

visualize t2 and rei in era5
 - Copied upper section of viz_stochmod_output_obs_scrap
 - Used output from reemergence/analysis/calc_remidx_general
 - 

Created on Thu Apr 17 11:48:22 2025

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
figpath         = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/05_SMIO/02_Figures/20250521/"
sm_output_path  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/"
procpath        = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"

proc.makedir(figpath)

# Indicate Experiment and Comparison Name 
comparename     = "kgm20250417"
expnames        = ["SST_Obs_Pilot_00_Tdcorr0_qnet","SST_ERA5_1979_2024"]
expnames_long   = ["Stochastic Model (with Re-emergence)","ERA5 Reanalysis (1979-2024)"]
expcols         = ["turquoise","k"]
expls           = ["dotted",'solid']


#%% Load REI and T2

t2s  = []
reis = []
nexps   = len(expnames)
for ex in range(nexps):
    expname = expnames[ex]
    
    # Load REI
    ncrei   = "%s%s/Metrics/REI_Pointwise.nc" % (sm_output_path,expname)
    ds_rei  = xr.open_dataset(ncrei).load()
    
    # Load T2
    nct2    = "%s%s/Metrics/T2_Timescale.nc" % (sm_output_path,expname)
    ds_t2   = xr.open_dataset(nct2).load()
    
    
    if 'ens' in ds_rei.dims:
        ds_rei = ds_rei.mean('ens')
        ds_t2  = ds_t2.mean('ens')
    
    reis.append(ds_rei)
    t2s.append(ds_t2)
    




#%% Compare the REI
iyr     = 0
kmonth  = 1
#apply_mask =  ds_masks.mask_mon

bbsel           = [-40,-12,50,62]

fsz_axis        = 24
fsz_ticks       = 18
fsz_title       = 30
label_ssh       = False
cints           = np.arange(0,1.05,0.05)
pmesh           = True
cmap            = 'cmo.deep_r'

#for iyr in range(5):
fig,axs,_       = viz.init_orthomap(1,2,bboxplot,figsize=(20,10))


for ii in range(2):
    ax      = axs[ii]
    ax      = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)
    
    # Plot the REI
    plotvar = reis[ii].isel(mon=kmonth,yr=iyr).rei #* apply_mask
    if pmesh:
        cf = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                              vmin=cints[0],vmax=cints[-1],
                              transform=proj,cmap=cmap)
    else:
        cf      = ax.contourf(plotvar.lon,plotvar.lat,plotvar,
                              levels=cints,
                              transform=proj,cmap=cmap)
    
    
    # Plot the sea ice edge
    plotvar = ds_masks.mask_mon
    cl = ax.contour(plotvar.lon, plotvar.lat,
                    plotvar, colors="cyan",
                    linewidths=2, transform=proj, levels=[0, 1], zorder=1)
    
    
    # Plot the SSH (Time Mean) --------------
    #if ii == 1:
    plotvar = ds_adt.mean('time')
    cl = ax.contour(plotvar.lon, plotvar.lat, plotvar.adt*100, colors="w",alpha=0.8,
                    linewidths=0.55, transform=proj, levels=cints_adt)
    if label_ssh:
        ax.clabel(cl,fontsize=fsz_ticks-2)
            
            
    ax.set_title(expnames_long[ii],fontsize=fsz_title)
    viz.plot_box(bbsel,ax=ax,proj=proj,color='yellow',linewidth=4,linestyle='solid')




cb = viz.hcbar(cf,ax=axs.flatten(),fontsize=fsz_ticks)
cb.set_label("Re-emergence Index",fontsize=fsz_axis)

figname = "%sREI_Plot_%s_month%02i_yr%0i.png" % (figpath,comparename,kmonth+1,iyr+1)
plt.savefig(figname,dpi=150,bbox_inches='tight')


#%% Compare the T2
#iyr     = 0
kmonth  = 1
#apply_mask =  ds_masks.mask_mon



fsz_axis        = 24
fsz_ticks       = 18
fsz_title       = 30
label_ssh       = False
cints           = np.arange(0,105,5) #np.arange(0,30,2)
cints_over      = np.arange(30,110,20)
pmesh           = True
cmap            = 'gist_ncar'#'cmo.tempo_r'

#for iyr in range(5):
for kmonth in range(12):
    fig,axs,_       = viz.init_orthomap(1,2,bboxplot,figsize=(20,10))
    
    
    for ii in range(2):
        ax      = axs[ii]
        ax      = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)
        
        # Plot the t2
        plotvar = t2s[ii].isel(mon=kmonth).T2.T #* apply_mask
        if pmesh:
            cf = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                                  vmin=cints[0],vmax=cints[-1],
                                  transform=proj,cmap=cmap,zorder=-1)
        else:
            cf      = ax.contourf(plotvar.lon,plotvar.lat,plotvar,
                                  levels=cints,
                                  transform=proj,cmap=cmap,zorder=-1)
        
        clover = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                              levels=cints_over,
                              transform=proj,colors="k",linewidths=0.75)
        clbl = ax.clabel(clover,fontsize=fsz_ticks-6)
        viz.add_fontborder(clbl)
        # clover = ax.contourf(plotvar.lon,plotvar.lat,plotvar,
        #                       levels=cints_over,
        #                       transform=proj,cmap="cmo.deep")
        # pcm2 = ax.contourf(plotvar.lon,plotvar.lat,plotvar,
        #                      vmin=cints_over[0],vmax=cints_over[-1],
        #                       #levels=cints,
        #                       transform=proj,cmap=cmap)
        
        
        # Plot the sea ice edge
        plotvar = ds_masks.mask_mon
        cl = ax.contour(plotvar.lon, plotvar.lat,
                        plotvar, colors="cyan",
                        linewidths=2, transform=proj, levels=[0, 1], zorder=1)
        
        
        # Plot the SSH (Time Mean) --------------
        #if ii == 1:
        plotvar = ds_adt.mean('time')
        cl = ax.contour(plotvar.lon, plotvar.lat, plotvar.adt*100, colors="navy",alpha=1,
                        linewidths=0.55, transform=proj, levels=cints_adt)
        if label_ssh:
            ax.clabel(cl,fontsize=fsz_ticks-2)
                
                
        ax.set_title(expnames_long[ii],fontsize=fsz_title)
    
    
        viz.plot_box(bbsel,ax=ax,proj=proj,color='yellow',linewidth=4,linestyle='solid')
    
    
    
    cb = viz.hcbar(cf,ax=axs.flatten(),fontsize=fsz_ticks)
    cb.set_label("%s Decorrelation Timescale $T^2$ [Months]" % (mons3[kmonth]),fontsize=fsz_axis)
    
    figname = "%sT2_Plot_%s_month%02i.png" % (figpath,comparename,kmonth+1)
    plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Zoom in on Gulf Stream Region

cints           = np.arange(6,13,1) #np.arange(0,30,2)
cmap            = 'cmo.tempo_r'
ii              = 1
pmesh           = True

fig,ax,bbox_gs = viz.init_regplot(regname="SAR",fontsize=36,bboxin=None)

# Plot the t2
plotvar = t2s[ii].isel(mon=kmonth).T2.T #* apply_mask
if pmesh:
    cf = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                          vmin=cints[0],vmax=cints[-1],
                          transform=proj,cmap=cmap,zorder=-1)
else:
    cf      = ax.contourf(plotvar.lon,plotvar.lat,plotvar,
                          levels=cints,
                          transform=proj,cmap=cmap,zorder=-1)

# clover = ax.contour(plotvar.lon,plotvar.lat,plotvar,
#                       levels=cints_over,
#                       transform=proj,colors="k",linewidths=0.75)
# clbl = ax.clabel(clover,fontsize=fsz_ticks-6)
# viz.add_fontborder(clbl)

# Plot the SSH (Time Mean) --------------
#if ii == 1:
plotvar = ds_adt.mean('time')
cl = ax.contour(plotvar.lon, plotvar.lat, plotvar.adt*100, colors="skyblue",alpha=1,
                linewidths=.55, transform=proj, levels=cints_adt) 
    
cb = viz.hcbar(cf,ax=ax,fontsize=22)
cb.set_label("Decorrelation Timescale $T^2$ [Months]",fontsize=fsz_axis)  

#%% Zoom in on SPG Region

bbsel = [-40, -12, 52, 62] # [-40, -12, 50, 62]

# Plot Settings
cints           = np.arange(0,95,5) #np.arange(0,30,2)
cmap            = 'gist_ncar'#'#cmo.tempo_r'
ii              = 1
pmesh           = True
kmonths         = np.arange(12)

#fig,ax,bbox_gs = viz.init_regplot(regname="IRM",fontsize=36,bboxin=None)
bboxin    = [-80,0,40,65]
figsize   = (25,14)
centlon   = -40
fontsize  = 32
fig,ax,_  = viz.init_orthomap(1,1,bboxin,figsize=figsize,centlon=centlon)
ax        = viz.add_coast_grid(ax,bbox=bboxin,fill_color='lightgray',fontsize=fontsize)



# Plot the t2
plotvar = t2s[ii].isel(mon=kmonths).mean('mon').T2.T #* apply_mask
if pmesh:
    cf = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                          vmin=cints[0],vmax=cints[-1],
                          transform=proj,cmap=cmap,zorder=-1)
else:
    cf      = ax.contourf(plotvar.lon,plotvar.lat,plotvar,
                          levels=cints,
                          transform=proj,cmap=cmap,zorder=-1)


# Plot the SSH (Time Mean) --------------
#if ii == 1:
plotvar = ds_adt.mean('time')
cl = ax.contour(plotvar.lon, plotvar.lat, plotvar.adt*100, colors="k",alpha=1,
                linewidths=2, transform=proj, levels=cints_adt) 
    
cb = viz.hcbar(cf,ax=ax,fontsize=22)
cb.set_label("Decorrelation Timescale $T^2$ [Months]",fontsize=fsz_axis)  

# Plot the Box
viz.plot_box(bbsel,ax=ax,proj=proj,color='yellow',linewidth=4,linestyle='solid')


locfn,loctitle = proc.make_locstring_bbox(bbsel)
ax.set_title(loctitle,fontsize=42)

#%% Inset Map with ERA5 Only (Paper Outline)

bbsel = [-40, -15, 52, 62] # [-40, -12, 50, 62]

centlat         = 45
#bboxplot_new    = [-80,0,] 
#apply_mask      =  ds_masks.mask_mon
# Plot Setting
cints           = np.arange(0,29,1) #np.arange(0,30,2)
cmap            = 'cmo.tempo_r'
ii              = 1
pmesh           = True
kmonths         = [1,2]# np.arange(12)

fig,ax,_       = viz.init_orthomap(1,1,bboxplot,figsize=(20,10),centlat=centlat)
ax             = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)

# Plot the t2
plotvar = t2s[ii].isel(mon=kmonths).mean('mon').T2.T #* apply_mask
if pmesh:
    cf = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                          vmin=cints[0],vmax=cints[-1],
                          transform=proj,cmap=cmap,zorder=-1)
else:
    cf      = ax.contourf(plotvar.lon,plotvar.lat,plotvar,
                          levels=cints,
                          transform=proj,cmap=cmap,zorder=-1)

# Plot the SSH (Time Mean) --------------
#if ii == 1:
plotvar = ds_adt.mean('time')
cl = ax.contour(plotvar.lon, plotvar.lat, plotvar.adt*100, colors="k",alpha=1,
                linewidths=.75, transform=proj, levels=cints_adt) 
    
cb = viz.hcbar(cf,ax=ax,fontsize=22)
cb.set_label("Wintertime Decorrelation Timescale $T_2$ [Months]",fontsize=fsz_axis)  

# Plot the Box
viz.plot_box(bbsel,ax=ax,proj=proj,color='indigo',linewidth=4,linestyle='dashed')



# Plot the sea ice edge
plotvar = ds_masks.mask_mon
cl = ax.contour(plotvar.lon, plotvar.lat,
                plotvar, colors="cyan",
                linewidths=2, transform=proj, levels=[0, 1], zorder=1)


# # Add an Inset! -----------
# centlon = -40
# centlat = 35
# oproj =  ccrs.Orthographic(central_longitude=centlon, central_latitude=35)
# import cartopy
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# axins = inset_axes(ax, width="60%", height="60%", loc='lower left',
#                    bbox_to_anchor=(-0.05, 0.2, 0.5, 0.5),
#                    bbox_transform=ax.transAxes,
#                    axes_class=cartopy.mpl.geoaxes.GeoAxes,
#                    axes_kwargs=dict(map_projection=oproj))


# axins = viz.add_coast_grid(axins,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)


# # Draw lines between inset map and box on main map
# rect, connectors = ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.5, transform=ax.transAxes)
# # By default only two of the connecting lines (connectors) are shown
# # it is possible to choose which of the lines to show by setting the visibility
# # connectors are counted clockwise from the lower-left corner
# connectors[0].set_visible(False)
# connectors[1].set_visible(True)
# connectors[3].set_visible(True)
# connectors[2].set_visible(True)

#locfn,loctitle = proc.make_locstring_bbox(bbsel)
#ax.set_title(loctitle,fontsize=42)

figname = "%sWinterimte_T2_Locator.png" % (figpath)
plt.savefig(figname,dpi=150,bbox_inches='tight')



#%% Scrap Below


#%% Load the ACFs for each experiment (old style, different format)

# # Indicate Observations
# append_obs      = True
# dpath_obs       = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/reanalysis/proc/NATL_proc_obs/"
# ncname_obs      = "ERA5_sst_NAtl_1979to2024.nc" 
# ncname_acf_obs  = "ERA5_NAtl_1979to2024_lag00to60_ALL_ensALL.nc"
# obsname         = "ERA5"
# obsname_long    = "ERA5 Reanalysis (1979-2024)"


# comparename     = "paperoutline"
# expnames        = ["SST_Obs_Pilot_00_Tdcorr0_qnet","SST_Obs_Pilot_00_Tdcorr1_qnet",obsname]
# expnames_long   = ["with re-emergence","no re-emergence",obsname_long]
# expcols         = ["turquoise","goldenrod","k"]
# expls           = ["dotted","dashed",'solid']


# # Load Output for the stochastic model
# ds_all  = []
# nexps   = len(expnames)
# for ex in tqdm.tqdm(range(nexps-1)):
    
#     # Load ACFs
#     expname = expnames[ex]
#     ncname  = "%sSM_%s_lag00to60_ALL_ensALL.nc" % (procpath,expname)
#     ds      = xr.open_dataset(ncname).load()#.mean('ens')
    
#     # Take the ensemble average (can change this later)
#     ds      = ds.mean('ens')
#     ds_all.append(ds)
    
# # Load Obs. ACFs
# ds_obs = xr.open_dataset(procpath + ncname_acf_obs).load().isel(ens=0)
# ds_all.append(ds_obs)


#%% Limit to a target month

kmonth = 1

acfs   = [ds.acf.isel(mons=kmonth,thres=0) for ds in ds_all]

# Take the ensemble average for now
#acfs[0] = acfs[]

# whatg iw as up to 
# Select Ens or avg
#Visualize T2

#%% Compute Re-emergence Index

rei_byexp = [proc.calc_remidx_xr(ds,return_rei=True) for ds in acfs]

#%% Plot the REI


fsz_ticks       = 18
label_ssh = False

fig,axs,_       = viz.init_orthomap(1,2,bboxplot,figsize=(20,10))


for ii in range(2):
    ax      = axs[ii]
    ax      = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)
    
    # Plot the REI
    plotvar = rei_byexp[ii]
    
    
    
    
    # Plot the SSH (Time Mean)
    if ii == 0:
        plotvar = ds_adt.mean('time')
        cl = ax.contour(plotvar.lon, plotvar.lat, plotvar.adt*100, colors="dimgray",alpha=0.8,
                        linewidths=0.75, transform=proj, levels=cints_adt)
        if label_ssh:
            ax.clabel(cl,fontsize=fsz_ticks-2)
            

#%%



    # SM_SST_Obs_Pilot_00_Tdcorr0_qnet_lag00to60_ALL_ensALL.nc




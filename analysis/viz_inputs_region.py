#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Stochastic Model Inputs Over a Region

- Copied upper section of viz_t2_rei_era
- Modeled on viz_inputs_paper_draft from reemergence/analysis

Created on Fri Apr 18 10:41:13 2025

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
figpath         = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20250523/"
sm_output_path  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/"
procpath        = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
input_path      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/"

expname         = "SST_ORAS5_avg_mld003"#"SST_Obs_Pilot_00_Tdcorr0_qnet_noPositive"
expname_long    = "ORAS5_MLD"#"Stochastic Model (with Re-emergence)"

# Indicate the Region
bbsel  = [-40,-15,52,62]
bbname = "SPGE"
bbname_long = "Eastern Subpolar Gyre"

# # Indicate Experiment and Comparison Name 
# comparename     = "kgm20250417"
# expnames        = ["SST_Obs_Pilot_00_Tdcorr0_qnet","ERA5_1979_2024"]
# expnames_long   = ["Stochastic Model (with Re-emergence)","ERA5 Reanalysis (1979-2024)"]
# expcols         = ["turquoise","k"]
# expls           = ["dotted",'solid']

#%% Functions (copied from viz_inputs_paper_draft)

def convert_ds(invar,lat,lon,):
    
    if len(invar.shape) == 4: # Include mode
        nmode = invar.shape[0]
        coords = dict(mode=np.arange(1,nmode+1),mon=np.arange(1,13,1),lat=lat,lon=lon)
    else:
        coords = dict(mon=np.arange(1,13,1),lat=lat,lon=lon)
    
    return xr.DataArray(invar,coords=coords,dims=coords)

#%% Load the inputs

# Load the Parameters
print("Loading inputs for %s" % expname)
expparams_raw   = np.load("%s%s/Input/expparams.npz" % (sm_output_path,expname),allow_pickle=True)
expparams       = scm.repair_expparams(expparams_raw)
paramset        = scm.load_params(expparams,input_path)

inputs,inputs_ds,inputs_type,params_vv = paramset

# Convert to the same units
convdict                               = scm.convert_inputs(expparams,inputs,return_sep=True)

# Get Lat/Lon
ds = inputs_ds['h']
lat = ds.lat.data
lon = ds.lon.data

# Convert t22o DataArray
varkeys = list(convdict.keys())
nk = len(varkeys)
conv_da     = {}
conv_ds     = []

for nn in range(nk):
    #print(nn)
    varkey = varkeys[nn]
    invar           = convdict[varkey]
    conv_da[varkey] = convert_ds(invar,lat,lon)
    
    # Also add to list to merge into dataset
    conv_ds.append(conv_da[varkey].rename(varkey))
    

# Also get the raw inputs
conv_ds_raw = []
varkeys_raw = list(inputs_ds.keys())
for kk in range(len(varkeys_raw)):
    
    varkey = varkeys_raw[kk]
    conv_ds_raw.append(inputs_ds[varkey].rename(varkey))
    
    
    
# Merge to Datasets
ds_inputs_converted = xr.merge(conv_ds)
ds_inputs = xr.merge(conv_ds_raw)

#%% Select parameters for the region



ds_inputs_reg = proc.sel_region_xr(ds_inputs,bbsel)




#%% Plot forcing, damping, mld

fsz_axis   = 28
fsz_ticks  = 18
fsz_legend = 24

fig,axs = viz.init_monplot(4,1,figsize=(8,18))

vnames      = ['lbd_a','Fprime','h','lbd_d']
vnames_long = ['Damping ($\lambda^a$)',"Stochastic Forcing ($F'$)","Mixed-Layer Depth ($h$)","Subsurface Damping ($\lambda^d$)"]
vunits      = [r"$\frac{W}{m^2 \, \degree C}$",r"$\frac{W}{m^2}$",r"$m$","$Correlation$"]
avg_later_byvar = []


vvcol = ["darkred","yellow","navy",'green']


for vv in range(4):
    ax = axs[vv]
    
    vname   = vnames[vv]
    plotvar = ds_inputs_reg[vname].data
    _,nlat,nlon = (plotvar).squeeze().shape
    plotvar = plotvar.reshape(12,nlat*nlon)
    
    
    if vv == 0: # For Hflx
        problem_pt  = np.zeros((nlat*nlon))
    
    lbl = 0
    avg_later = []
    for nn in tqdm.tqdm(range(nlat*nlon)):
        
        if lbl == 1:
            label_in = ""
        else:
            label_in = "Individual Point"
        
        if np.any(plotvar[:,nn] == 0):
            if vv == 0:
                problem_pt[nn] = 1
            continue
        else:
            ax.plot(mons3,plotvar[:,nn],alpha=0.2,label=label_in)
            avg_later.append(plotvar[:,nn])
        
        if label_in == "Individual Point":
            lbl = 1
    
    avg_later = np.array(avg_later)
    avg_later_byvar.append(avg_later)
    ax.plot(mons3,avg_later.mean(0),alpha=1,c=vvcol[vv],label="Region Average")
    
    if vv == 0: 
        ax.legend(fontsize=fsz_legend)
    ax.tick_params(labelsize=fsz_ticks)
    ax.set_ylabel("%s \n [%s]" % (vnames_long[vv],vunits[vv]),fontsize=fsz_axis)
    
problem_pt=problem_pt.reshape(nlat,nlon)

figname = "%sSM_Inputs_Scycle_%s.png" % (figpath,expname)
plt.savefig(figname,bbox_inches='tight',dpi=150)

#%% 

fsz_tick   = 24
fsz_axis   = 32

imon       = 1
vv         = 0

for imon in range(12):
    
    cints_mld   = np.arange(0,840,40)
    cints_frc   = np.arange(0,110,10)
    cints_dmp   = np.arange(-40,42,2)
    fig,ax,bbb  = viz.init_regplot("SPGE",fontsize=fsz_tick)
    
    # Plot the Forcing
    if vv == 0:
        
        plotvar     = ds_inputs.lbd_a.isel(mon=imon).squeeze()
        
        cints_in = cints_dmp
        cmap_in  = 'cmo.balance'
        
    else:
        
        plotvar     = ds_inputs.Fprime.isel(mon=imon).squeeze()
        
        cints_in = cints_frc
        cmap_in  = 'cmo.thermal'
    
     
        
    pcm         = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,     
                    cmap=cmap_in,transform=proj,
                    vmin=cints_in[0],vmax=cints_in[-1])
    
    
    cb = viz.hcbar(pcm,ax=ax,fontsize=fsz_tick)
    cb.set_label(r"%s %s [%s]" % (mons3[imon],vnames_long[vv],vunits[vv]),fontsize=fsz_axis)
    
    # Plot the Mixed-Layer
    plotvar     = ds_inputs.h.isel(mon=imon)
    cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                    levels=cints_mld,
                    colors='k',linewidths=0.75,
                    transform=proj)
    ax.clabel(cl,fontsize=fsz_tick)
    
    
    # PLot zero points
    plotvar = ds_inputs_reg.lbd_a
    viz.plot_mask(plotvar.lon,plotvar.lat,problem_pt.T,reverse=False,ax=ax,proj=proj,geoaxes=True)
    
    
    # Plot the sea ice edge
    plotvar = ds_masks.mask_mon
    cl      = ax.contour(plotvar.lon, plotvar.lat,
                    plotvar, colors="cyan",
                    linewidths=2, transform=proj, levels=[0, 1], zorder=1)
    
    
    
    # Plot the SSH (Time Mean) --------------
    #if ii == 1:
    plotvar = ds_adt.isel(time=imon)
    cl      = ax.contour(plotvar.lon, plotvar.lat, plotvar.adt*100, colors="w",alpha=0.8,
                    linewidths=0.55, transform=proj, levels=cints_adt)
    
    viz.plot_box(bbsel,ax=ax,proj=proj,color='yellow',linewidth=4,linestyle='solid')
    
    figname = "%sSPG_Box_Inputs_%s_%s_mon%02i.png" % (figpath,expname,vnames[vv],imon+1)
    plt.savefig(figname,dpi=150,bbox_inches='tight')
    
#%% Plot Subsurface damping



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Checking Observational Mixed-Layer Products


Created on Wed Sep  2 10:33:27 2026

@author: gliu

"""

import xarray as xr
import numpy as np
import sys
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

#%%
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import cvd_utils as cvd


#%% Additional Functions

def reformat_ds(ds,lonname=None,latname=None,monname=None):
    
    rdict = {}
    if lonname is not None:
        print("Renaming Lon")
        rdict[lonname] = 'lon'
    if latname is not None:
        print("Renaming Lat")
        rdict[latname] = 'lat'
    if monname is not None:
        print("Renaming Month")
        rdict[monname] = 'month'
    
    return ds.rename(rdict)

#%% Figure Paths
 
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20260902/"
proc.makedir(figpath)

#%% Load de Boyer 2007

deboypath = "/Users/gliu/Globus_File_Transfer/Observations/Mixed_Layer/deBoyer2007/"
ncnames   = ["mld_DT02_c1m_reg2.0.nc","mld_DR003_c1m_reg2.0.nc","mld_DReqDTm02_c1m_reg2.0.nc"]
dsboyers  = []
monname   = "time"

for nc in ncnames:
    ds = xr.open_dataset(deboypath + nc).load()
    dsboyers.append(ds)

dsboyers  = [reformat_ds(ds.mld,monname=monname) for ds in dsboyers]
dsboyers  = [proc.lon360to180_xr(ds) for ds in dsboyers]




#%% Load Hote et al. 2017

ncname  = "/Users/gliu/Globus_File_Transfer/Observations/Mixed_Layer/Holte_etal_2017/Argo_mixedlayers_monthlyclim_04142022.nc"
dsholte = xr.open_dataset(ncname).load()

def assign_latlon_holte(ds,dsholte):
    ds['iLAT'] = dsholte.lat
    ds['iLON'] = dsholte.lon
    return ds

latname         = "iLAT"
lonname         = "iLON"
monname         = "iMONTH"

# Get "Monthly mean of the MLDs found by the density algorithm in 1 degree bins"
holte_da_mean   = dsholte.mld_da_mean

# Get "Monthly mean of the MLDs found by the density threshold in 1 degree bins"
holte_dt_mean   = dsholte.mld_dt_mean

holte_mlds      = [holte_da_mean,holte_dt_mean]
holte_mlds      = [assign_latlon_holte(ds,dsholte) for ds in holte_mlds]
holte_mlds      = [reformat_ds(ds,lonname,latname,monname) for ds in holte_mlds]
holte_mlds      = [proc.lon360to180_xr(ds) for ds in holte_mlds]


#%% Load GOSML

ncname    = "/Users/gliu/Globus_File_Transfer/Observations/Mixed_Layer/GOSML/mixed_layer_properties_mean.nc"
dsgosml   = xr.open_dataset(ncname).load()

latname   = "latitude"
lonname   = "longitude"
monname   = None

gosml_mld = dsgosml.depth_mean
gosml_mld = reformat_ds(gosml_mld,lonname,latname,)


#%% Load Schmidtko et al. 2013

ncmimoc   = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/smio_data_final/MIMOC_regridERA5_h_pilot.nc"
dsmimoc   = xr.open_dataset(ncmimoc).load()

monname   = 'mon'
mimoc_mld = reformat_ds(dsmimoc.h,monname=monname)

#%% Get Deepest MLD

mld_all = dsboyers + holte_mlds + [gosml_mld,mimoc_mld]

dnames  = [
    "de Boyer et al. 2007\n(Temp. Fixed Threshold)",
    "de Boyer et al. 2007\n(Density Fixed Threshold)",
    "de Boyer et al. 2007\n(Density Variable Threshold)",
    "Holte et al. 2017\n(Density Algorithm)",
    "Holte et al. 2017\n(Density Fixed Threshold)",
    "GOSML\n(Jonhson and Lyman 2022)",
    "MIMOC\n(Schmidtko et al. 2013)",
    ]

bbsel    = [-80,5,40,65]
mlds_spg = [proc.sel_region_xr(ds,bbsel) for ds in mld_all]

mldmax   = [ds.max('month') for ds in mlds_spg]

#%% Set other Parameters

dcolors = [
    "coral",
    "hotpink",
    'firebrick',
    'royalblue',
    'cornflowerblue',
    'orange',
    'k'
    ]
dls = [
       'dashed',
       'dashed',
       'dashed',
       'dotted',
       'dotted',
       'dashdot',
       'solid',
       ]

bbox_spgne = [-40,-10,52,62]
bbox_labrador = [-60,-45,55,60]
#%% Plot it

ndata   = len(dnames)
bboxin  = [-80,0,40,65]
centlon = -40
figsize = (25,14)
proj    = ccrs.PlateCarree()
fsz_title = 32
cints   = np.arange(0,1300,100)
fsz_tick  = 18

fig,axs,_ = viz.init_orthomap(2,4,figsize=(32,12),
                             bboxplot=bboxin,centlon=centlon)

for a in range(ndata):
    ax = axs.flatten()[a]
    ax = viz.add_coast_grid(ax,bbox=bboxin,proj=proj)
    
    ax.set_title(dnames[a],fontsize=fsz_title)
    
    plotvar = mldmax[a]
    pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                            cmap='cmo.deep',vmin=0,vmax=800)
    
    cl     = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                            colors="k",levels=cints,linewidths=0.75)
    ax.clabel(cl,fontsize=fsz_tick)
    
    viz.plot_box(bbox_spgne,ax=ax,color='magenta',linewidth=2.5)
    viz.plot_box(bbox_labrador,ax=ax,color='cyan',linewidth=2.5)

fig.delaxes(axs.flatten()[-1])
cb      = viz.vcbar(pcm,ax=axs.flatten()[-1],pad=-0.1,fraction=0.025)
#cb      = viz.hcbar(pcm,ax=axs.flatten()[-1],pad=-0.5,fraction=0.15)
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label("Max Climatological MLD (meters)",fontsize=fsz_tick)
    
    
figname = "%sMax_MLD_SPG_Comparison.png" % (figpath)
plt.savefig(figname,dpi=250,bbox_inches='tight')


#%% Plot Difference with a Reference Map
# Note: I should Regrid this first...

refmap     = mldmax[-1]
cints_diff = np.arange(-1000,1100,100)
fig,axs,_ = viz.init_orthomap(2,4,figsize=(32,12),
                             bboxplot=bboxin,centlon=centlon)

for a in range(ndata):
    ax = axs.flatten()[a]
    ax = viz.add_coast_grid(ax,bbox=bboxin,proj=proj)
    
    ax.set_title(dnames[a],fontsize=fsz_title)
    
    plotvar = mldmax[a] - refmap
    pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                            cmap='cmo.balance',vmin=-250,vmax=250)
    
    cl     = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                            colors="k",levels=cints_diff,linewidths=0.75)
    ax.clabel(cl,fontsize=fsz_tick)

fig.delaxes(axs.flatten()[-1])
cb      = viz.vcbar(pcm,ax=axs.flatten()[-1],pad=-0.1,fraction=0.025)
#cb      = viz.hcbar(pcm,ax=axs.flatten()[-1],pad=-0.5,fraction=0.15)
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label("Difference with MIMOC (meters)",fontsize=fsz_tick)
    

figname = "%sMax_MLD_SPG_Comparison_DiffMIMOC.png" % (figpath)
plt.savefig(figname,dpi=250,bbox_inches='tight')

#%% Plot Difference over SPGNE region

#bbox_spgne = [-40,-10,52,62]
spgne_aavg = [proc.aavg(ds,bbox_spgne) for ds in mld_all]

#%% Plot Mean Seasonal Cycle for each case
mons3  = proc.get_monstr()
fig,ax = viz.init_monplot(1,1,figsize=(8,4.5))


for dd in range(ndata):
    plotvar = spgne_aavg[dd]
    ax.plot(mons3,plotvar,label=dnames[dd],lw=2.5,ls=dls[dd],c=dcolors[dd])
ax.legend(ncol=2)

ax.invert_yaxis()
ax.set_ylabel("SPGNE Average \nClimatological MLD (meters)",fontsize=16)

figname = "%sSPGNE_Seasonality_Comparison.png" % (figpath)
plt.savefig(figname,dpi=250,bbox_inches='tight')

#%% Repeat but for Labrador Sea
lab_aavg = [proc.aavg(ds,bbox_labrador) for ds in mld_all]

fig,ax = viz.init_monplot(1,1,figsize=(8,4.5))


for dd in range(ndata):
    plotvar = lab_aavg[dd]
    ax.plot(mons3,plotvar,label=dnames[dd],lw=2.5,ls=dls[dd],c=dcolors[dd])
ax.legend(ncol=2)

ax.invert_yaxis()
ax.set_ylabel("Labrador Sea Average \nClimatological MLD (meters)",fontsize=16)

figname = "%sLabrador_Seasonality_Comparison.png" % (figpath)
plt.savefig(figname,dpi=250,bbox_inches='tight')

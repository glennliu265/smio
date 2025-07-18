#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Quicky Visualize Mixed-Layer Depth in MIMOC

- 2025.07.17: Add comparison from ORAS5..

Created on Tue Jul 15 14:53:19 2025

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
import matplotlib.gridspec as gridspec

from scipy.io import loadmat
import matplotlib as mpl

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

#%% Copy from [regrid_mimoc_EN4]

figpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20250716/"


# Load and query MIMOC netCDFs
mimocpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/MIMOC_ML_v2.2_PT_S/"
mimoc_ncs = glob.glob(mimocpath+'*.nc')
mimoc_ncs.sort()
ds_mimocs = xr.open_mfdataset(mimoc_ncs,concat_dim='month',combine='nested').load()#.DEPTH_MIXED_LAYER.load()


#%% Pre-process MLD variable

bbnatl      = [-90,10,0,70]
latmimoc    = ds_mimocs.LATITUDE.isel(month=1).data
lonmimoc    = ds_mimocs.LONGITUDE.isel(month=1).data

mld         = ds_mimocs.DEPTH_MIXED_LAYER.data
coords      = dict(month=np.arange(1,13,1),lat=latmimoc,lon=lonmimoc)
mldraw      = xr.DataArray(mld,coords=coords,dims=coords,name="mld")


mldraw      = proc.lon360to180_xr(mldraw)

mldreg      = proc.sel_region_xr(mldraw,bbnatl)

mld_nonan   = xr.where(np.isnan(mldreg),0,mldreg)

monmax      = mld_nonan.argmax('month') + 1
monmax      = xr.where(np.isnan(mldreg.isel(month=0)),np.nan,monmax)

#%% Count month of Max


maxmons_plot = [1,2,3,12]
markers      = ["d","x","o","+"]
markercol    = ["red","cyan",'yellow','magenta']
mons3 = proc.get_monstr()


# Get Counts
bboxSPGNE    = [-40,-15,52,62]
monmax_reg = proc.sel_region_xr(monmax,bboxSPGNE)
nnmax      = np.array(monmax_reg.shape).prod()
nnsum      = []

for ii in range(len(maxmons_plot)):
    monsel = maxmons_plot[ii]
    mask   = np.where(monmax_reg==monsel,1,np.nan)
    nnsum.append(np.nansum(mask))
perc = np.array(nnsum)/nnmax * 100


#%% Select the region



xx,yy        = np.meshgrid(monmax.lon,monmax.lat)

plotmon     = False
proj        = ccrs.PlateCarree()

fsz_tick    = 32

cints       = np.arange(1,13,1)

fig,ax,bb   = viz.init_regplot(regname='SPGE',fontsize=fsz_tick)

if plotmon:
    plotvar     = monmax
    # pcm         = ax.contourf(plotvar.lon.data,
    #                         plotvar.lat.data,
    #                         plotvar,transform=proj,levels=cints,cmap='twilight')'
    pcm         = ax.pcolormesh(plotvar.lon.data,
                            plotvar.lat.data,
                            plotvar,transform=proj,vmin=1,vmax=12,cmap='twilight')
    # pcm         = ax.contour(plotvar.lon.data,
    #                         plotvar.lat.data,
    #                         plotvar,transform=proj,levels=cints,colors='w')
    # ax.clabel(pcm)
    


else:
    plotvar     = mld_nonan.max('month')#monmax
    #pcm     = plotvar.plot(ax=ax,transform=proj)
    pcm         = ax.pcolormesh(plotvar.lon.data,
                            plotvar.lat.data,
                            plotvar,transform=proj,vmin=0,vmax=450,cmap='cmo.deep')
    
    
    
for ii in range(len(maxmons_plot)):
    monsel = maxmons_plot[ii]
    mask   = np.where(monmax==monsel,1,np.nan)
    label  = "%s (%i, %.2f" % (mons3[ii],nnsum[ii],perc[ii]) + "%)"
    ax.scatter(xx*mask,yy*mask,
               c=markercol[ii],marker=markers[ii],s=80,
               transform=proj,label=label)
    #ymask  = np.where()

cb          = viz.hcbar(pcm,fontsize=fsz_tick)
cb.set_label("Max MLD in Seasonal Cycle [meters]",fontsize=fsz_tick)
ax.legend(fontsize=fsz_tick)
ax          = viz.add_coast_grid(ax,bbox=bb,fill_color='lightgray',fontsize=fsz_tick)


viz.plot_box(bboxSPGNE,ax=ax,color='purple',proj=proj,linewidth=4)

savename = "%sDeepest_MLD_and_Month_MIMOC.png" % figpath
plt.savefig(savename,dpi=150)


# =====================================
#%% Also Load ORAS5 Data (and process)
# =====================================

bbsel2          = [-40,-15,52,62]
reprocess_oras5 = False
# Set to False if you have already run this and regridded to ERA5 grid
# using the regrid_subsurface_damping_era5.py script.

if reprocess_oras5:
    
    # Load ORAS5 output (processed by crop_oras5_natl)
    ncoras     = "ORAS5_CDS_mld_NAtl_1979_2024.nc"
    pathoras   = "/Users/gliu/Globus_File_Transfer/Reanalysis/ORAS5/proc/"
    dsoras     = xr.open_dataset(pathoras + ncoras).load() #"/Users/gliu/Globus_File_Transfer/Reanalysis/ORAS5/proc/"
    
    #% Compute mean seasonal cycle and save
    hcycle_oras = dsoras.groupby('time.month').mean('time')
    ncout = "ORAS5_CDS_mld_NAtl_1979_2024_scycle.nc"
    hcycle_oras.to_netcdf(ncout)
    
    horas_reg = proc.sel_region_xr_cv(hcycle_oras,bbsel2,debug=False).mld
    #plt.scatter(horas_reg.TLONG,horas_reg.TLAT,c=horas_reg.mld.isel(month=0)),plt.colorbar()
    oras_aavg   = horas_reg.mean('nlat').mean('nlon')

else:
    
    ncpath      = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
    ncout       = "ORAS5_CDS_mld_NAtl_1979_2024_scycle_regridERA5.nc"
    hclim_oras  = xr.open_dataset(ncpath + ncout).load()
    
    horas_reg    = proc.sel_region_xr(hclim_oras,bbsel2).mld
    oras_aavg    = horas_reg.mean('lat').mean('lon')
    
#plt.scatter(dsoras.TLONG,dsoras.TLAT,c=dsoras.mld.isel(time=0)),plt.colorbar()


hclim_mimoc = mldraw.copy()
hmimoc_reg  = proc.sel_region_xr(mldraw,bbsel2)
mimoc_aavg  = hmimoc_reg.mean('lat').mean('lon')
#%%

fig,ax  = viz.init_monplot(1,1,)
mons3   = proc.get_monstr()
ax.plot(mons3,oras_aavg,label="ORAS5")
ax.plot(mons3,mimoc_aavg,label="MIMOC")

ax.legend()

#%% Compute Maximum and deepest MLD

def get_monmax(dsmld): # Get month of maximum MLD
    dsmld_nonan    = xr.where(np.isnan(dsmld),0,dsmld) # Set all NaN to zero
    monmax         = dsmld_nonan.argmax('month') + 1 # Get argmax
    monmax         = xr.where(np.isnan(dsmld.isel(month=0)),np.nan,monmax)
    return monmax

def count_monmax(monmax,bbreg,maxmons=np.arange(1,13,1),return_dict=True):
    monmax_reg = proc.sel_region_xr(monmax,bbreg)
    nnmax      = np.array(monmax_reg.shape).prod()
    nnsum      = []
    for ii in range(12):
        monsel = maxmons[ii]
        mask   = np.where(monmax_reg==monsel,1,np.nan)
        nnsum.append(np.nansum(mask))
    perc = np.array(nnsum)/nnmax * 100
    dictout = dict(
        countbymon=nnsum,
        percbymon=perc,
        ntotal=nnmax)
    return dictout

mnames    = ["MIMOC","ORAS5 0.03"]
mlds_in   = [hclim_mimoc,hclim_oras.mld]
maxmlds   = [ds.max('month') for ds in mlds_in]
monmaxes  = [get_monmax(ds) for ds in mlds_in]
regcounts = [count_monmax(mm,bboxSPGNE) for mm in monmaxes]

#%% Remake the plot above
ii           = 1

bboxspg     = [-80,0,40,65]


maxmons_plot = [1,2,3,12]
markers      = ["d","x","o","+"]
markercol    = ["red","cyan",'yellow','magenta']


fsz_tick     = 38
fig,ax,_ = viz.init_regplot(regname="SPGE",fontsize=fsz_tick)


plotvar     = proc.sel_region_xr(maxmlds[ii],bboxspg)
monmax      = proc.sel_region_xr(monmaxes[ii],bboxspg)
pcm         = ax.pcolormesh(plotvar.lon.data,
                        plotvar.lat.data,
                        plotvar,transform=proj,vmin=0,vmax=450,cmap='cmo.deep')


for im in range(len(maxmons_plot)):
    monsel = maxmons_plot[im]
    
    mask   = np.where(monmax==monsel,1,np.nan)
    xx,yy  = np.meshgrid(plotvar.lon,plotvar.lat)
    nnsum  = regcounts[ii]['countbymon'][im]
    perc   = regcounts[ii]['percbymon'][im]
    
    if ii == 0:
        ss = 80
    else:
        ss = 10
    
    label  = "%s (%i, %.2f" % (mons3[im],nnsum,perc) + "%)"
    ax.scatter(xx*mask,yy*mask,
               c=markercol[im],marker=markers[im],s=ss,
               transform=proj,label=label)
    #ymask  = np.where()

cb          = viz.hcbar(pcm,fontsize=fsz_tick)
cb.set_label("Max MLD in Seasonal Cycle [meters]",fontsize=fsz_tick)
ax.legend(fontsize=fsz_tick)
ax          = viz.add_coast_grid(ax,bbox=bb,fill_color='lightgray',fontsize=fsz_tick)

viz.plot_box(bboxSPGNE,ax=ax,color='purple',proj=proj,linewidth=4)
ax.set_title(mnames[ii],fontsize=fsz_tick+12)
savename    = "%sDeepest_MLD_and_Month_%s.png" % (figpath,mnames[ii])
plt.savefig(savename,dpi=150)

#%% Compare MLD over both regions 



cints = [
    np.arange(0,1250,50),
    np.arange(0,1250,50)
    
    ]

ii           = 0

bboxspg     = [-80,0,40,65]


maxmons_plot = [1,2,3,12]
markers      = ["d","x","o","+"]
markercol    = ["red","cyan",'yellow','magenta']


fsz_tick     = 38
fig,ax,_ = viz.init_regplot(regname="SPGE",fontsize=fsz_tick)


plotvar     = proc.sel_region_xr(maxmlds[ii],bboxspg)
monmax      = proc.sel_region_xr(monmaxes[ii],bboxspg)
pcm         = ax.contourf(plotvar.lon.data,
                        plotvar.lat.data,
                        plotvar,transform=proj,levels=cints[ii],cmap='cmo.deep')

cl = ax.contour(plotvar.lon.data,
                        plotvar.lat.data,
                        plotvar,transform=proj,levels=cints[ii],colors="k",linewidths=0.55)
ax.clabel(cl,fontsize=fsz_tick)


cb          = viz.hcbar(pcm,fontsize=fsz_tick,fraction=0.055)
cb.set_label("Max MLD in Seasonal Cycle [meters]",fontsize=fsz_tick)
ax.legend(fontsize=fsz_tick)
ax          = viz.add_coast_grid(ax,bbox=bb,fill_color='lightgray',fontsize=fsz_tick)

viz.plot_box(bboxSPGNE,ax=ax,color='purple',proj=proj,linewidth=4)
ax.set_title(mnames[ii],fontsize=fsz_tick+12)
savename    = "%sDeepest_MLD_%s.png" % (figpath,mnames[ii])
plt.savefig(savename,dpi=150)








#%% Take area mean


#%%




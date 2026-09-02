#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Regrid GOSML dataset to ERA5 Resolution (0.25 deg)

copied format from: combine_regrid_mimoc.py

Created Wed Sep 2 2026

@author: gliu
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import time
import sys
import cartopy.crs as ccrs
import glob
import xesmf as xe


#%% Import modules
stormtrack = 0
# if stormtrack:
#     sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
#     sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
    
#     # Path to the processed dataset (qnet and ts fields, full, time x lat x lon)
#     #datpath =  "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/hfdamping_RCP85/01_PREPROC/"
#     datpath =  "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/output/anom/"
#     figpath =  "/home/glliu/02_Figures/01_WeeklyMeetings/20240621/"
    
# else:
#     sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
#     sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")

#     # Path to the processed dataset (qnet and ts fields, full, time x lat x lon)
#     datpath =  "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/reanalysis/proc/"
#     figpath =  "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/02_Figures/20220511/"
# from amv import proc,viz
# import scm


# import amv.proc as hf # Update hf with actual hfutils script, most relevant functions
# import amv.loaders as dl

#%% GOSML load from mixed_layer_product_comparison.py

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

#%% Load


ncname    = "/Users/gliu/Globus_File_Transfer/Observations/Mixed_Layer/GOSML/mixed_layer_properties_mean.nc"
dsgosml   = xr.open_dataset(ncname).load()

latname   = "latitude"
lonname   = "longitude"
monname   = None

gosml_mld = dsgosml.depth_mean
gosml_mld = reformat_ds(gosml_mld,lonname,latname,)

#%%

#%% Load ERA5 Grid

figpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20250305/"

dpath_proc = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/reanalysis/proc/NATL_proc_obs/"
nc_era5    = "ERA5_sst_NAtl_1979to2021.nc"
ds_era5    = xr.open_dataset(dpath_proc + nc_era5).load()

lat_e = ds_era5.lat
lon_e = ds_era5.lon

# Regrid (Copied form predict_nasst/regrid_ocean_variable_hmxl.py)
method    = 'bilinear'
ds_out    = xr.Dataset({'lat':lat_e,"lon":lon_e})
ds        = gosml_mld

# Initialize Regridder
regridder = xe.Regridder(ds,ds_out,method,periodic=False)

# Regrid
daproc    = regridder(ds) # Need to input dataarray

daproc    = daproc.rename("mld")

#%% 

path_out   = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
outname    = path_out + "GOSML_RegridERA5_mld_Climatology.nc"

daproc.to_netcdf(outname)

#%% Visualize Regridding


bbplot  = [-80, 0, 35, 75]
proj    = ccrs.PlateCarree()
imon    = 1
mons3   = np.arange(1,13,1)
cints   = np.arange(0,550,50)

fig,axs = plt.subplots(1,2,figsize=(14,6),subplot_kw=dict(projection=proj))


for a,ax in enumerate(axs):
    ax.coastlines()
    
    if a == 0:
        title   = "Original (0.5$\degree$)"
        plotvar = ds.isel(month=imon)
        
    elif a == 1:
        title = "Regrid ERA5 (0.25$\degree$)"
        plotvar = daproc.isel(month=imon)
        
    # if a == 0:
    #     plotvar = plotvar.sel(lon=slice(bbplot[0],bbplot[1]),
    #                           lat=slice(bbplot[2],bbplot[3]))
    ax.set_extent(bbplot)
    
    ax.set_title(title)
    
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,vmin=0,vmax=500)
    cl  = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,levels=cints,
                     colors="k",linewidths=0.75)
    ax.clabel(cl)
cb = fig.colorbar(pcm,ax=axs.flatten(),pad=0.01,fraction=0.025)
#cb = viz.hcbar(pcm,ax=axs.flatten(),pad=0.01)
cb.set_label("%s Mixed Layer Depth [m]" % mons3[imon])

#plt.show()
savename = "%sMIMOC_MLD_ERA5_Regrid_Comparison.png" % figpath
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)


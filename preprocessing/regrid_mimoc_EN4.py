#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Regrid MIMOC mixed-layer depths to the EN4 resolution (0.5 deg to 1 deg)

Created on Tue Jun 24 12:34:51 2025

@author: gliu
"""

import xesmf as xe
import numpy as np
import matplotlib.pyplot as plt
import glob
import xarray as xr
import cartopy.crs as ccrs

#%% Load the datasets

# Load EN4 Profiles
en4path     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/EN4/proc/"
en4nc       = "EN4_3D_TEMP_NAtl_1900_2021.nc"
ds_en4      = xr.open_dataset(en4path + en4nc).temperature.load()


# Load and query MIMOC netCDFs
mimocpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/MIMOC_ML_v2.2_PT_S/"
mimoc_ncs = glob.glob(mimocpath+'*.nc')
mimoc_ncs.sort()
ds_mimocs = xr.open_mfdataset(mimoc_ncs,concat_dim='month',combine='nested').load()#.DEPTH_MIXED_LAYER.load()


#%% cCheck Resolutions

# EN4 is 1 degree (NATL)
laten4      = ds_en4.lat.data
lonen4      = ds_en4.lon.data
resen4      = [lonen4[1:] - lonen4[:-1] , laten4[1:] - laten4[:-1]]
print("Lon and Lat Differences for EN4:")
print(resen4)

# MIMOC is 0.5 degree (global)
latmimoc    = ds_mimocs.LATITUDE.isel(month=1).data
lonmimoc    = ds_mimocs.LONGITUDE.isel(month=1).data
resmimoc    = [lonmimoc[1:] - lonmimoc[:-1] , latmimoc[1:] - latmimoc[:-1]]
print("Lon and Lat Differences for MIMOC:")
print(resmimoc)

#%% Preprocess MIMOC for regridding

mld         = ds_mimocs.DEPTH_MIXED_LAYER.data
coords      = dict(month=np.arange(1,13,1),lat=latmimoc,lon=lonmimoc)
mldraw      = xr.DataArray(mld,coords=coords,dims=coords,name="mld")


#%% Perform Regridding

# Get Global Stuff
method  = 'bilinear'
lat1deg = np.arange(-90,90,1)
lon1deg = np.arange(-180,180,1)

# Set up regridder
lon_new      = lon1deg
lat_new      = lat1deg
ds_out       = xr.Dataset({'lat': (['lat'], lat_new), 'lon': (['lon'], lon_new) })

regridder     = xe.Regridder(mldraw, ds_out, method)
mld_regridded = regridder(mldraw)


#%% Save Output

outpath       = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
outname       = "%sMIMOC_RegridEN4_mld_%s_Global_Climatology.nc" % (outpath,method)

mld_regridded = mld_regridded.rename('mld')
edict = {'mld':{'zlib':True}}
mld_regridded.to_netcdf(outname,encoding=edict)

#%% Do a visual comparison

imon  = 1
inmlds = [mldraw,mld_regridded]
mldnames =["0.5 degree","1 degree"]
proj  = ccrs.PlateCarree()
bbsel = [-50,0,50,70]

cints   = np.arange(0,525,25)
fig,axs = plt.subplots(1,2,figsize=(12.5,4.5),
                       subplot_kw={'projection':proj},
                       constrained_layout=True)

for a,ax in enumerate(axs):
    ax.coastlines()
    ax.set_extent(bbsel)
    
    plotvar = inmlds[a].isel(month=imon)
    #pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,levels=cints)
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,vmin=0,vmax=cints[-1])
    ax.set_title(mldnames[a])
    
fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',pad=0.05,fraction=0.025)

plt.show()

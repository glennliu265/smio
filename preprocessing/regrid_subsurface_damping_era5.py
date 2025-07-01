#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Regrid Subsurface Damping from EN4 Grid (1deg) to ERA5 (0.25)

Created on Wed Jun 25 11:27:40 2025

@author: gliu

"""

import xesmf as xe
import numpy as np
import matplotlib.pyplot as plt
import glob
import xarray as xr
import cartopy.crs as ccrs


#%% Load Files

# Load Subsurface Damping, estimated in [calc_subsurface_damping_en4.py]
pathen4  = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
ncen4    = "EN4_MIMOC_corr_d_TEMP_detrendbilinear_lagmax3_interp1_ceil0_imshift1_dtdepth1_1979to2021.nc"
dsen4    = xr.open_dataset(pathen4 + ncen4).load().lbd_d

laten4 = dsen4.lat.data
lonen4 = dsen4.lon.data
bbox_en4 = [lonen4[0],lonen4[-1],laten4[0],laten4[-1]]


# Load Lat and Lon from ERA5
ncera5      = "ERA5_sst_NAtl_1979to2024.nc"
pathera5    = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
dsera5      = xr.open_dataset(pathera5+ncera5)

dsera5      = dsera5.sel(lon=slice(bbox_en4[0],bbox_en4[1]),lat=slice(bbox_en4[2],bbox_en4[3]))
lonera5     = dsera5.lon.load()
latera5     = dsera5.lat.load()



#%% Prepare Regridding

method         = 'bilinear'
ds_out         = xr.Dataset({'lat': (['lat'], latera5.data), 'lon': (['lon'], lonera5.data) })
regridder      = xe.Regridder(dsen4, ds_out, method)
lbdd_regridded = regridder(dsen4)
lbdd_regridded = lbdd_regridded.rename('lbd_d')


#%%


ncen4out    = "EN4_MIMOC_corr_d_TEMP_detrendbilinear_lagmax3_interp1_ceil0_imshift1_dtdepth1_1979to2021_regridERA5.nc"
ncout       = pathen4 + ncen4out
edict       = {'lbd_d' : {'zlib':True}}
lbdd_regridded.to_netcdf(ncout,encoding=edict)

#%% Compare Regridding
imon = 9

proj        = ccrs.PlateCarree()
lbdin       = [dsen4,lbdd_regridded]
lbdnames    = ["EN4 (1deg)","EN4 (0.25deg)"]
bbsel       = [-50,0,50,65]

fig,axs = plt.subplots(1,2,)

cints   = np.arange(0,1.05,.05)
fig,axs = plt.subplots(1,2,figsize=(12.5,4.5),
                       subplot_kw={'projection':proj},
                       constrained_layout=True)

for a,ax in enumerate(axs):
    ax.coastlines()
    ax.set_extent(bbsel)
    
    plotvar = lbdin[a].isel(mon=imon)
    #pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,levels=cints)
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,vmin=0,vmax=cints[-1])
    ax.set_title(lbdnames[a])

fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',pad=0.05,fraction=0.025)
plt.show()

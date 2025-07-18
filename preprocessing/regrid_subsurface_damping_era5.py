#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Regrid Subsurface Damping from EN4 Grid (1deg) to ERA5 (0.25)
Also regrid from curvilinear ORAS5 estimates to ERA5...


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

dataset_name = "oras5_mld_clim_cds"

#"ORAS5_avg" 
# EN4
#ORAS5
# Load Subsurface Damping, estimated in [calc_subsurface_damping_en4.py]

if dataset_name == "EN4":
    pathen4  = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
    ncin    = pathen4 + "EN4_MIMOC_corr_d_TEMP_detrendbilinear_lagmax3_interp1_ceil0_imshift1_dtdepth1_1979to2021.nc"
    dsen4    = xr.open_dataset(ncin).load().lbd_d
    vname = "lbd_d"
    ds_in    = dsen4
    
    laten4 = dsen4.lat.data
    lonen4 = dsen4.lon.data
    bbox   = [lonen4[0],lonen4[-1],laten4[0],laten4[-1]]
    
elif dataset_name == "ORAS5":
    path_ora = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
    ncin     = path_ora + "ORAS5_MIMOC_corr_d_TEMP_detrendRAW_lagmax3_interp1_ceil0_imshift1_dtdepth1_1979to2024.nc"
    dsora    = xr.open_dataset(ncin).load()#.lbd_d
    vname = "lbd_d"
    ds_in = dsora
    
    tlat = dsora.TLAT.data
    tlon = dsora.TLONG.data
    bbox = [np.nanmin(tlon.flatten()),
            np.nanmax(tlon.flatten()),
            np.nanmin(tlat.flatten()),
            np.nanmax(tlat.flatten())]
    
    # Fix some issues with duplicate nlon.nlat
    ds_in = ds_in.drop(['nlat','nlon'])
    ds_in = ds_in.rename(dict(lat='nlat',lon='nlon'))
    
elif dataset_name == "ORAS5_avg":
    path_ora = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
    ncin     = path_ora + "ORAS5_avg_MIMOC_corr_d_TEMP_detrendRAW_lagmax3_interp1_ceil0_imshift1_dtdepth1_1979to2024.nc"
    dsora    = xr.open_dataset(ncin).load()#.lbd_d
    vname = "lbd_d"
    ds_in = dsora
    
    tlat = dsora.TLAT.data
    tlon = dsora.TLONG.data
    bbox = [np.nanmin(tlon.flatten()),
            np.nanmax(tlon.flatten()),
            np.nanmin(tlat.flatten()),
            np.nanmax(tlat.flatten())]
    
    # Fix some issues with duplicate nlon.nlat
    ds_in = ds_in.drop(['nlat','nlon'])
    ds_in = ds_in.rename(dict(lat='nlat',lon='nlon'))
elif dataset_name == "oras5_mld_clim_cds":
    
    
    ncname = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
    ncout  = "ORAS5_CDS_mld_NAtl_1979_2024_scycle.nc"
    ncin    = ncname + ncout
    ds_in     = xr.open_dataset(ncin)
    
    tlat = ds_in.TLAT.data
    tlon = ds_in.TLONG.data
    bbox = [np.nanmin(tlon.flatten()),
            np.nanmax(tlon.flatten()),
            np.nanmin(tlat.flatten()),
            np.nanmax(tlat.flatten())]
    vname = "mld"
    
else:
    
    print("Currently only supports the following datasets:\n")
    print("\tEN4")
    print("\tORAS5")
    print("\tORAS5_avg")
    print("\toras5_mld_clim_cds")

# Load Lat and Lon from ERA5
ncera5      = "ERA5_sst_NAtl_1979to2024.nc"
pathera5    = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
dsera5      = xr.open_dataset(pathera5+ncera5)

dsera5      = dsera5.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
lonera5     = dsera5.lon.load()
latera5     = dsera5.lat.load()


#%% Examine a specific month (debugging for ORAS5)

plt.scatter(ds_in.TLONG,ds_in.TLAT,c=ds_in[vname].isel(mon=11)),plt.colorbar(),plt.show()
plt.scatter(ds_in.TLONG,ds_in.TLAT,c=ds_in[vname].isel(mon=9)),plt.colorbar(),plt.show()

#%% Prepare Regridding


# Do some regridding for subsurface damping
if dataset_name == "ORAS5":
    ds_in = ds_in.rename({"TLONG": "lon", "TLAT": "lat",})
    
method         = 'bilinear' #'bilinear'
ds_out         = xr.Dataset({'lat': (['lat'], latera5.data), 'lon': (['lon'], lonera5.data) })
regridder      = xe.Regridder(ds_in, ds_out, method)
lbdd_regridded = regridder(ds_in)

# Set incorrectly regridded estimates to zero
lbdd_regridded = xr.where(lbdd_regridded ==0.,np.nan,lbdd_regridded)
if dataset_name == "EN4": # Not sure if this is necessary, i guess i made it into a data array
    lbdd_regridded = lbdd_regridded.rename(vname)


#%%
def addstrtoext(name,addstr,adjust=0):
    """
    Add [addstr] to the end of a string with an extension [name.ext]
    Result should be "name+addstr+.ext"
    -4: 3 letter extension. -3: 2 letter extension
    """
    return name[:-(4+adjust)] + addstr + name[-(4+adjust):]


ncout       = addstrtoext(ncin,"_regridERA5",adjust=-1)
edict       = {vname : {'zlib':True}}
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

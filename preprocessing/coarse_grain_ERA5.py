#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Coarse Grain ERA5 Output and see how this might impact the re-emergence signal

Created on Wed Jun 18 13:57:47 2025

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
from scipy.io import loadmat
import matplotlib as mpl
import xesmf as xe

#%% Load Custom Modules

# local device (currently set to run on Astraeus, customize later)
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import cvd_utils as cvd

#%% Path to Data

# Load the SST
dpath   = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
nc1     = "ERA5_sst_NAtl_1979to2024.nc"
nc2     = "ERA5_sst_NAtl_1940to1978.nc"
ncs     = [nc2,nc1]

dsall   = [xr.open_dataset(dpath + nc).load() for nc in ncs]

dsraw   = xr.concat(dsall,dim='time')
sstraw  = dsraw.sst


#%% Set Up New Grid
deg       = 1
method    = 'bilinear'
lon_ori   = sstraw.lon.data
lat_ori   = sstraw.lat.data



lon_new   = np.arange(lon_ori[0],lon_ori[-1]+deg,deg)
lat_new   = np.arange(lat_ori[0],lat_ori[-1]+deg,deg)

#% Visualize New Grid

xxori,yyori = np.meshgrid(lon_ori,lat_ori)
xxnew,yynew = np.meshgrid(lon_new,lat_new)


# fig,ax,_ = viz.init_regplot(regname="NAT")

# proj     = ccrs.PlateCarree()

# sc1 = ax.scatter(xxori,yyori,c='k',s=0.05,marker="o",transform=proj)


# sc2 = ax.scatter(xxnew,yynew,c='b',s=5,marker="x",transform=proj)

# ax.set_extent([-40,-30,50,60])
# plt.show()

#%%


ds_out       = xr.Dataset({'lat': (['lat'], lat_new), 'lon': (['lon'], lon_new) })
regridder    = xe.Regridder(sstraw, ds_out, method)

ds_regridded = regridder(sstraw)


#%% Plot the Output to check

fig,ax,_ = viz.init_regplot(regname="NAT")


#plotvar = sstraw.isel(time=0)
plotvar = ds_regridded.isel(time=0)
pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,vmin=270,vmax=285)

sc1      = ax.scatter(xxori,yyori,c='k',s=0.05,marker="o",transform=proj)
sc2      = ax.scatter(xxnew,yynew,c='b',s=5,marker="x",transform=proj)

ax.set_extent([-40,-30,50,60])
plt.show()



#%%
fname_out = "%sERA5_sst_NAtl_1940to2024_regrid_%ideg_%s.nc" % (dpath,deg,method)
ds_regridded = ds_regridded.rename('sst')
edict = proc.make_encoding_dict(ds_regridded)
ds_regridded.to_netcdf(fname_out,encoding=edict)
print("Saved regridded output to %s" % fname_out)
#proc.addstrtoext(adjust=-1)


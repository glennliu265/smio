#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Analyze the AMOC Index in the CESM2 PiControl Simulationß

Created on Tue Sep 23 13:37:56 2025

@author: gliu
"""


import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import os
import matplotlib.patheffects as PathEffects
import glob

#%%


amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
scmpath = "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/" # scm module

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl

#%% User Edits

# b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.MOC.100001-109912.nc

datpath         = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/FCM/ocn/MOC/"

datpath_index   = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/05_SMIO/data/region_average/SPGNE_lon320to345_lat052to062/"
spgnenc         = "CESM2_FCM_PiControl_SST_0000_2000_raw.nc"

#%% Get list of AMOC files (and load)

"""

Transport Components
[b'Total', b'Eulerian-Mean Advection',
       b'Eddy-Induced Advection (bolus) + Diffusion',
       b'Eddy-Induced (bolus) Advection', b'Submeso Advection']

Transport Regions
[b'Global Ocean - Marginal Seas',
       b'Atlantic Ocean + Mediterranean Sea + Labrador Sea + GIN Sea + Arctic Ocean + Hudson Bay']


"""

# Transport Components
comps = [b'Total', b'Eulerian-Mean Advection',
       b'Eddy-Induced Advection (bolus) + Diffusion',
       b'Eddy-Induced (bolus) Advection', b'Submeso Advection']


nclist = glob.glob(datpath + "*.nc")
nclist.sort()

dsall = xr.open_mfdataset(nclist,combine='nested',concat_dim='time')

keepvars = ["MOC","time",
            #"transport_regions","transport_components",
            "transport_reg","moc_comp",
            "moc_z","lat_aux_grid",
            "TLAT","z_t"]

dsall = proc.ds_dropvars(dsall,keepvars)
dsall = dsall.isel(transport_reg=1)

dsall = dsall.sel(lat_aux_grid=slice(0,90))

st = time.time()
dsall = dsall.load()
proc.printtime(st,print_str="Loaded")

st = time.time()
outpath = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/proc/"
ncout   = "%sMOC_merge_0to90.nc" % outpath
edict = proc.make_encoding_dict(dsall)
dsall.to_netcdf(ncout,encoding=edict)
proc.printtime(st,print_str="Saved")

#%% Load SPGNE Averaged Index

dsspg = xr.open_dataset(datpath_index + spgnenc).load() # 70 sec

#%% Calculate some AMOC Indices

# First, find the maximum value along depth
amocidx  = dsall.max('moc_z')
depthmax = dsall.idxmax('moc_z')
amoc26   = amocidx.sel(lat_aux_grid=26,method='nearest')


amocmean = dsall.MOC.mean('time')

#%% Deseasonalize and detrend

def preprocess_ds(ds):
    dsa = proc.xrdetrend(ds)
    dsa = proc.xrdetrend(dsa)
    return dsa

spga    = preprocess_ds(dsspg.SST.squeeze())
amoca   = preprocess_ds(dsall.MOC) # 105 sec + 90 sec


#%% Let's check the regression patterns (Latitude x Depth) to SPGNE

for mc in range(3):

mc    = 1
mocin = amoca.isel(moc_comp=mc)

# Do a regression

mocarr = mocin.transpose('moc_z','lat_aux_grid','time').data
spgarr = spga.data

st     = time.time()
output = proc.regress_ttest(mocarr,spgarr)
proc.printtime(st,print_str="Completed regression in")

#% Make the plot

#%%

levels  = np.arange(-30,32,2)
fig,ax  = plt.subplots(1,1,constrained_layout=True,figsize=(12.5,6))

plotvar = output['regression_coeff']
mask    = output['sigmask']
lat     = mocin.lat_aux_grid.data
z       = mocin.moc_z.data/100

pcm     = ax.pcolormesh(lat,z,plotvar,cmap='cmo.balance',vmin=-2.5,vmax=2.5)

#cc  = ax.contour(lat,z,plotvar,colors='blue',levels=[0,])

sigp    = viz.plot_mask(lat,z,mask.T,ax=ax,color='gray')

cl      = ax.contour(lat,z,amocmean.isel(moc_comp=mc),colors="k",levels=levels,linewidths=.75)

#ax.set_ylim([0,200])
ax.invert_yaxis()

cb      = viz.hcbar(pcm,ax=ax)
cb.set_label("Sv MOC Transport per degC SPGNE Index")
ax.set_title(comps[mc])

ax.set_xlabel("Latitude")
ax.set_ylabel("Depth (meters)")


plt.show()

#dtout = proc.detrend_by_regression(mocin,spga)














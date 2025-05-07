#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Examine the sea ice for ERA5, particularly over the SPGNE
Also make a land-ice mask for ERA5

Created on Wed May  7 14:37:21 2025

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

import cmcrameri as cm

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


bbox_spgne = [-40,-15,52,62]

#%% Load the data

# Load Ice Mask
ncpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
ncice  =  ncpath + "ERA5_siconc_1940_2024_NATL.nc"
dsice  = xr.open_dataset(ncice)
dsice  = dsice.sel(time=slice('1979-01-01','2024-12-31')).siconc.load()


# Load SST to get land
ncsst = ncpath + "ERA5_sst_NAtl_1979to2024.nc"
dssst = xr.open_dataset(ncsst)

dssst = dssst.isel(time=0).load().sst

# Load Heat Flux
#siconc.load()

#%% Check Mean and Max Sea Ice Concentration

icemean = dsice.mean('time')
icemax  = dsice.max('time')



#%%

cints_max = np.arange(0,0.06,0.01)
fig,ax,_  = viz.init_regplot(regname="SPGE")
#fig,ax,_  = viz.init_regplot()


invar     = icemax
iceplot   = xr.where(invar <= 0.05,np.nan,invar)

plotvar   = iceplot #iceplot

pcm       = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                        cmap='cmo.ice',vmin=0.05,vmax=1.00,
                        transform=proj)
cl     = ax.contour(plotvar.lon,plotvar.lat,invar,
                        colors="red",levels=[0.05,],
                        transform=proj,linewidths=2)
ax.clabel(cl)

plotvar = icemean
cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                        colors="yellow",levels=[0.05,],
                        transform=proj,linewidths=2,linestyles='dotted')
ax.clabel(cl)


# pcm     = ax.contour(plotvar.lon,plotvar.lat,plotvar,
#                         colors="k",levels=cints_max,
#                         transform=proj)
#ax.clabel(pcm)

cb = viz.hcbar(pcm)
viz.plot_box(bbox_spgne,ax=ax,color='limegreen',proj=proj,linewidth=2.5)


#%% Make new Ice Mask

icemask_new  = xr.where(icemax > 0.05,np.nan,1)
icemask_new  = xr.where(np.isnan(dssst),np.nan,icemask_new)
outpath_mask = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/masks/"
icemask_new  = icemask_new.rename('mask')
outname      = "%sERA5_1979_2024_limask_0.05p.nc" %  outpath_mask
edict        = proc.make_encoding_dict(icemask_new)
icemask_new.to_netcdf(outname,encoding=edict)




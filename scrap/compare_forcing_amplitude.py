#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Check the forcing amplitude for EOF-based and stdev based forcing

Created on Fri Sep 19 16:13:38 2025

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

bboxplot                    = [-80, 0, 20, 65]
mpl.rcParams['font.family'] = 'Avenir'
mons3                       = proc.get_monstr(nletters=3)

fsz_tick    = 18
fsz_axis    = 20
fsz_title   = 16

rhocrit     = proc.ttest_rho(0.05, 2, 86)
proj        = ccrs.PlateCarree()

#%% Load some other things to plot

# # Load Sea Ice Masks
dpath_ice = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/reanalysis/proc/NATL_proc_obs/"
nc_masks = dpath_ice + "OISST_ice_masks_1981_2020.nc"
ds_masks = xr.open_dataset(nc_masks).load()

#%% Load forcings

fpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/forcing/"


nc1     = "ERA5_Fprime_QNETpilotObsAConly_std_pilot.nc"
nc2     = "ERA5_Fprime_QNET_timeseries_QNETpilotObsAConly_nroll0_NAtl_EOFFilt090_corrected.nc"

#ncs    = [nc1,nc2]
ds1     = xr.open_dataset(fpath+nc1).load()
ds2     = xr.open_dataset(fpath+nc2).load()


#%%

eof_amp             = np.sqrt((ds2.Fprime**2).sum('mode'))
eof_amp_total       = eof_amp+ds2.correction_factor

lon = ds1.lon.data
lat = ds1.lat.data


#%%

dsin        = [ds1.Fprime.squeeze(),eof_amp_total,eof_amp]
expnames    = ["Stdev Forcing","EOF Forcing","EOF Forcing (no correction)"]
expcols     = ["hotpink","midnightblue","cornflowerblue"]
els         = ["solid","dashed","dotted"]

# Plot the values at a point
lonf        = -35
latf        = 55

mons3 = proc.get_monstr()

dspt = [proc.selpt_ds(ds,lonf,latf,) for ds in dsin]


fig,ax = viz.init_monplot(1,1)

for ex in range(3):
    ax.plot(mons3,dspt[ex],c=expcols[ex],label=expnames[ex],ls=els[ex])
    
ax.legend()

#%%
## Save for EOF-based forcing
# scrappath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/scrap/"
# outname   = scrappath + "EOF_forcing_reconstructed.npy"
# np.save(outname,forcing_in)


# Save for EOF-based forcing
scrappath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/scrap/"
outname   = scrappath + "Fstd_forcing_reconstructed.npy"
np.save(outname,forcing_in)




#%% Load Data from stormtrack

scrappath   = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/scrap/"
ncnames     = ["Fstd_forcing_reconstructed.npy","EOF_forcing_reconstructed.npy"]

dsf         = [np.load(scrappath+ds) for ds in ncnames]

#%%

o = 44
a = 11
fig,ax = viz.init_monplot(1,1)

for ex in range(2):
    plotvar = np.std(dsf[ex][:,a,o].reshape(1000,12),0)
    ax.plot(mons3,plotvar,c=expcols[ex],label=expnames[ex],ls=els[ex])
    
ax.legend()


#%% Look at spatial pattern in the difference of forcing

dsfpat = [np.nanstd(ds,0) for ds in dsf]

fig,ax = plt.subplots(1,1)

#for ex in range(2):
diff   = dsfpat[1] - dsfpat[0]
pcm    = ax.pcolormesh(diff,vmin=-.05,vmax=.05,cmap='cmo.balance')

print(np.abs(np.nanmax(np.abs(diff))))

cb = viz.hcbar(pcm,ax=ax)



#%% Lets try doing so with different white noise timeseries

tscont = np.random.normal(0,1,12000)

tsmon = []
for ii in range(12):
    ts = np.random.normal(0,1,(1000))
    tsmon.append(ts)
    
tsmon = np.array(tsmon)
tsmon_flatten = tsmon.T.flatten()

#%%

tsall = [tscont,tsmon_flatten]

tsyrmon = [ts.reshape(1000,12) for ts in tsall]
tsmonvar = [np.nanvar(ts,0) for ts in tsyrmon]

#%% PLot it

fig,ax = viz.init_monplot(1,1)

for ex in range(2):
    plotvar = tsmonvar[ex]
    ax.plot(mons3,plotvar,c=expcols[ex],label=expnames[ex],ls=els[ex])
    
ax.legend()


#%%

tsmon  = np.random.normal(0,1,(1000,12))




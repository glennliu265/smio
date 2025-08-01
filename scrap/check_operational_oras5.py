#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Check to see how similar the data is between operational and consolidated ERA5

Created on Wed Jul  9 14:58:59 2025

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

dpath   = "/Users/gliu/Globus_File_Transfer/Reanalysis/ORAS5/proc/"
nc1     = "ORAS5_opa0_TEMP_NAtl_1979_2018.nc"
nc2     = "ORAS5_CDS_TEMP_NAtl_2019_2024.nc"

#%%

dspre  = xr.open_dataset(dpath + nc1)
dspost = xr.open_dataset(dpath + nc2)

#%% Check a point

lonf = -40
latf = 55

pt1 = proc.find_tlatlon(dspre,lonf,latf).isel(z_t=0).load()
pt2 = proc.find_tlatlon(dspost,lonf,latf).isel(z_t=0).load()

#%% Plot the timeseries

t1      = pt1.TEMP
t2      = pt2.TEMP

tcomb   = xr.concat([t1,t2],dim='time')

fig,ax  = plt.subplots(1,1,figsize=(12.5,4))
ax.plot(t1.time,t1)
ax.plot(t2.time,t2)
ax.plot(tcomb.time,tcomb)
ax.set_xlims(['2017-01-01','2020-12-31'])

#%%

pre_snapshot = dspre.isel(time=-1).load()
post_snapshot = dspost.isel(time=0).load()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Using output from get_point_oras5.py
Compare output from each opa with copernicus download to understand which one is which...

Created on Mon Jul  7 13:54:10 2025

@author: gliu
"""


import xarray as xr
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import tqdm

amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import cvd_utils as cvd
#%%

ncpath   = "/Users/gliu/Globus_File_Transfer/Reanalysis/ORAS5/proc/point_data/"
ncsearch = "votemper_1979to1981__lon295_lat55_opa%02i.nc"
ncc = "votemper_1979to1981__lon295_lat55_copernicus.nc"


opas     = np.arange(5)
ds_all   = [xr.open_dataset(ncpath+ncsearch % (op)).load() for op in opas]


dsc      = xr.open_dataset(ncpath+ncc)


#%% calculate mean seasonal cycle

# Take Mean of All (Incl. Copernicus)
meanall = dsc.votemper.copy()
for op in opas:
    meanall += ds_all[op].votemper
meanall     = meanall / 6
    

mean_scycle = meanall.groupby('time.month').mean('time')


#%% Check the Timeseries

opa_col = ["cyan","blue","lightseagreen","orange","hotpink"]
lw = 2.5

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12.5,4))


for op in opas:
    
    plotvar = ds_all[op].votemper.groupby('time.month') - mean_scycle
    ax.plot(plotvar,label="opa%0i" % op,lw=lw,c=opa_col[op])

# Plot Coerpnicus Download
plotvar = dsc.votemper.groupby('time.month') - mean_scycle
ax.plot(plotvar,label="CDS_Download",c='k',ls='dashed',lw=lw)

# Plot All Mean
plotvar = meanall.groupby('time.month')  - mean_scycle
ax.plot(plotvar,label="All Mean",c='violet',ls='dotted',lw=lw)
ax.legend()


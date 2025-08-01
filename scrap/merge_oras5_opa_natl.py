#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Using output from crop_oras5_natl...
merge and take the ensemble average (1979-2024)


Created on Thu Jul 10 15:47:13 2025

@author: gliu

"""

import xarray as xr
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import tqdm

#%%
device = "Astraeus"

if device == "Astraeus":

    # local device (currently set to run on Astraeus, customize later)
    amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
    scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

elif device == "stormtrack":
    amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
    scmpath = "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/" # scm module

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl

#%% Indicate Paths

# Find and loop for a single file (1 year)
if device == 'Astraeus':
    dpath      = "/Users/gliu/Globus_File_Transfer/Reanalysis/ORAS5/proc/"
elif device == "stormtrack":
    dpath      = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/05_SMIO/data/ORAS5/proc/"
    
# Indicate a point for checking
lonf = -30
latf = 50


#%% For Consolidated Period (1979-2018), Load each output

# For each opa
ops     = np.arange(5)
ystart  = 1979   #2019#1979
yend    = 2018   # 2024#2018

test_timeseries = [] # Test Timeseries to see averaging
for op in tqdm.tqdm(range(len(ops))):
    
    outname      = "%sORAS5_opa%i_TEMP_NAtl_%04i_%04i.nc" % (dpath,op,ystart,yend)
    print(outname)
    
    ds   = xr.open_dataset(outname).load()
    dspt = proc.find_tlatlon(ds,lonf,latf)
    
    if op == 0:
        ds_add = ds.copy() # Make DataArray to accumulate values
    else:
        ds_add += ds
    test_timeseries.append(dspt.copy())    
    dspt.close()
    ds.close()

ds_avg   = ds_add / len(ops)
test_avg = proc.find_tlatlon(ds_avg,lonf,latf)
xrm      = xr.concat(test_timeseries,dim='opa')

#%% Check plot to make sure average was properly taken

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12.5,4.5))
for op in tqdm.tqdm(range(len(ops))):
    plotvar = test_timeseries[op]
    ax.plot(plotvar.TEMP.isel(z_t=1),label="opa%i" % op,lw=2)
    
ax.plot(xrm.mean('opa').TEMP.isel(z_t=1),label="Mean Manual",lw=1.5,color='cyan')
ax.plot(test_avg.TEMP.isel(z_t=1),label="Mean",ls='dashed',color="k")
ax.legend()
ax.set_xlim([100,150])

#%% Now load the operation production

ds_new  = xr.open_dataset(dpath + "ORAS5_CDS_TEMP_NAtl_2019_2024.nc").load()
ds_full = xr.concat([ds_avg,ds_new],dim='time')
    
#%% Check timeseries

dspt = proc.find_tlatlon(ds_full,lonf,latf)
plt.plot(dspt.time,dspt.TEMP.isel(z_t=11))


#%% Save final output


ncout_fin = "%sORAS5_opaAVG_TEMP_NAtl_1979_2024.nc" % (dpath)
edict     = proc.make_encoding_dict(ds_full)
ds_full.to_netcdf(ncout_fin,encoding=edict)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Load files and take the area average
Concatenate ensemble x time into a continuous timeseries...

Supports output from [run_SSS_basinwide], set smoutput=True


Created on Tue May  6 15:08:16 2025

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

#%%
# stormtrack

amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
scmpath = "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/" # scm module

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl

#%% User Edits

smoutput = True # Set to True for stochastic model output

# If smoutput is True...
expname  = "SST_ERA5_1979_2024"
vname    = "SST"
if smoutput:
    sm_output_path = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/sm_experiments/"
    outpath        = "%s%s/Metrics/" % (sm_output_path,expname) # Save into experiment directory

# Otherwise, specify an ncsearch
#ncpath = ""
ncsearch = ""


# Indicate cropping region
regname = "SPGNE"
bbsel   = [-40,-15,52,62]
locfn,locstring = proc.make_locstring_bbox(bbsel,lon360=True)
bbfn    = "%s_%s" % (regname,locfn)

if smoutput:
    outname = "%sArea_Avg_%s.nc" % (outpath,bbfn)

#%% Declare some functions

def reshape_ens2year(ds):
    # Convert Runs to Year
    ds          = ds.transpose('ens','time')
    nens,ntime  = ds.shape
    arr         = ds.data.flatten()
    timedim      = xr.cftime_range(start='0000',periods=nens*ntime,freq="MS",calendar="noleap")
    coords      = dict(time=timedim)
    return xr.DataArray(arr,coords=coords,dims=coords,name=ds.name)


#%% Load the files

if smoutput:
    nclist = dl.load_smoutput(expname,output_path=sm_output_path,return_nclist=True)
else:
    nclist = glob.glob(ncsearch)
    nclist.sort()
    
print("Found %i files" % (len(nclist)))
print(nclist)

#%% Loop for each file

aavgs  = []
nfiles = len(nclist)
for ff in tqdm.tqdm(range(nfiles)):
    
    ncname = nclist[ff]
    ds     = xr.open_dataset(ncname)
    dsreg  = proc.sel_region_xr(ds,bbsel).load()
    dsavg  = proc.area_avg_cosweight(dsreg)
    aavgs.append(dsavg)
    

# ds     = xr.open_dataset(nclist[0])
# dsreg  = proc.sel_region_xr(ds,bbsel).load()

#%% Do further processing

# if smoutput:
#     aavgs   = xr.concat(aavgs,dim='ens')
#     da_in   = aavgs.SST
#     da_out  = reshape_ens2year(da_in)
    
if len(nclist) > 1:
    aavgs   = xr.concat(aavgs,dim='ens')
    da_in   = aavgs[vname]
    da_out  = reshape_ens2year(da_in)
else:
    
    da_out = aavgs[0]
    
edict = proc.make_encoding_dict(da_out)
da_out.to_netcdf(outname,encoding=edict)
    
print("File saved to %s" % outname)


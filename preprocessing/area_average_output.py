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

#%% stormtrack modules

amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
scmpath = "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/" # scm module

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl

#%% User Edits: Set up Region Average Directory

# Indicate Path to Area-Average Files
dpath_aavg      = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/05_SMIO/data/region_average/"
regname         = "SPGNE"
bbsel           = [-40,-15,52,62]

# regname         = "NNAT"
# bbsel           = [-80,0,20,60]

# Set up Output Path
locfn,locstring = proc.make_locstring_bbox(bbsel,lon360=True)
bbfn            = "%s_%s" % (regname,locfn)
outpath         = "%s%s/" % (dpath_aavg,bbfn)
proc.makedir(outpath)


#%% User Edits


smoutput = True # Set to True for stochastic model output

# outformat = "{outpath}{expname}_{vname}_{ystart:04d}_{yend:04d}_{procstring}.nc"
# outname    = outformat.format(outpath=outpath,
#                               expname=expname,
#                               vname=vname,
#                               ystart=ystart,
#                               yend=yend,
#                               procstring=procstring)#"CESM2_POM3_SHF_0200"

outformat = "%s%s_%s_%04d_%04d_%s.nc"

# If smoutput is <True>... ----------------------------------------------------
# Use sm loader and output path to metrics folder
expname     = "SST_ORAS5_avg_GMSST" #"SST_ORAS5_avg_EOF" #"SST_ORAS5_avg_mld003" #"SST_ORAS5_avg" #"SST_ERA5_1979_2024"
vname       = "SST"
concat_dim  = "time"

if smoutput:
    sm_output_path = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/sm_experiments/"
    outpath        = "%s%s/Metrics/" % (sm_output_path,expname) # Save into experiment directory
    # Output path to "Metrics" Folder of stochastic model output...
    outname = "%sArea_Avg_%s.nc" % (outpath,bbfn)
    
else:
    # Otherwise specify an ncsearch
    
    # CESM2_POM3_PiControl_SHF_0100_0500_raw.nc ~~~~~~~~~~~~~~~~
    concat_dim = "time"
    vname      = "TS"
    expname    = "CESM2_POM3_PiControl"
    ystart     = 000
    yend       = 500
    procstring = "raw"
    searchpath = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/POM/"
    ncsearch   = searchpath + "*TS.nc"
    
    #CESM2_FCM_PiControl_SHF_ ~~~~~~~~~~~~~~~~
    concat_dim = "time"
    vname      = "SHF"
    expname    = "CESM2_FCM_PiControl"
    ystart     = 0000
    yend       = 2000
    procstring = "raw"
    searchpath = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/FCM/ocn/SHF/"
    ncsearch   = searchpath + "*SHF*.nc"
    
    # CESM2_FCM_PiControl_SST_ ~~~~~~~~~~~~~~~~
    concat_dim = "time"
    vname      = "SST"
    expname    = "CESM2_FCM_PiControl"
    ystart     = 0000
    yend       = 2000
    procstring = "raw"
    searchpath = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/FCM/ocn/SST/"
    ncsearch   = searchpath + "*SST*.nc"
    
    # SOM TS
    concat_dim = "time"
    vname      = "LHFLX"#"FLNS"#"TS"
    expname    = "CESM2_SOM_PiControl"
    ystart     = 0
    yend       = 360
    procstring = "raw"
    searchpath = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/SOM/"
    ncsearch   = searchpath + "*%s*.nc" % vname
    
    outname    = outformat % (outpath,expname,vname,ystart,yend,procstring)

print("Region is        : %s" % bbfn)
print("Output Path is   : %s" % outpath)
print("Output file is   : %s" % outname)

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
    
    if 'lon' in ds.coords:
        ds     = proc.lon360to180_xr(ds)
        dsreg  = proc.sel_region_xr(ds,bbsel)[vname].load()
        dsavg  = proc.area_avg_cosweight(dsreg)
        
    elif "TLONG" in ds.coords:
        dsreg = proc.sel_region_xr_cv(ds,bbsel).load()
        dsavg = proc.area_avg_cosweight_cv(dsreg,vname)
        
        # Sanity Check
        #plt.scatter(dsreg.TLONG,dsreg.TLAT,c=dsreg[vname].isel(time=0)),plt.show()
    
    aavgs.append(dsavg)

# ds     = xr.open_dataset(nclist[0])
# dsreg  = proc.sel_region_xr(ds,bbsel).load()

#%% Do further processing

# if smoutput:
#     aavgs   = xr.concat(aavgs,dim='ens')
#     da_in   = aavgs.SST
#     da_out  = reshape_ens2year(da_in)
    
if len(nclist) > 1:
    if concat_dim == 'ens':
        aavgs   = xr.concat(aavgs,dim='ens')
        da_in   = aavgs[vname]
        da_out  = reshape_ens2year(da_in)
    elif concat_dim == "time":
        da_out  =  xr.concat(aavgs,dim='time')
else:
    
    da_out = aavgs[0]
    
edict = proc.make_encoding_dict(da_out)
da_out.to_netcdf(outname,encoding=edict)
print("File saved to %s" % outname)


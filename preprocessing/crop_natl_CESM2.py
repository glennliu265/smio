#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Crop to just North Atlantic

copied form area_average_output
copied regridding section from regrid_pop_cesm

Created on Wed May 14 15:48:45 2025

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

import xesmf as xe

#%% stormtrack modules

amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
scmpath = "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/" # scm module

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc#,viz
#import scm
#import amv.loaders as dl

#%% User Edits: Set up Region Average Directory

outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/05_SMIO/data/NAtl/"

bbcrop  = [-100,20,-10,90]
# # Indicate Path to Area-Average Files
# dpath_aavg      = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/05_SMIO/data/region_average/"
# regname         = "SPGNE"
# bbsel           = [-40,-15,52,62]

# # regname         = "NNAT"
# # bbsel           = [-80,0,20,60]

# # Set up Output Path
# locfn,locstring = proc.make_locstring_bbox(bbsel,lon360=True)
# bbfn            = "%s_%s" % (regname,locfn)
# outpath         = "%s%s/" % (dpath_aavg,bbfn)
# proc.makedir(outpath)


#%% User Edits


smoutput = False # Set to True for stochastic model output

# outformat = "{outpath}{expname}_{vname}_{ystart:04d}_{yend:04d}_{procstring}.nc"
# outname    = outformat.format(outpath=outpath,
#                               expname=expname,
#                               vname=vname,
#                               ystart=ystart,
#                               yend=yend,
#                               procstring=procstring)#"CESM2_POM3_SHF_0200"

outformat = "%s%s_%s_%04d_%04d_%s.nc"

# Otherwise specify an ncsearch

# CESM2_POM3_PiControl_TS_0100_0500_raw.nc ~~~~~~~~~~~~~~~~
# vname      = "TS"
# expname    = "CESM2_POM3_PiControl"
# ystart     = 000
# yend       = 500
# procstring = "raw"
# searchpath = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/POM/"
# ncsearch   = searchpath + "*TS.nc"

# CESM2_FCM_PiControl_TS_ ~~~~~~~~~~~~~~~~
vname      = "TS"
expname    = "CESM2_FCM_PiControl"
ystart     = 000
yend       = 2000
procstring = "raw"
searchpath = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/FCM/atm/TS/"
ncsearch   = searchpath + "*TS*.nc"

# # SOM TS
vname      = "TS"
expname    = "CESM2_SOM_PiControl"
ystart     = 0
yend       = 360
procstring = "raw"
searchpath = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/SOM/"
ncsearch   = searchpath + "*%s*.nc" % vname

## CESM2_POM_SHF_ ~~~~~~~~~~~~~~~~
vname      = "SHF"
expname    = "CESM2_POM3_PiControl"
ystart     = 000
yend       = 2000
procstring = "raw"
searchpath = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/POM/"
ncsearch   = searchpath + "*SHF*.nc"

# CESM2_FCM_PiControl_SHF_ ~~~~~~~~~~~~~~~~
vname      = "SHF"
expname    = "CESM2_FCM_PiControl"
ystart     = 000
yend       = 2000
procstring = "raw"
searchpath = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/FCM/ocn/SHF/"
ncsearch   = searchpath + "*SHF*.nc"

# # SOM Fluxes
vnames      = ["FLNS","FSNS","SHFLX","LHFLX"]#"TS"
#for vname in vnames:
expname    = "CESM2_SOM_PiControl"
ystart     = 0
yend       = 360
procstring = "raw"
searchpath = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/SOM/"
ncsearch   = searchpath + "*%s*.nc" % vname

# # SOM ICEFRAC
vname      = "ICEFRAC"
expname    = "CESM2_SOM_PiControl"
ystart     = 0
yend       = 360
procstring = "raw"
searchpath = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/SOM/"
ncsearch   = searchpath + "*%s*.nc" % vname


## CESM2_POM_ICE_ ~~~~~~~~~~~~~~~~
vname      = "IFRAC"
expname    = "CESM2_POM3_PiControl"
ystart     = 000
yend       = 100
procstring = "raw"
searchpath = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/POM/"
ncsearch   = searchpath + "*%s*.nc" % vname

# CESM2_FCM_PiControl_SHF_ ~~~~~~~~~~~~~~~~
# concat_dim = "time"
# vname      = "SHF"
# expname    = "CESM2_FCM_PiControl"
# ystart     = 0000
# yend       = 2000
# procstring = "raw"
# searchpath = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/FCM/ocn/SHF/"
# ncsearch   = searchpath + "*SHF*.nc"

# # CESM2_FCM_PiControl_SST_ ~~~~~~~~~~~~~~~~
# concat_dim = "time"
# vname      = "SST"
# expname    = "CESM2_FCM_PiControl"
# ystart     = 0000
# yend       = 2000
# procstring = "raw"
# searchpath = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/FCM/ocn/SST/"
# ncsearch   = searchpath + "*SST*.nc"

# # SOM TS
# concat_dim = "time"
# vname      = "LHFLX"#"FLNS"#"TS"
# expname    = "CESM2_SOM_PiControl"
# ystart     = 0
# yend       = 360
# procstring = "raw"
# searchpath = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/SOM/"
# ncsearch   = searchpath + "*%s*.nc" % vname

outname    = outformat % (outpath,expname,vname,ystart,yend,procstring)



print("Output Path is   : %s" % outpath)
print("Output file is   : %s" % outname)
print("Cropping to North Atlantic Region")

#%% Load Reference Lat.Lon for regridding


method         = 'bilinear'


# Indicate Reference
refpath        = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/proc/"
refnc          = refpath + "FCM_1600_2000_TS_clim.nc"

# Load reference dataset
dsref          = xr.open_dataset(refnc).load()

#% Make the Grid fron the reference file
lat            = dsref.lat
lon            = dsref.lon
xx,yy          = np.meshgrid(lon,lat)
newgrid        = xr.Dataset({'lat':lat,'lon':lon})


#%% Query the files

nclist = glob.glob(ncsearch)
nclist.sort()
    
print("Found %i files" % (len(nclist)))
print(nclist)


#%% Loop for each file, load and append to list

dsregs  = []
nfiles = len(nclist)
for ff in tqdm.tqdm(range(nfiles)):
    
    ncname = nclist[ff]
    ds     = xr.open_dataset(ncname)
    
    if 'lon' in ds.coords:
        ds     = proc.lon360to180_xr(ds)
        dsreg  = proc.sel_region_xr(ds,bbcrop)[vname].load()
        
    elif "TLONG" in ds.coords:
        # Time to Regrid?
        
        ds          = ds.rename({"TLONG": "lon", "TLAT": "lat"})
        ds          = ds.squeeze().load()
        
        # Set up Regridder
        oldgrid     = ds
        regridder   = xe.Regridder(oldgrid,newgrid,method,periodic=True)
        
        
        #
        if type(ds) == xr.core.dataset.Dataset:
            ds = ds[vname]

        # Regrid
        st     = time.time()
        daproc = regridder(ds) # Need to input dataarray
        print("Regridded in %.2fs" % (time.time()-st))
        
        daproc = proc.lon360to180_xr(daproc)
        dsreg = proc.sel_region_xr(daproc,bbcrop).load()
        
    dsregs.append(dsreg)
    

#%% Do some preprocessing

dsregs = [proc.fix_febstart(ds) for ds in dsregs]
dsregs = [proc.format_ds(ds) for ds in dsregs]


# Concatenate Along Time
if len(nclist) > 1:
    da_out  =  xr.concat(dsregs,dim='time')
else:
    da_out  = dsregs[0]

edict   = proc.make_encoding_dict(da_out)
da_out.to_netcdf(outname,encoding=edict)
print("File saved to %s" % outname)

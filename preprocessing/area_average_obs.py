#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate Average SST for several observational products

Created on Thu May  8 09:15:56 2025

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

# local device (currently set to run on Astraeus, customize later)
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import cvd_utils as cvd

#%% indicate cropping region and check for a folder

# Indicate Path to Area-Average Files
dpath_aavg      = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/region_average/"
regname         = "SPGNE"
bbsel           = [-40,-15,52,62]

# regname         = "NNAT"
# bbsel           = [-80,0,20,60]


# Set up Output Path
locfn,locstring = proc.make_locstring_bbox(bbsel,lon360=True)
bbfn            = "%s_%s" % (regname,locfn)
outpath         = "%s%s/" % (dpath_aavg,bbfn)
proc.makedir(outpath)
print("Region is        : %s" % bbfn)
print("Output Path is   : %s" % outpath)

# String Format is <bbfn>/<expname>_<vname>_<ystart>_<yend>_<procstring>.nc

# ============================================
#%% Preprocess for HadISST (already detrended)
# ============================================

# I think this was preprocessed in a much earlier script...
# Need to find source and include newer years.

expname     = "HadISST"
vname       = "SST"
ystart      = 1920
yend        = 2017
procstring  = "detrend_deseason"
vname_out       = 'sst'
outname     = "%s%s_%s_%s_%s_%s.nc" % (outpath,expname,vname_out,ystart,yend,procstring)
print(outname)


dpathraw    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/"
rawnc       = "HadISST_Detrended_Deanomalized_1920_2018.nc"

st          = time.time()
dsfull      = xr.open_dataset(dpathraw + rawnc) # Lon180 x Lat x Time
dsreg       = proc.sel_region_xr(dsfull,bbsel).load()
proc.printtime(st,print_str='Loaded')
aavg        = proc.area_avg_cosweight(dsreg)
try:
    aavg            = aavg.rename(vname_out)
except:
    aavg = aavg.rename({vname:vname_out})
edict       = proc.make_encoding_dict(aavg)
aavg.to_netcdf(outname,encoding=edict)
print("Loaded in %.2fs" % (time.time()-st))

# Make a debugging plot
dsreg[vname].isel(time=0).T.plot()



# ============================================
#%% Do for ERA5, Qnet, SST
# ============================================

expname     = "ERA5"
vname       = "sst"
ystart      = 1979
yend        = 2024
procstring  = "raw_IceMask5"
outname     = "%s%s_%s_%s_%s_%s.nc" % (outpath,expname,vname,ystart,yend,procstring)
print(outname)

#
dpathraw    = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
rawnc       = "ERA5_%s_NAtl_1979to2024.nc" % vname
st          = time.time()
dsfull      = xr.open_dataset(dpathraw + rawnc) # Lon180 x Lat x Time
dsreg       = proc.sel_region_xr(dsfull,bbsel).load()[vname]

# Load Ice Mask and Apply
icemask     = dl.load_mask(expname="ERA5").mask
dsreg       = dsreg * icemask
proc.printtime(st,print_str='Loaded')

# Take Area Average and Save
aavg        = proc.area_avg_cosweight(dsreg)
aavg        = aavg.rename(vname)
edict       = proc.make_encoding_dict(aavg)
aavg.to_netcdf(outname,encoding=edict)
print("Loaded in %.2fs" % (time.time()-st))


# =============================================
#%% Do for OISST SST
# =============================================

expname         = "OISST"
vname           = "sst"
ystart          = 1982
yend            = 2020
procstring      = "raw_IceMaskMax5"
outname         = "%s%s_%s_%s_%s_%s.nc" % (outpath,expname,vname,ystart,yend,procstring)
print(outname)

# Note: Copied from hfcalc/reanalysis/proc folder on 2025.05.13
dpathraw        = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
rawnc           = "OISST_sst_1982_2020_NATL.nc"
st              = time.time()
dsfull          = xr.open_dataset(dpathraw + rawnc) # Lon180 x Lat x Time
dsreg           = proc.sel_region_xr(dsfull,bbsel).load()[vname]

# Load Ice Mask and Apply
dpath_ice       = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/reanalysis/proc/NATL_proc_obs/"
nc_masks        = dpath_ice + "OISST_ice_masks_1981_2020.nc"
ds_masks        = xr.open_dataset(nc_masks).load().mask_mon
icemask         = xr.where(ds_masks==0,np.nan,1)
icemask,dsfull  = proc.resize_ds([icemask,dsfull])
icemask         = xr.where(np.isnan(dsfull.sst.isel(time=0)),np.nan,icemask)
dsreg           = dsreg * icemask
proc.printtime(st,print_str='Loaded')

# Take Area Average and Save
aavg            = proc.area_avg_cosweight(dsreg)
aavg            = aavg.rename(vname)
edict           = proc.make_encoding_dict(aavg)
aavg.to_netcdf(outname,encoding=edict)
print("Loaded in %.2fs" % (time.time()-st))

# ===========================================
#%% Preprocess for... EN4?
# ===========================================
# Note: Ice Mask is not Applied!

expname         = "EN4"
vname           = "temperature"
ystart          = 1900
yend            = 2021
procstring      = "raw"
outname         = "%s%s_%s_%s_%s_%s.nc" % (outpath,expname,vname_out,ystart,yend,procstring)
print(outname)
vname_out       = 'sst'

dpathraw        = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/EN4/proc/"
rawnc           = "EN4_concatenate_1900to2021_lon-80to00_lat00to65.nc"
st              = time.time()
dsfull          = xr.open_dataset(dpathraw + rawnc) # Lon180 x Lat x Time
dsreg           = proc.sel_region_xr(dsfull,bbsel).load()[vname]
proc.printtime(st,print_str='Loaded')

# Take Area Average and Save
aavg            = proc.area_avg_cosweight(dsreg)
try:
    aavg            = aavg.rename(vname_out)
except:
    aavg = aavg.rename({vname:vname_out})
edict           = proc.make_encoding_dict(aavg)
aavg.to_netcdf(outname,encoding=edict)
print("Loaded in %.2fs" % (time.time()-st))

# %% ERSST 5
#

expname         = "ERSST5"
vname           = "sst"
ystart          = 1854
yend            = 2017
procstring      = "raw"
outname         = "%s%s_%s_%s_%s_%s.nc" % (outpath,expname,vname,ystart,yend,procstring)
print(outname)
vname_out       = 'sst'

rawnc           = "ERSST5.nc"
dpathraw        = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/"
st              = time.time()
dsfull          = xr.open_dataset(dpathraw + rawnc) # Lon180 x Lat x Time
dsreg           = proc.sel_region_xr(dsfull,bbsel).load()[vname]
proc.printtime(st,print_str='Loaded')

# Select time
dsreg = dsreg.sel(time=slice(None,'2017-12-31'))

# Take Area Average and Save
aavg            = proc.area_avg_cosweight(dsreg)
try:
    aavg            = aavg.rename(vname_out)
except:
    aavg = aavg.rename({vname:vname_out})
edict           = proc.make_encoding_dict(aavg)
aavg.to_netcdf(outname,encoding=edict)
print("Loaded in %.2fs" % (time.time()-st))

# ======================
#%% Preprocess for DCENT
# ======================

# Indicate Output
expname         = "DCENT_EnsMean"
vname           = "sst"
ystart          = 1850
yend            = 2023
procstring      = "raw"
outname         = "%s%s_%s_%s_%s_%s.nc" % (outpath,expname,vname,ystart,yend,procstring)
print(outname)
vname_out       = 'sst'

# Open the File and crop to region
rawnc           = "DCENT_ensemble_1850_2023_ensemble_mean.nc"
dpathraw        = "/Users/gliu/Globus_File_Transfer/Reanalysis/DCENT/"
st              = time.time()
dsfull          = xr.open_dataset(dpathraw + rawnc) # Lon180 x Lat x Time

# Do some preprocessing (Flip Longitude)
dsfull          = proc.format_ds(dsfull[vname])

dsreg           = proc.sel_region_xr(dsfull,bbsel).load()#[vname]
proc.printtime(st,print_str='Loaded')

# Take Area Average and Save
aavg            = proc.area_avg_cosweight(dsreg)
try:
    aavg            = aavg.rename(vname_out)
except:
    aavg = aavg.rename({vname:vname_out})
edict           = proc.make_encoding_dict(aavg)
aavg.to_netcdf(outname,encoding=edict)
print("Loaded in %.2fs" % (time.time()-st))


# =================
# %% COBE2 
# =================

# Open the File and crop to region
st              = time.time()
rawnc           = "sst.mon.mean.nc"
dpathraw        = "/Users/gliu/Globus_File_Transfer/Reanalysis/COBE2/"
dsfull          = xr.open_dataset(dpathraw + rawnc) # Lon180 x Lat x Time

# Indicate Output Settings
expname         = "COBE2"
vname           = "sst"
ystart          = 1850
yend            = 2024
procstring      = "raw"
outname         = "%s%s_%s_%s_%s_%s.nc" % (outpath,expname,vname,ystart,yend,procstring)
print(outname)
vname_out       = 'sst'

# Slice to Time, Flip Longitude to 180
dsfull          = proc.format_ds(dsfull)
tstart          = '%04i-01-01' % ystart
tend            = "%04i-12-31" % yend
dsfull          = dsfull.sel(time=slice(tstart,tend))
dsfull          = dsfull[vname]

# Crop to Region
dsreg           = proc.sel_region_xr(dsfull,bbsel).load()
proc.printtime(st,print_str='Loaded')

# Take Area Average and Save
aavg            = proc.area_avg_cosweight(dsreg)
try:
    aavg        = aavg.rename(vname_out)
except:
    aavg = aavg.rename({vname:vname_out})
edict           = proc.make_encoding_dict(aavg)
aavg.to_netcdf(outname,encoding=edict)
print("Loaded in %.2fs" % (time.time()-st))

# =================
# %% Glorys12V1 (Note this was for salt...)
# =================

# expname         = "glorys12v1"
# vname           = "temperature"
# ystart          = 1993
# yend            = 2019
# procstring      = "raw"
# outname         = "%s%s_%s_%s_%s_%s.nc" % (outpath,expname,vname,ystart,yend,procstring)
# print(outname)
# vname_out       = 'sst'


# dpathraw   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/glorys12v1/"
# rawnc      = "glorys12v1_so_NAtl_1993_2019_merge.nc"
# st              = time.time()
# dsfull          = xr.open_dataset(dpathraw + rawnc) # Lon180 x Lat x Time
# dsreg           = proc.sel_region_xr(dsfull,bbsel).load()[vname]
# proc.printtime(st,print_str='Loaded')



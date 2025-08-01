#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Preprocess CESM2 Hierarchy 
(Crop to NATL
 concatenate by time
 fix February start
 regrid where necessary)

Copied upper section of compute_enso_index

Created on Wed Jul  2 00:29:12 2025

@author: gliu

"""


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import time
import sys
import cartopy.crs as ccrs
import glob
import pandas as pd
import xesmf as xe

#%% Import modules
stormtrack = 1
if stormtrack:
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
    
    # Path to the processed dataset (qnet and ts fields, full, time x lat x lon)
    #datpath =  "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/hfdamping_RCP85/01_PREPROC/"
    datpath =  "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/output/anom/"
    figpath =  "/home/glliu/02_Figures/01_WeeklyMeetings/20240621/"
    
else:
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")

    # Path to the processed dataset (qnet and ts fields, full, time x lat x lon)
    datpath =  "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/reanalysis/proc/"
    figpath =  "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/02_Figures/20220511/"

from amv import proc,viz
import scm
import amv.proc as hf # Update hf with actual hfutils script, most relevant functions

#%% ENSO Calculation and Cropping Options

# Select time crop (prior to preprocessing)
# croptime          = True # Cut the time prior to detrending, EOF, etc
# tstart            = '1979-01-01' #'1920-01-01'#'0001-01-01' # "2006-01-01" # 
# tend              = '2024-12-31'#'2021-12-31' #'2005-12-31'#'2000-02-01' # "2101-01-01" # 
# timestr           = "%sto%s" % (tstart[:4],tend[:4])

# ENSO Parameters
bbox_natl         = [-90, 20, 0, 90] # ENSO Bounding Box

# Toggles and Options
overwrite        = True # Set to True to overwrite existing output...
save_netcdf      = True # Set true to save netcdf version, false to save npz
debug            = True # Debug toggle

#%% Dataset option (load full TS variable in [time x lat x lon360])
# Example provided below here is for CESM1


# Output Path (Checks for an "enso" folder)
outpath             = ""#"/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/output/"

# CESM2 PiControl FOM (TS)
dataset_name        = "CESM2_FOM"
datpath             = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/FCM/atm/TS/"
vname               = "TS"
lonname             = "lon"
latname             = "lat"
timename            = "time"
concat_dim          = "time"
keepvars            = [timename,latname,lonname,vname]
ensnum              = 1 # Irrelevant for now, need to add ensemble support...
detrend             = 1 # 1 to remove linear trend 
outpath             = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/output/enso/"
#yr_range            = "1984to2007" # Other topin is 1979to2021
croptime            = True # Cut the time prior to detrending, EOF, etc
tstart              = '0200-01-01' #'1920-01-01'#'0001-01-01' # "2006-01-01" # 
tend                = '2000-12-31'#'2021-12-31' #'2005-12-31'#'2000-02-01' # "2101-01-01" # 
timestr             = "%sto%s" % (tstart[:4],tend[:4])
yr_range            = timestr
searchstr           = datpath + "b.e21.B1850.f09_g17.CMIP6-piControl.001.cam.h0.TS.*.nc" 

# # CESM2 PiControl SOM
# dataset_name        = "CESM2_SOM"
# datpath             = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/SOM/"
# vname               = "TS"
# lonname             = "lon"
# latname             = "lat"
# timename            = "time"
# concat_dim          = "time"
# keepvars            = [timename,latname,lonname,vname]
# ensnum              = 1 # Irrelevant for now, need to add ensemble support...
# detrend             = 1 # 1 to remove linear trend 
# outpath             = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/output/enso/"
# #yr_range            = "1984to2007" # Other topin is 1979to2021
# croptime            = True # Cut the time prior to detrending, EOF, etc
# tstart              = '0060-01-01' #'1920-01-01'#'0001-01-01' # "2006-01-01" # 
# tend                = '0360-12-31'#'2021-12-31' #'2005-12-31'#'2000-02-01' # "2101-01-01" # 
# timestr             = "%sto%s" % (tstart[:4],tend[:4])
# yr_range            = timestr

# # CESM2 PiControl MCOM
# dataset_name        = "CESM2_MCOM"
# datpath             = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/POM/"
# vname               = "TS"
# lonname             = "lon"
# latname             = "lat"
# timename            = "time"
# concat_dim          = "time"
# keepvars            = [timename,latname,lonname,vname]
# ensnum              = 1 # Irrelevant for now, need to add ensemble support...
# detrend             = 1 # 1 to remove linear trend 
# outpath             = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/output/enso/"
# #yr_range            = "1984to2007" # Other topin is 1979to2021
# croptime            = True # Cut the time prior to detrending, EOF, etc
# tstart              = '0100-01-01' 
# tend                = '0500-12-31'
# timestr             = "%sto%s" % (tstart[:4],tend[:4])
# yr_range            = timestr



# CESM2 PiControl FOM (SHF)
dataset_name        = "CESM2_FOM"
datpath             = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/FCM/ocn/SHF/"
vname               = "SHF"
lonname             = "lon"
latname             = "lat"
timename            = "time"
concat_dim          = "time"
keepvars            = [timename,latname,lonname,vname,"TLONG","TLAT"]
ensnum              = 1 # Irrelevant for now, need to add ensemble support...
detrend             = 1 # 1 to remove linear trend 
#yr_range            = "1984to2007" # Other topin is 1979to2021
croptime            = True # Cut the time prior to detrending, EOF, etc
tstart              = '0200-01-01' #'1920-01-01'#'0001-01-01' # "2006-01-01" # 
tend                = '2000-12-31'#'2021-12-31' #'2005-12-31'#'2000-02-01' # "2101-01-01" # 
timestr             = "%sto%s" % (tstart[:4],tend[:4])
yr_range            = timestr
searchstr           = datpath + "b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.SHF.*.nc"


# CESM2 PiControl MCOM (SHF)
dataset_name        = "CESM2_MCOM"
datpath             = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/POM/"
vname               = "SHF"
lonname             = "lon"
latname             = "lat"
timename            = "time"
concat_dim          = "time"
keepvars            = [timename,latname,lonname,vname,"TLONG","TLAT"]
ensnum              = 1 # Irrelevant for now, need to add ensemble support...
detrend             = 1 # 1 to remove linear trend 
croptime            = True # Cut the time prior to detrending, EOF, etc
tstart              = '0100-01-01' 
tend                = '0500-12-31'
timestr             = "%sto%s" % (tstart[:4],tend[:4])
yr_range            = timestr
searchstr           = datpath + "b.e21.B1850.f09_g17.1dpop2-gterm.005.*.SHF.nc"



# CESM2 PiControl SOM (Fluxes)
dataset_name        = "CESM2_SOM"
datpath             = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/SOM/"
vnames               = ["FLNS","FSNS","SHFLX","LHFLX"]#"TS"
lonname             = "lon"
latname             = "lat"
timename            = "time"
concat_dim          = "time"
ensnum              = 1 # Irrelevant for now, need to add ensemble support...
detrend             = 1 # 1 to remove linear trend 
croptime            = True # Cut the time prior to detrending, EOF, etc
tstart              = '0060-01-01' #'1920-01-01'#'0001-01-01' # "2006-01-01" # 
tend                = '0360-12-31'#'2021-12-31' #'2005-12-31'#'2000-02-01' # "2101-01-01" # 
timestr             = "%sto%s" % (tstart[:4],tend[:4])
yr_range            = timestr
# for vname in vnames: # Loop only for SOM
#     keepvars            = [timename,latname,lonname,vname]
#     searchstr           = datpath + "e.e21.E1850.f09_g17.CMIP6-piControl.001_branch2.cam.h0.%s.*.nc" % vname
    



# CESM2 PiControl SOM (IFRAC)
dataset_name        = "CESM2_SOM"
datpath             = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/SOM/"
vname               = "ICEFRAC"
keepvars            = [timename,latname,lonname,vname]
lonname             = "lon"
latname             = "lat"
timename            = "time"
concat_dim          = "time"
ensnum              = 1 # Irrelevant for now, need to add ensemble support...
detrend             = 1 # 1 to remove linear trend 
croptime            = True # Cut the time prior to detrending, EOF, etc
tstart              = '0060-01-01' #'1920-01-01'#'0001-01-01' # "2006-01-01" # 
tend                = '0360-12-31'#'2021-12-31' #'2005-12-31'#'2000-02-01' # "2101-01-01" # 
timestr             = "%sto%s" % (tstart[:4],tend[:4])
yr_range            = timestr
searchstr           = datpath + "e.e21.E1850.f09_g17.CMIP6-piControl.001_branch2.cam.h0.ICEFRAC.*.nc"



# CESM2 PiControl MCOM (IFRAC)
dataset_name        = "CESM2_MCOM"
datpath             = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/POM/"
vname               = "IFRAC"
lonname             = "lon"
latname             = "lat"
timename            = "time"
concat_dim          = "time"
keepvars            = [timename,latname,lonname,vname,"TLONG","TLAT"]
ensnum              = 1 # Irrelevant for now, need to add ensemble support...
detrend             = 1 # 1 to remove linear trend 
croptime            = True # Cut the time prior to detrending, EOF, etc
tstart              = '0100-01-01' 
tend                = '0500-12-31'
timestr             = "%sto%s" % (tstart[:4],tend[:4])
yr_range            = timestr
searchstr           = datpath + "b.e21.B1850.f09_g17.1dpop2-gterm.005.*.IFRAC.nc"




# Mask Information (first run a maskmaker script/section such as that in preproc_CESM2_PiControl.py)
maskpath            = None
maskname            = None


#%% Find File and Load Variable

# 1A. Load Variable ----------

# # Create filename/list and load 
# if dataset_name == "CESM2_FOM":
    
# elif dataset_name == "CESM2_SOM":
#     searchstr = datpath + "e.e21.E1850.f09_g17.CMIP6-piControl.001_branch2.cam.h0.TS.*.nc"
# elif dataset_name == "CESM2_MCOM":
#     searchstr = datpath + "b.e21.B1850.f09_g17.1dpop2-gterm.005.*.TS.nc"
# else:
#     print("Enter one of: [CESM2_FOM,CESM2_SOM,CESM2_MCOM]")
    
nclist    = glob.glob(searchstr)
nclist.sort()
print("Found %i files for %s" % (len(nclist),vname))

# Drop Unnecessary variables
if concat_dim is None or len(nclist) == 1: # Assume no concantenation is needed
    ds_all    = xr.open_dataset(nclist[0])
else:
    print("Concatenating files by dim <%s>" % (concat_dim))
    ds_all    = xr.open_mfdataset(nclist,concat_dim=concat_dim,combine='nested')
ds_all    = hf.ds_dropvars(ds_all,keepvars)
try:
    ds_all    = hf.fix_febstart(ds_all)
except:
    print("Warning, Time is not in datetime... converting")
    print("First timestep (pre conversion) is %s" % (ds_all[timename][0]))
    timeconv = pd.to_datetime(ds_all.time.data)
    ds_all[timename] = timeconv
    print("First timestep (post conversion) is %s" % (ds_all[timename][0]))
    
    #ds_all['time2'] = pd.to_datetime(ds_all.time.data)
    print("Warning: February Start Fix was not implemented")
    #print("First timestep is %s" % (ds_all[timename][0]))

#%% Flip Longitude

if "TLONG" in ds_all:
    
    # Prepare for regridding
    bbox_natl_360 = [-90,20,0,90]
    dsreg         = proc.sel_region_xr_cv(ds_all,bbox_natl,debug=False)
    plt.scatter(dsreg.TLONG,dsreg.TLAT,c=dsreg[vname].isel(time=0)),plt.show()
    
    # Load File
    dsreg         = dsreg.load()
    
    # Also Load Dummy File with correct coordinates
    testnc        = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/proc/NAtl/CESM2_SOM_TS_NAtl_0060to0360.nc"
    testds        = xr.open_dataset(testnc).isel(time=0).load()
    
    ds_out         = xr.Dataset({'lat': (['lat'],testds.lat.data), 'lon': (['lon'], testds.lon.data) })
    
    
    # Rename and Prep
    dsreg_rename  = dsreg.rename({"TLONG": "lon", "TLAT": "lat",})
    
    # Regrid
    method = "bilinear"
    regridder      = xe.Regridder(dsreg_rename, ds_out, method,periodic=True)
    ds_regridded = regridder(dsreg_rename)
    
    dsreg = ds_regridded
    
    
else:
    

    
    # Select Region
    ds_format = proc.format_ds(ds_all)
    dsreg     = proc.sel_region_xr(ds_format,bbox_natl)
    
    
    # Select Time
    dsreg_times = dsreg.sel(time=slice(tstart,tend))
    
    # Load
    st    = time.time()
    dsreg = dsreg_times.load()
    print("Loaded in %.2fs" % (time.time()-st))


#%% Save output

outpath = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/proc/NAtl/"
outname = "%s%s_%s_NAtl_%s.nc" % (outpath,dataset_name,vname,timestr)
edict   = {vname : {'zlib':True}}
dsreg.to_netcdf(outname,encoding=edict)




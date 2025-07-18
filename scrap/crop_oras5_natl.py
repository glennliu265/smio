#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Crop ORAS5 to the North Atlantic

Created on Wed Jun 25 16:38:00 2025

@author: gliu

"""

import xarray as xr
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import tqdm

#%%
device = "stormtrack"

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

# Settings
#vname      = 'votemper'
#vname_out  = "TEMP"


vname       = "somxl030"
vname_out   = "mld"

opastr  = "CDS"#'opa4' #'opa1'
ystart  =1979#2019#1979
yend    =2024# 2024#2018
years   = np.arange(ystart,yend+1,1)
nyrs    = len(years)


# Find and loop for a single file (1 year)
#ncsearch     = "votemper_ORAS5_1m_%04i%02i_grid_T_02.nc" #"votemper_ORAS5_1m_201801_grid_T_02.nc"
if device == 'Astraeus':
    dpath        = "/Users/gliu/Globus_File_Transfer/Reanalysis/ORAS5/%s/" % opastr
    outpath      = "/Users/gliu/Globus_File_Transfer/Reanalysis/ORAS5/proc/"
elif device == "stormtrack":
    dpath        = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/05_SMIO/data/ORAS5/%s/" % opastr
    outpath      = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/05_SMIO/data/ORAS5/proc/"
    

if opastr == "CDS":
    if device == "Astraeus":
        print("Note: Must change device to stormtrack for 2019 to 2024")
    else:
        print("Changing dpath for 2019 to 2024")
        dpath = "/stormtrack/data4/glliu/01_Data/Reanalysis/ORAS5/%s/" % vname
        
    

outname      = "%sORAS5_%s_%s_NAtl_%04i_%04i.nc" % (outpath,opastr,vname_out,ystart,yend)

# %% Crop the data

def crop_oras5_natl(ds,depth=True):
    
    tlat    = ds.nav_lat.data
    tlon    = ds.nav_lon.data
    if depth:
        z_t     = ds.deptht.data
    
    # Rename to CESM1 conventions
    oras5_2_cesm = dict(
        nav_lat="TLAT",
        nav_lon ="TLONG",
        
        x='nlon',
        y='nlat',
        time_counter='time'
        )
    if depth:
        oras5_2_cesm['deptht'] = 'z_t'
    dsformat = ds.rename(oras5_2_cesm)
    
    # Bbox seems to be ok in degrees west
    bbox    = [-80,0,20,65]
    #dpath   = "/Volumes/proj/cmip6/data/ocean_reanalysis/ORAS5/oras5/monthly/oras5/ORCA025/votemper/"
    
    # Select the box
    dsreg   = proc.sel_region_xr_cv(dsformat,bbox,debug=False)
    
    return dsreg

#%% Load and crop to North Atlantic


dsall   = []
for yy in tqdm.tqdm(range(nyrs)): # Loop by Year
    
    yr   = years[yy]
    for im in range(12): # Loop by Mon
        if vname == "votemper":
            depth=True
            if yr >= 2019:
                ncsearch = "%s%s_control_monthly_highres_3D_%04i%02i_OPER_v0.1.nc" % (dpath,vname,yr,im+1)
            else:
                ncsearch = "%s%s_ORAS5_1m_%04i%02i_grid_T_02.nc"  % (dpath,vname,yr,im+1)#"votemper_ORAS5_1m_201801_grid_T_02.nc"
        else:
            depth=False
            if yr >= 2015:
                ncsearch = "%s%s_control_monthly_highres_2D_%04i%02i_OPER_v0.1.nc" % (dpath,vname,yr,im+1)
            else:
                ncsearch = "%s%s_control_monthly_highres_2D_%04i%02i_CONS_v0.1.nc" % (dpath,vname,yr,im+1)
        dsmon        = xr.open_dataset(ncsearch).load()
        dsreg        = crop_oras5_natl(dsmon,depth=depth)
        dsall.append(dsreg)
        dsmon.close()
        
dsall_merge = xr.concat(dsall,dim='time')

#%% Save output (about ~4.83 GB...)

dsall_merge = dsall_merge.rename({vname:vname_out})
edict       = proc.make_encoding_dict(dsall_merge)


dsall_merge.to_netcdf(outname,encoding=edict)

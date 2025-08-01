#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

To compare timeseries in oras5, get a single point

Created on Mon Jul  7 13:06:22 2025

@author: gliu

"""

import sys
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob


#%%
amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
scmpath = "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/" # scm module

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl

#%% Set some information




# OPA2 and OPA3, OPA4
ncpath_type1    = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/05_SMIO/data/ORAS5/opa%i/"
ncsearch_type1  = "votemper_ORAS5_1m_%s_grid_T_02.nc"

# Copernicus Data Information
ncpath_type2    = "/stormtrack/data4/glliu/01_Data/Reanalysis/ORAS5/potential_temperature/"
ncsearch_type2  = "votemper_control_monthly_highres_3D_%s_CONS_v0.1.nc"


ystart          = 1979
yend            = 1981

years = np.arange(ystart,yend+1,1)
nyrs = len(years)
monstr = []
for yy in years:
    for im in range(12):
        monstr.append("%04i%02i" % (yy,im+1))

        

outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/05_SMIO/data/ORAS5/proc/point_data/"
lonf            = -35+330
latf            = 55
zid             = 0

locfn,loctitle  = proc.make_locstring(lonf,latf)


def oras5_2_cesm(ds):
    swapdict = dict(
        nav_lat="TLAT",
        nav_lon ="TLONG",
        deptht="z_t",
        x='nlon',
        y='nlat',
        time_counter='time'
        )
    return ds.rename(swapdict)

#%% Get Point for Type1





ds_byop = []
ntime = len(monstr)
for op in range(2,5):
    ncnames  = [ncpath_type1 % (op) + ncsearch_type1 % (dstr) for dstr in monstr]
    dsall_op = xr.open_mfdataset(ncnames,concat_dim='time_counter',combine='nested')
    
    # Rename Dim
    dsall_op = oras5_2_cesm(dsall_op)
    
    # Select Depth
    dsall_op = dsall_op.isel(z_t=zid)
    
    # Restrict to Point
    dspt = proc.find_tlatlon(dsall_op,lonf,latf,)
    dspt = dspt.load()
    
    outname = "%svotemper_1979to1981__%s_opa%02i.nc" % (outpath,locfn,op)
    dspt.to_netcdf(outname)
    

#%% Repeat Procedure for Type2 (Copernicus Download)



ncnames  = [ncpath_type2 + ncsearch_type2 % (dstr) for dstr in monstr]
dsall_op = xr.open_mfdataset(ncnames,concat_dim='time_counter',combine='nested')      
        
        
# Rename Dim
dsall_op = oras5_2_cesm(dsall_op)

# Select Depth
dsall_op = dsall_op.isel(z_t=zid)

# Restrict to Point
dspt = proc.find_tlatlon(dsall_op,lonf,latf,)
dspt = dspt.load()

outname = "%svotemper_1979to1981__%s_copernicus.nc" % (outpath,locfn)
dspt.to_netcdf(outname)
    

#%% Now move to Astraeus for OPA1 and OPA0


amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import cvd_utils as cvd


#%%

# OPA0, OPA1
ncpath_type1    = "/Users/gliu/Globus_File_Transfer/Reanalysis/ORAS5/opa%i/"
ncsearch_type1  = "votemper_ORAS5_1m_%s_grid_T_02.nc"


outpath         = "/Users/gliu/Globus_File_Transfer/Reanalysis/ORAS5/proc/point_data/"




ds_byop = []
ntime = len(monstr)
for op in range(0,2):
    ncnames  = [ncpath_type1 % (op) + ncsearch_type1 % (dstr) for dstr in monstr]
    dsall_op = xr.open_mfdataset(ncnames,concat_dim='time_counter',combine='nested')
    
    # Rename Dim
    dsall_op = oras5_2_cesm(dsall_op)
    
    # Select Depth
    dsall_op = dsall_op.isel(z_t=zid)
    
    # Restrict to Point
    dspt = proc.find_tlatlon(dsall_op,lonf,latf,)
    dspt = dspt.load()
    
    outname = "%svotemper_1979to1981__%s_opa%02i.nc" % (outpath,locfn,op)
    dspt.to_netcdf(outname)
    
    
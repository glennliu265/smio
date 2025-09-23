#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare Detrending Effects on HFF estimates

   - Uses estimates from hfcalc/main/calc_hff_general_new
   - Uses enso index calculated in smio/scrap/calc_enso_ERA5

Created on Thu Sep 18 14:46:03 2025

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

#%% Import modules

stormtrack = 0
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

#%%

dtnames         = ["1","linearmon","GMSST","GMSSTmon",]
dtnames_long    = ["Linear","Linear (Monthly)","GMSST Removal","GMSST Removal (Monthly)"]
expcols         = ["hotpink","red","midnightblue","cornflowerblue"]
els  = ["solid",'dashed',"solid",'dashed',]
ncformat = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/hff/ERA5_qnet_damping_NAtl_1979to2024_ensorem1_detrend%s.nc"

# Note, renamed "1" in the old form from "hfdamping" to "damping"

ndt = len(dtnames)
dsall = []
for nn in range(ndt):
    ds = xr.open_dataset(ncformat % dtnames[nn]).load()
    dsall.append(ds)
    

bboxspgne = [-40,-15,52,62]
dsallreg  = [proc.sel_region_xr(ds,bboxspgne) for ds in dsall]
aavgs = [proc.area_avg_cosweight(ds) for ds in dsallreg]



#%% Examine differences at a point

lonf = -15
latf = 50

locfn,loctitle = proc.make_locstring(lonf,latf)

mons3 = proc.get_monstr()
fig,ax = viz.init_monplot(1,1)
for nn in range(ndt):
    plotvar = dsall[nn].sel(lon=lonf,lat=latf,method='nearest').qnet_damping.isel(lag=0) * -1
    ax.plot(mons3,plotvar,label=dtnames_long[nn],c=expcols[nn],ls=els[nn])
ax.legend()
ax.set_title("Lag 1 Damping @ %s" % loctitle)

#%%  Plot Area Avg

loctitle = "SPGNE Average"

mons3 = proc.get_monstr()
fig,ax = viz.init_monplot(1,1)
for nn in range(ndt):
    plotvar = aavgs[nn].qnet_damping.isel(lag=0) * -1
    ax.plot(mons3,plotvar,label=dtnames_long[nn],c=expcols[nn],ls=els[nn])
ax.legend()
ax.set_title("Lag 1 Damping @ %s" % loctitle)

#%% Look at some differences in the spatial pattern

im          = 1
ilag        = 0
cints       = np.arange(-5,6.5,.5)

for im in range(12):
    proj = ccrs.PlateCarree()
    
    # Effect of separate detrend by month (Linear)
    diff_lin_mon = dsallreg[0] - dsallreg[1]
    diffname     = "Difference in %s Lag %i Qnet Damping [Linear - Linear (Sep. Months)]" % (mons3[im],ilag+1)
    
    # Effect of separate detrend by month (GMSST)
    diff_lin_mon = dsallreg[2] - dsallreg[3]
    diffname     = "Difference in %s Lag %i Qnet Damping [GMSST - GMSST (Sep. Months)]" % (mons3[im],ilag+1)
    
    # Effect of Linear to GMSST Removal
    diff_lin_mon = dsallreg[0] - dsallreg[2]
    diffname     = "Difference in %s Lag %i Qnet Damping [Linear - GMSST]" % (mons3[im],ilag+1)
    
    # LInear ALl Month to GMSST Sep Month
    diff_lin_mon = dsallreg[0] - dsallreg[3]
    diffname     = "Difference in %s Lag %i Qnet Damping [Linear(All) - GMSST(Sep)]" % (mons3[im],ilag+1)
    
    plotvar = diff_lin_mon.isel(lag=ilag,month=im).qnet_damping
    fig,ax,_ = viz.init_regplot("SPGE")
    #pcm         = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,cmap='cmo.balance',vmin=-2,vmax=2,transform=proj)
    pcm         = ax.contourf(plotvar.lon,plotvar.lat,plotvar,cmap='cmo.balance',levels=cints,transform=proj)
    clbl = ax.clabel(pcm)
    
    viz.add_fontborder(clbl)
    
    cb      = viz.hcbar(pcm)
    
    cb.set_label(diffname,fontsize=24)

#/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/hff/ERA5_qnet_hfdamping_NAtl_1979to2024_ensorem1_detrend1.nc

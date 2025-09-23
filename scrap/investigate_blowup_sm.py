#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Investigate runs where things are blowing up on the stochastic model
2025.09.22 Edition, runs on stormtrack, based on [check_experiment_nan]

Created on Mon Sep 22 13:32:34 2025

@author: gliu

"""


import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import scipy as sp

import matplotlib.patheffects as PathEffects

import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time



#%%

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


# Set up Output Path
locfn,locstring = proc.make_locstring_bbox(bbsel,lon360=True)
bbfn            = "%s_%s" % (regname,locfn)
outpath         = "%s%s/" % (dpath_aavg,bbfn)
proc.makedir(outpath)



#%% User Edits




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
expname     = "SST_ORAS5_avg_GMSSTmon" #"SST_ORAS5_avg_EOF" #"SST_ORAS5_avg_mld003" #"SST_ORAS5_avg" #"SST_ERA5_1979_2024"
vname       = "SST"
concat_dim  = "time"


sm_output_path = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/sm_experiments/"
outpath        = "%s%s/Metrics/" % (sm_output_path,expname) # Save into experiment directory
# Output path to "Metrics" Folder of stochastic model output...
outname = "%sArea_Avg_%s.nc" % (outpath,bbfn)

#%% Load the Output
dsall = dl.load_smoutput(expname,output_path=sm_output_path,)#return_nclist=True)


#%% Check stdev by run

dsstd = dsall.SST.std('time')

# idstart = 0 
# idend   = 1000*12

# dssel = dsall.isel(time=slice(idstart,idend))


#%% Check Blowup by region

levels = [100,1e3,1e4,1e5,1e6,]

fig,axs = plt.subplots(2,5,figsize=(24,6.5))

for nr in tqdm.tqdm( range(10)):
    ax      = axs.flatten()[nr]
    plotvar = dsstd.isel(run=nr)
    
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,vmin=0,vmax=10,cmap='cmo.thermal')
    cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,levels=levels,colors='w',linewidths=.75)
    ax.clabel(cl,fontsize=14)
    ax.set_title("Run %02i" % (nr+1))
    
    ax.plot(-39.75,59.75,marker="x",color="cyan",markersize=26)
    
    
    ax.set_xticks(np.arange(-40,-37.25,0.25))
    ax.set_yticks(np.arange(59,60.25,0.25))
    ax.set_ylim([59,60])
    ax.set_xlim([-40,-38])

cb = viz.hcbar(pcm,ax=axs.flatten())


plt.show()



#%% Look at the timeseries

dsall.sel(lat=59.75,lon=-39.75,method='nearest').isel(run=0).plot()
plt.plot(dsall.sel(lat=59.75,lon=-39.75,method='nearest').isel(run=0).SST)

 dsstd.isel(run=0).sel(lat=59.75,lon=-39.75,method='nearest')
    
    #ds = dsstd.isel(run=nr,ax)
    
#%% Isolate the timeseries


dsprob = dsall.sel(lat=59.75,lon=-39.75,method='nearest').isel(run=0).SST
dTdt   = dsprob.data[1:] - dsprob.data[:(-1)]
dTdt1  = np.hstack([dTdt,[0]])
dTmon  = proc.calc_clim(dTdt1,0)

#%% Save it for analysis on local

dsprob_allrun = dsall.sel(lat=59.75,lon=-39.75,method='nearest')
ncname      = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/05_SMIO/data/smscrap/%s_blowup.nc" % expname
dsprob_allrun.to_netcdf(ncname)
    


#%% Find first nan indices along dimension

def getfirstnan(x):
    idout =  np.where(np.isnan(x))[0] #[0][0]
    if len(idout) == 0: # No NaN Found
        return len(x)
    else:
        return idout[0]

# Use Xr_ufunc
dsid = xr.apply_ufunc(
    getfirstnan,
    dsall.SST,
    input_core_dims=[['time']],
    output_core_dims=[[]],
    vectorize=True,
    )

#%% Make same plot as above but index of where it happens

fig,axs = plt.subplots(2,5,figsize=(24,6.5))

for nr in tqdm.tqdm( range(10)):
    ax      = axs.flatten()[nr]
    plotvar = dsid.isel(run=nr)
    
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,vmin=0,vmax=12000,cmap='cmo.dense')
    #cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,levels=levels,colors='w',linewidths=.75)
    #ax.clabel(cl,fontsize=14)
    ax.set_title("Run %02i" % (nr+1))
    
    ax.plot(-39.75,59.75,marker="x",color="cyan",markersize=26)
    
    
    ax.set_xticks(np.arange(-40,-37.25,0.25))
    ax.set_yticks(np.arange(59,60.25,0.25))
    ax.set_ylim([59,60])
    ax.set_xlim([-40,-38])

cb = viz.hcbar(pcm,ax=axs.flatten())


plt.show()

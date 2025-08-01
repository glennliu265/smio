#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute Fprime for each level of the CESM2 Hierarchy

Based on calc_Fprime_lens.py from /remeergence/calculations/


Requires: 
    Heat Flux Feedback (unprocessed in this case)
    Qnet/SHF
    SST/TS

Created on Mon Jul  7 15:49:52 2025

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

# local device (currently set to run on Astraeus, customize later)
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import cvd_utils as cvd
#%%

# Calculation Options

# (CESM2 FOM) -----------------------------------------------------------------
# SST Information
sstname  = "TS"
sstnc    = "CESM2_FOM_TS_NAtl_0200to2000.nc"
sstpath  = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/NAtl/"

# Flux Information
flxname  = "SHF"
flxnc    = "CESM2_FOM_SHF_NAtl_0200to2000.nc"
flxpath  = sstpath

# Heat Flux Damping Information
hffname  = flxname + "_damping"
hffnc    = "CESM2_FOM_SHF_damping_NAtl_0200to2000_ensorem1_detrend1.nc"
hffpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/hff/"

tstart   = '0200-01-01'
tend     = '2000-12-31'
# -----------------------------------------------------------------------------

# (CESM2 SOM) -----------------------------------------------------------------
# SST Information
sstname  = "TS"
sstnc    = "CESM2_SOM_TS_NAtl_0060to0360.nc"
sstpath  = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/NAtl/"

# Flux Information
flxname  = "SHF"
flxnc    = "CESM2_SOM_SHF_NAtl_0060to0360.nc"
flxpath  = sstpath

# Heat Flux Damping Information
hffname  = flxname + "_damping"
hffnc    = "CESM2_SOM_SHF_damping_NAtl_0060to0360_ensorem1_detrend1.nc"
hffpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/hff/"

tstart   = '0060-01-01'
tend     = '0360-12-31'
# -----------------------------------------------------------------------------

# (CESM2 MCOM) -----------------------------------------------------------------
# SST Information
sstname  = "TS"
sstnc    = "CESM2_MCOM_TS_NAtl_0100to0500.nc"
sstpath  = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/NAtl/"

# Flux Information
flxname  = "SHF"
flxnc    = "CESM2_MCOM_SHF_NAtl_0100to0500.nc"
flxpath  = sstpath

# Heat Flux Damping Information
hffname  = flxname + "_damping"
hffnc    = "CESM2_MCOM_SHF_damping_NAtl_0100to0500_ensorem1_detrend1.nc"
hffpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/hff/"

tstart   = '0100-01-01'
tend     = '0500-12-31'
# -----------------------------------------------------------------------------




#%% Load everything

st    = time.time()
dsflx = xr.open_dataset(flxpath + flxnc).load()
dssst = xr.open_dataset(sstpath + sstnc).load()
dshff = xr.open_dataset(hffpath + hffnc).load()
proc.printtime(st,print_str="Loaded")

#%% Check if Upwards POsitive

bbox_gs    = [-80,-60,20,40]
dsflx      = proc.check_flx(dsflx,flxname=flxname,bbox_gs=bbox_gs)


#%% Preprocess SST and FLX (Remove Seasonal Cycle and Detrend Linear)
# Also Check if Upwards Positive


st    = time.time()

def preproc(ds,tstart,tend):
    dsa = proc.xrdeseason(ds)
    dsadt = proc.xrdetrend(dsa)
    dsadt = dsadt.sel(time=slice(tstart,tend))
    return dsadt

ds_raw  = [dssst[sstname],dsflx[flxname]]
ds_anom = [preproc(ds,tstart,tend) for ds in ds_raw]

ssta,flxa  = ds_anom

proc.printtime(st,print_str="Preproc")



#%% Preprocess Heat Flux Feedback, Check Sign (change so that it is heat flux damping)

ilag   = 0
hffsel = dshff[hffname].isel(lag=ilag) # ilag = 0
hffsel = proc.check_flx(hffsel,bbox_gs=bbox_gs)


#%% Tile and compute

flxarr          = flxa.transpose('time','lat','lon').data[:,None,...]
ntime,nlat,nlon = flxa.shape
sstarr          = ssta.transpose('time','lat','lon').data[:,None,...] # [time x ens x lat x lon]

hffarr          = hffsel.transpose('lat','lon','month').data[None,...] # {ens x lat x lon x time}
nyrs            = int(ntime/12)
hfftile         = np.tile(hffarr,nyrs)
hfftile         = hfftile.transpose(3,0,1,2) # [time x ens x lat x lon]

#%%
#% Calculate F' (positive upwards)
nroll        = 0
Fprime       = flxarr - hfftile*np.roll(sstarr,nroll,axis=0) # Minus is the correct way to go
# Note, it is negative here as we multiply (F'=Qnet + Lambda^aT') by -1
# see obsidian note: [Fprime_sign_clarification]


coords    = dict(time=dssst.time,lat=dssst.lat,lon=dssst.lon)
da_Fprime = xr.DataArray(Fprime.squeeze(),coords=coords,dims=coords,name='Fprime')

outname   = flxpath + flxnc.replace(flxname,"Fprime_upwards")
edict     = proc.make_encoding_dict(da_Fprime)
da_Fprime.to_netcdf(outname,encoding=edict)

#%%
#% Scrap Below (this used to be just below Fprime calculation to visualize the spectra)

klon        = 33
klat        = 30
nsmooth     = 120

# Check Values at a point
arrcol     = ['purple',"red",'cornflowerblue','lightgray']
arrname    = [flxname,sstname,"F'",'wn']
arr_in     = [flxarr,sstarr,Fprime,]
timeseries = [a[:,0,klat,klon] for a in arr_in]

timeseries.append(np.random.normal(0,np.nanstd(timeseries[2]),len(timeseries[2])))


metrics= scm.compute_sm_metrics(timeseries,nsmooth=nsmooth)

#% Plot the power spectra

dtplot = 30*24*3600
decadal_focus=False

if decadal_focus:
    xper            = np.array([10,5,1,0.5])
else:
    xper            = np.array([40,10,5,1,0.5])
xper_ticks          = 1 / (xper*12)



fig,ax1 = plt.subplots(1,1,figsize=(10,4),constrained_layout=True)
ax2     = ax1.twinx()
for ii in range(4):
    
    if ii == 1:
        axin = ax2
    else:
        axin = ax1
    
    plotfreq = metrics['freqs'][ii]*dtplot
    plotvar  = metrics['specs'][ii]/dtplot
    
    axin.plot(plotfreq,plotvar,label=arrname[ii],c=arrcol[ii])

ax1.legend()

for a,ax in enumerate([ax1,ax2]):
    
    ax.set_xlim([xper_ticks[0],0.5])
    
    ax.axvline([1/(6)],label="",ls='dotted',c='gray')
    ax.axvline([1/(12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(5*12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(10*12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(40*12)],label="",ls='dotted',c='gray')
    
    ax.set_xlabel("Frequency (1/Month)",fontsize=14)
    if a == 0:
        ax.set_ylabel(r"Power [$(\frac{W}{m^2}) ^2 cycle \, per \, mon$]")
    else:
        ax.set_ylabel(r"Power [$(\degree C^2 cycle \, per \, mon$]")
    
    if a == 0:
        
        axp = ax.twiny()
        axp.set_xlim([xper_ticks[0],0.5])
        #axp.set_xscale('log')
        axp.set_xticks(xper_ticks,labels=xper,rotation=90)
        axp.set_xlabel("Period (Years)",fontsize=14)


#%%
    
    
    

#sstin = [flxarr[:,0,klat,klon],Fprime[:,0,klat,klon]]



ax#%%
ntime,nens,nlat,nlon        = qnet.shape # Check sizes and get dimensions for tiling
ntimeh,nensh,nlath,nlonh    = hff.shape
nyrs                        = int(ntime/12)
hfftile                     = np.tile(hff.transpose(1,2,3,0),nyrs)
hfftile                     = hfftile.transpose(3,0,1,2)
# Check plt.pcolormesh(hfftile[0,0,:,:]-hfftile[12,0,:,:]),plt.colorbar(),plt.show()

#% Calculate F'
Fprime       = qnet - hfftile*np.roll(sst,nroll,axis=0) # Minus is the correct way to go
#Fprime_minus = qnet - hfftile*np.roll(sst,nroll)





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare stochastic model inputs

based on "viz_inputs_paper_draft"
Created with the intention of comparing input parameters for different
detrending cases to see why there is no longer an agreement...

Created on Mon Sep 22 09:56:53 2025

@author: gliu

"""

import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import os
import matplotlib.patheffects as PathEffects
from cmcrameri import cm

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine     = "Astraeus"

# First Load the Parameter File
cwd         = os.getcwd()
sys.path.append(cwd+ "/..")
sys.path.append("../")
import reemergence_params as rparams

# Paths and Load Modules
machine     = "Astraeus"
pathdict    = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])
from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx

# Set needed paths
figpath     = pathdict['figpath']
proc.makedir(figpath)
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
rawpath     = pathdict['raw_path']

# ----------------------------------
#%% User Edits
# ----------------------------------

# Indicate the expriment names for the parameters to load

expnames        = ["SST_ORAS5_avg","SST_ORAS5_avg_GMSSTmon"]
expcols         = ["hotpink","navy",]
expnames_long   = ["LinearDetrend","GMSSTmon"]

# # # (10) Linear Detrend (All Month) vs GMSST Removal (Sep Month)
# comparename     = "DetrendingEffect"
# expnames        = ["SST_ORAS5_avg","SST_ORAS5_avg_GMSST","SST_ORAS5_avg_GMSSTmon",]#"SST_ERA5_1979_2024"]
# expnames_long   = ["LinearDetrend","GMSST","GMSSTmon","ERA5"]
# expnames_short  = ["LinearDetrend","GMSST","GMSSTmon","ERA5"]
# expcols         = ["hotpink","cornflowerblue","navy","k"]
# expls           = ["dotted",'solid',"dashed",'solid']
# detect_blowup   = True

# Constants
dt                  = 3600*24*30 # Timestep [s]
cp                  = 3850       # 
rho                 = 1026    #`23      # Density [kg/m3]
B                   = 0.2        # Bowen Ratio, from Frankignoul et al 1998
L                   = 2.5e6      # Specific Heat of Evaporation [J/kg], from SSS model document

fsz_tick            = 18
fsz_title           = 24
fsz_axis            = 22


debug               = False



#%% Add some functions to load (and convert) inputs

def stdsqsum(invar,dim):
    return np.sqrt(np.nansum(invar**2,dim))

def stdsq(invar):
    return np.sqrt(invar**2)

def stdsqsum_da(invar,dim):
    return np.sqrt((invar**2).sum(dim))

def convert_ds(invar,lat,lon,):
    
    if len(invar.shape) == 4: # Include mode
        nmode = invar.shape[0]
        coords = dict(mode=np.arange(1,nmode+1),mon=np.arange(1,13,1),lat=lat,lon=lon)
    else:
        coords = dict(mon=np.arange(1,13,1),lat=lat,lon=lon)
    
    return xr.DataArray(invar,coords=coords,dims=coords)

def compute_detrain_time(kprev_pt):
    
    detrain_mon   = np.arange(1,13,1)
    delta_mon     = detrain_mon - kprev_pt#detrain_mon - kprev_pt
    delta_mon_rev = (12 + detrain_mon) - kprev_pt # Reverse case 
    delta_mon_out = xr.where(delta_mon < 0,delta_mon_rev,delta_mon) # Replace Negatives with 12+detrain_mon
    delta_mon_out = xr.where(delta_mon_out == 0,12.,delta_mon_out) # Replace deepest month with 12
    delta_mon_out = xr.where(kprev_pt == 0.,np.nan,delta_mon_out)
    
    return delta_mon_out


#%% Plotting Params

mpl.rcParams['font.family'] = 'Avenir'
bboxplot                    = [-80,0,20,65]
proj                        = ccrs.PlateCarree()
#lon                         = daspecsum.lon.values
#lat                         = daspecsum.lat.values
mons3                       = proc.get_monstr()


plotver                     = "rev1" # [sub1]

#%% Load BSF and Ice Mask (copied from compare_detrainment_damping)

# Load Region Information
regionset       = "SSSCSU"
regiondicts     = rparams.region_sets[regionset]
bboxes          = regiondicts['bboxes']
regions_long    = regiondicts['regions_long']
rcols           = regiondicts['rcols']
rsty            = regiondicts['rsty']
regplot         = [0,1,3]
nregs           = len(regplot)


# Get Point Info
pointset        = "PaperDraft02"
ptdict          = rparams.point_sets[pointset]
ptcoords        = ptdict['bboxes']
ptnames         = ptdict['regions']
ptnames_long    = ptdict['regions_long']
ptcols          = ptdict['rcols']
ptsty           = ptdict['rsty']


#%% Check and Load Params (copied from run_SSS_basinwide.py on 2024-03-04)

# Load the parameter dictionary
expparams_byvar     = []
paramset_byvar      = []
convdict_byvar      = []
convda_byvar        = []
inputs_byvar        = []
for expname in expnames:
    
    print("Loading inputs for %s" % expname)
    
    expparams_raw   = np.load("%s%s/Input/expparams.npz" % (output_path,expname),allow_pickle=True)
    
    expparams       = scm.repair_expparams(expparams_raw)
    
    # Get the Variables (I think only one is really necessary)
    #expparams_byvar.append(expparams.copy())
    
    # Load Parameters
    paramset = scm.load_params(expparams,input_path)
    inputs,inputs_ds,inputs_type,params_vv = paramset
    
    # Convert to the same units
    convdict                               = scm.convert_inputs(expparams,inputs,return_sep=True)
    
    # Get Lat/Lon
    ds = inputs_ds['h']
    lat = ds.lat.data
    lon = ds.lon.data
    
    # Convert t22o DataArray
    varkeys = list(convdict.keys())
    nk = len(varkeys)
    conv_da = {}
    for nn in range(nk):
        #print(nn)
        varkey = varkeys[nn]
        invar  = convdict[varkey]
        conv_da[varkey] =convert_ds(invar,lat,lon)
        
    
    # Append Output
    expparams_byvar.append(expparams)
    paramset_byvar.append(paramset)
    convdict_byvar.append(convdict)
    convda_byvar.append(conv_da)
    inputs_byvar.append(inputs_ds)


#ds_params = [xr.merge(ds) for ds in paramset_byvar]


#%% Do some Basic Comparison



#%% Forcing Amplitude

vname  = "lbd_d" # "Fprime","Alpha","lbd_a"
fig,ax = viz.init_monplot(1,1)

for ex in range(3):
    expname     = expnames_long[ex]
    print(expname)
    if vname == "lbd_d":
        plotvar = inputs_byvar[ex][vname]
    else:
        plotvar     = convda_byvar[ex][vname]
    _,nlat,nlon = plotvar.shape
    aavg = proc.area_avg_cosweight(plotvar)
    ax.plot(mons3,aavg,label=expname,c=expcols[ex])
    
ax.legend()
ax.set_title(vname)


#%% Same as above, but visualize for each point

lonf = -39.75
latf = 60.00
_,nlat,nlon = (inputs_byvar[ex][vname]).squeeze().shape
#
vname  = "h"#"lbd_d" #"Fprime"#"lbd_a"#"lbd_d" # "Fprime","Alpha","lbd_a"
fig,ax = viz.init_monplot(1,1)

for ex in range(3):
    expname     = expnames_long[ex]
    print(expname)
    if vname == "lbd_d" or "h":
        plotvar = inputs_byvar[ex][vname]
    else:
        plotvar     = convda_byvar[ex][vname]
    _,nlat,nlon = (plotvar.squeeze()).shape
    
    # Plot Area Average
    aavg = proc.area_avg_cosweight(plotvar)
    ax.plot(mons3,aavg,label=expname,c=expcols[ex])
    
    std = plotvar.std(['lon','lat'])
    ax.fill_between(mons3,aavg-std,aavg+std,label="",color=expcols[ex],alpha=0.1)
    
    # Plot Problem Point
   
    probpt = plotvar.sel(lon=lonf,lat=latf,method='nearest')
    if ex == 0:
        probpt_label = "Lon=%.2f, Lat = %.2f" % (probpt.lon.data.item(),probpt.lat.item())
    else:
        probpt_label=""
    ax.plot(mons3,probpt,label=probpt_label,c=expcols[ex],ls='dotted')
    
    # # Plot individual points
    # for a in range(nlat):
    #     for o in range(nlon):
            
    #         plotpt = plotvar.isel(lat=a,lon=o)
    #         ax.plot(mons3,plotpt,label=expname,c=expcols[ex],alpha=0.05)
    
    # Plot the problem point
    
ax.legend(ncol=2)
ax.set_title(vname)


==
    
    
#%% Look at Parameter Difference Maps for each month


vname       = "Fprime" # "Fprime","Alpha","lbd_a"


if vname == "lbd_d":
    inds    = inputs_byvar
    vmax    = 0.05
else:
    inds    = convda_byvar
    if vname == "lbd_a":
        vmax    = 0.05
    else:
        vmax    = 0.025





for im in range(12):
    plotvar     = (inds[1][vname] - inds[0][vname]).isel(mon=im)
    
    fig,ax,_ = viz.init_regplot(regname="SPGE",fontsize=30)
    
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,cmap='cmo.balance',vmin=-vmax,vmax=vmax)
    ax.set_extent([-42,-14,50,64])
    cb = viz.hcbar(pcm,ax=ax)
    cb.ax.tick_params(labelsize=30)
    
    
    cb.set_label("%s %s: %s - %s" % (mons3[im],vname,expnames_long[1],expnames_long[0]),fontsize=30)


#%% Compare Monthly Maps of Lbd_a
vname = "lbd_a"
ex    = 1
vmax  = 35
cints = [0,]

im    = 0


for im in range(12):
    fig,ax,_ = viz.init_regplot(regname="SPGE",fontsize=30)
    plotvar  = inputs_byvar[ex][vname].isel(mon=im)
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,cmap='cmo.balance',vmin=-vmax,vmax=vmax)
    cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,levels=cints,linewidths=0.5,colors="k")
    ax.set_extent([-42,-14,50,64])
    cb = viz.hcbar(pcm,ax=ax)
    cb.ax.tick_params(labelsize=30)
    cb.set_label("%s %s for %s [W/m2]" % (mons3[im],vname,expnames_long[ex]),fontsize=30)


#%% Load in the area-averaged timeseries to compare...

aavgs = []
for expname in expnames:
    
    print("Loading area-average timeseries for %s" % expname)
    aavgpath = "%s%s/Metrics/Area_Avg_SPGNE_lon320to345_lat052to062.nc" % (output_path,expname)

    aavg = xr.open_dataset(aavgpath).load()
    aavgs.append(aavg)
    

#%% Look at just the timeseries

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12.5,4))
for ex in range(2):
    
    plotvar = aavgs[ex].SST
    ax.plot(plotvar,label=expnames_long[ex],c=expcols[ex])
    
    
ax.set_ylim([-4,4])
#ax.set_xlim([6000,13000])
ax.set_xlim([24000,37000])
ax.set_ylabel("SST Anomaly [degC]")
ax.set_xlabel("Timestep (Month)")
for ii in range(10):
    if ii == 0:
        label="Simulation Chunk (1kyr)"
    else:
        label=""
    ax.axvline([ii*12000],color="k",ls='dotted',label=label)
ax.legend(ncol=3)

#%%
fig,ax = viz.init_monplot(1,1)

for ex in range(2):
    expname     = expnames_long[ex]
    print(expname)
    if vname == "lbd_d":
        plotvar = inputs_byvar[ex][vname]
    else:
        plotvar     = convda_byvar[ex][vname]


#%% Load Timeseries for the actual Point 
# Note: Only works for the case without GMSST all months

bup = []
smscrap_path = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/scrap/"
for expname in expnames:
    ds = xr.open_dataset("%s%s_blowup.nc" % (smscrap_path,expname)).load()
    bup.append(ds)
    
#%% Also manually load the output where lbd_a was just replaced

output_path= "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/"
dsrep = dl.load_smoutput("SST_ORAS5_avg_GMSSTmon_lbdswap_pt",output_path)


dsreppt = dsrep.SST.sel(lon=lonf,lat=latf,method='nearest')
    
#%% Examine Behavior for a particular run

nr     = 0


fig,ax = plt.subplots(1,1)
for ex in range(2):
    plotvar = bup[ex].SST.isel(run=nr).data
    xdim    = np.arange(len(bup[ex].SST.isel(run=nr).time))
    ax.plot(xdim,plotvar,label=expnames_long[ex],c=expcols[ex])
    
# Plot for the lambda swap
plotvar = dsreppt.isel(run=nr)
xdim = np.arange(len(plotvar))
ax.plot(xdim,plotvar,label="Replace Lbd_a",c='magenta')

# for ii in range(10):
    
#     ijan = ii*12 + 0
#     ax.axvline(ijan,color='cyan')
    
#     ijuly = ii*12 + 6
#     ax.axvline(ijuly,color='red')
    
    
ax.legend()
ax.set_xlim([0,1200])
ax.set_ylim([-10,10])

#ax.set_xticks(np.arange(0,132,12))

#%% Try to diagnose differences for particular run

nr      = 0
diffrun = bup[1].isel(run=nr).SST - bup[0].isel(run=nr).SST # Difference between runs

movmean = lambda x,N : np.convolve(x, np.ones(N)/N, mode='same')


diffmean = movmean(diffrun,12)


peturb = diffrun-diffmean


monpeturb = peturb.groupby('time.month').mean('time')#.plot()


#%%

fig,ax=viz.init_monplot(1,1)

ax.plot(np.abs(monpeturb))
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


import glob

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
expnames        = ["SST_ORAS5_avg","SST_ORAS5_avg_EOF","SST_ORAS5_avg_GMSSTmon_EOF"]
# expnames        = ["SST_ORAS5_avg","SST_ORAS5_avg_GMSSTmon"]
expcols         = ["hotpink","cornflowerblue","navy",]
# expnames_long   = ["LinearDetrend","GMSSTmon"]
expnames_long   = ["stdev(F') Forcing","EOF-based Forcing","EOF-based Forcing with GMSSTmon Detrend"]


expnames        = ["SST_ORAS5_avg","SST_ORAS5_avg_GMSSTmon_EOF","SST_ORAS5_avg_GMSSTmon_EOF_usevar",]#"SST_ERA5_1979_2024"]
expnames_long   = ["stdev(F') Forcing","EOF-based Forcing","EOF-based Forcing (corrected)","ERA5"]


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
    print(expparams['Fprime'])
    
    # Get the Variables (I think only one is really necessary)
    #expparams_byvar.append(expparams.copy())
    
    # For correction factor, check to see if "usevar" is in the forcing name
    eof_flag = expparams['eof_forcing']
    #if eof_flag:
    if 'usevar' in expparams['Fprime']:
        usevar=True
        print("Using variance for white noise forcing...")
    else:
        usevar = False



    
    # Load Parameters
    paramset = scm.load_params(expparams,input_path)
    inputs,inputs_ds,inputs_type,params_vv = paramset
    
    if usevar:
        inputs['correction_factor'] = np.sqrt(inputs['correction_factor'])
        inputs_ds['correction_factor'] = np.sqrt(inputs_ds['correction_factor'])
        
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

#%% Also Load the Metric Files (Stdev)

stdevs      = []
stdevs_mon  = []
for expname in expnames:
    metricpath = "%s%s/Metrics/" % (output_path,expname)
    
    
    dsstd = xr.open_dataset(metricpath + "Pointwise_Variance.nc").load()
    stdevs.append(dsstd)
    
    dsstdmon = xr.open_dataset(metricpath + "Pointwise_Variance_Monthly.nc").load()
    stdevs_mon.append(dsstdmon)
    
    
    
    
#ds_params = [xr.merge(ds) for ds in paramset_byvar]



#%% Check Fprime between runs

#vname= 'correction'
fig,ax = plt.subplots(1,1)
plotvars = []
for ex in range(3):
    
    plotvar = convda_byvar[ex][vname]
    #plotvar = inputs_byvar[ex]['Fprime'].squeeze()#.isel(mode=0)
    if ex == 0:
        continue
    plotvar = inputs_byvar[ex]['correction_factor'].squeeze()#.isel(mode=0)
    # if ex == 2:
    #     plotvar = np.sqrt(plotvar)
    
    eof_flag=False
    
    if len(plotvar.shape) > 3:
        eof_flag=True
        plotvar = stdsqsum_da(plotvar,'mode') + convda_byvar[ex]['correction_factor']
    
    
    
    if ex == 2:
        ls = 'dotted'
    else:
        ls='solid'
        
    aavg = proc.area_avg_cosweight(plotvar)
    ax.plot(mons3,aavg,label=expnames[ex],c=expcols[ex],ls=ls)
    
    plotvars.append(plotvar)
ax.legend()




#%% Do some Basic Comparison



#%% Forcing Amplitude

vname  = "lbd_a"#"Fprime"#"lbd_d" # "Fprime","Alpha","lbd_a"
fig,ax = viz.init_monplot(1,1)

plotvars =[]
for ex in range(3):
    expname     = expnames[ex]#_long[ex]
    print(expname)
    
    if vname == "lbd_d":
        plotvar = inputs_byvar[ex][vname]
    else:
        plotvar     = convda_byvar[ex][vname]
        
    
    
    eof_flag=False
    if len(plotvar.shape) > 3:
        print("True for %i" % ex)
        eof_flag=True
        #if ex == 1:
        plotvar = stdsqsum_da(plotvar,'mode') + convda_byvar[ex]['correction_factor']

    # if ex == 2:
    #     plotvar = np.sqrt(plotvar)
            
    
    _,nlat,nlon = plotvar.shape
    
    aavg = proc.area_avg_cosweight(plotvar)
    ax.plot(mons3,aavg,label=expname,c=expcols[ex])
    plotvars.append(plotvar)
    
ax.legend()
ax.set_title(vname)

#%% Check Forcing Differences

fprime            = convda_byvar[0]['Fprime']
eof_uncombine     = np.concatenate([convda_byvar[1]['Fprime'].data,convda_byvar[1]['correction_factor'].data[None,...]],axis=0)

eof               = stdsqsum_da(convda_byvar[1]['Fprime'],'mode') +  convda_byvar[1]['correction_factor']

correction_factor = convda_byvar[1]['correction_factor']

# Take Ratio of Both, max is 1.07
fratio            = eof/fprime

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


#%% Load White Noise Forcings

wn_fstd_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/SST_Obs_Pilot_00_Tdcorr0_qnet/Input/"
wn_eof_path  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/SST_ORAS5_avg_EOF/Input/"

wnpaths = [wn_fstd_path,wn_eof_path]
wnsall = []
for ii in range(2):
    
    searchstr = "%s*.npy" % wnpaths[ii]
    filelist  = glob.glob(searchstr)#.sort()
    filelist.sort()
    print(filelist)
    nfiles = len(filelist)
    
    wnlist = []
    for ff in range(nfiles):
        ld = np.load(filelist[ff])
        wnlist.append(ld)
    wnsall.append(wnlist)
    
#%% Check the variance

wnsall       = [np.array(wn) for wn in wnsall]
wnfstd,wneof = wnsall # [Run x Year x Mon], [Run x Year x Mon x Mode]



#%% Check the variance of the timeseries

wnfstd_monvar = wnfstd.reshape((10*1000,12)).std(0)

wneof_monvar = stdsqsum(wneof.reshape((10*1000,12,47)),-1).std(0)

# Rehape yr mon to time
wneof1 = wneof.reshape((10*1000,12,47)) # [time x mode x mon]
    
#%%

fig,ax = plt.subplots(1,1,figsize=(8,4.5))
ax.plot(wnfstd_monvar,label="Fstd Wn")
ax.plot(wneof_monvar,label="EOF Wn",ls='dashed')
ax.legend()


#%%

#wneof_monvar/wnfstd_monvar

mwn     = lambda x: np.random.normal(0,1,x) # make white noise
wn10k   = [mwn(1000) for ii in range(1000)]
stds    = [np.nanstd(ts) for ts in wn10k]

#%%

cumustd = []
for ii in range(1000):
    
    ts = []
    for cu in range(ii+1):
        ts.append(wn10k[cu])
    
    tsall = np.array(ts)
    tssum = np.nansum(tsall,0)
    cumustd.append(np.nanstd(tssum))

#%% Try to remake the forcing 


eof_usevar      = True
eof_uncombine2  = np.concatenate([convda_byvar[2]['Fprime'].data,convda_byvar[2]['correction_factor'].data[None,...]],axis=0)

# Prepare Forcing (EOF) =======
if eof_usevar:
    eof_uncombine_rs  = np.concatenate([convda_byvar[2]['Fprime'].data,convda_byvar[2]['correction_factor'].data[None,...]],axis=0)
    eof_uncombine_rs  = eof_uncombine_rs.transpose(0,2,3,1) # [Mode x Lat x Lon x Month]
else:
    eof_uncombine_rs  = eof_uncombine.transpose(0,2,3,1) # [Mode x Lat x Lon x Month]



    # Apply Sqrt Correction
    eof_uncombine_rs = np.sign(eof_uncombine_rs) * np.sqrt(np.abs(eof_uncombine_rs)) 

# Tile and compbine
eof_uncombine_rs = np.tile(eof_uncombine_rs,1000) # {mode x Lat x Lon x Time}0
wn_eof_rs        = wneof[1,:,:,:].reshape(12000,47).transpose(1,0)
eof_wn_sum       = eof_uncombine_rs * wn_eof_rs[:,None,None,:]

# Sum across modes
eof_force_final  = eof_wn_sum.sum(0)
#eof_force_final  = np.sqrt(np.abs(eof_force_final)) * np.sign(eof_force_final)


# Prepare Forcing (Fstd) ======
# Reshape and Tile
wnfstd_rs        = wnfstd[0,:,:].reshape(12000)[None,None,:] # [1 x 1 x time] 
fprime_rs        = np.tile(fprime.data.transpose(2,1,0),1000)

# Apply Sqrt Correction
fstd_force_final =  wnfstd_rs * fprime_rs

# Prepare Coords
coords           = dict(lat=fprime.lat,lon=fprime.lon,time=np.arange(12000))
forcing_fstd     = (fstd_force_final)
#eof_forcing_sep = eof_umcombine
#%% Check Spatial Pattern of forcing
# Note: The overall variance looks pretty decent, close to 1:1 ratio

stdrat = np.nanvar(eof_force_final,-1) / np.nanvar(fstd_force_final,-1).T

fig,ax = plt.subplots(1,1)

pcm = ax.pcolormesh(lon,lat,stdrat)    
fig.colorbar(pcm,ax=ax)

#%% Now check the monthly variance
# NOte: This looks less good, with large underestimates in July

# Take pointwise variance
in_forcings     = [eof_force_final,fstd_force_final.transpose(1,0,2)]
forcings_monvar = [ts.reshape(41,101,1000,12).var(2) for ts in in_forcings]

# Area-Average, then take monthly variance
in_aavgs        = [ts.mean((0,1)) for ts in in_forcings]
in_monvars      = [ts.reshape(1000,12).var(0) for ts in in_aavgs]

# Take Area-Average of the monthly variances
monvar_aavg = forcings_monvar[0].mean((0,1)) / forcings_monvar[1].mean((0,1))

# Look at monthly ratio
ratio_ts = in_aavgs[0]/in_aavgs[1]
ratio_monmean = ratio_ts.reshape(1000,12).mean(0)


#%%
mons3       = proc.get_monstr()
inlabels    = ["EOF","Fstd"]
fig,ax = viz.init_monplot(1,1)
vratio_monvar = in_monvars[0]/in_monvars[1] 

for ii in range(2):
    ax.plot(mons3,in_monvars[ii],label=inlabels[ii])




ax2 = ax.twinx()
ax2.plot(mons3,vratio_monvar,c='pink')

ax2.plot(mons3,monvar_aavg,c='green')
#ax2.plot(mons3,ratio_monmean,c='midnightblue')

ax.legend()

#%% Check pointwise montly variance (and the ratio)

ttls = ["EOF","Fstd","EOF/Fstd"]
for im in range(12):
    fig,axs = plt.subplots(1,3,constrained_layout=True,figsize=(12.5,4.5))
    
    for ii in range(3):
        ax  = axs[ii]
        
        if ii<2:
            plotvar = forcings_monvar[ii][:,:,im]
            cmap = 'cmo.balance'
            vlm = [-.2,.2]
            
        else:
            plotvar = forcings_monvar[0][:,:,im] / forcings_monvar[1][:,:,im]
            vlm = [0.8,1.2]
            cmap = 'cmo.dense'
            
        pcm = ax.pcolormesh(lon,lat,plotvar,vmin=vlm[0],vmax=vlm[1],cmap=cmap)
        cb = viz.hcbar(pcm,ax=ax)
        
        
        
        ax.set_title(ttls[ii])
        
        if ii==2:
            minrat = np.nanmin(np.abs(plotvar.flatten()))
        
    plt.suptitle("%s (minrat = %.4f)" % (mons3[im],minrat))

#%% Plot Overall Variance

var_ratio_forcingall = in_forcings[0].var(-1) / in_forcings[1].var(-1)

fig,ax     = plt.subplots(1,1,constrained_layout=True,figsize=(8,4.5))
cints      = np.arange(.7,1.225,.025)
cf = ax.contourf(var_ratio_forcingall,levels=cints,cmap='cmo.dense')
cl = ax.contour(var_ratio_forcingall,levels=cints,colors="k",linewidths=0.5)
ax.clabel(cl)
cb = viz.hcbar(cf,fraction=0.055)

# ----------------------------------    
#%% Test 1: Is it due to the area average?
# gradually select a larger and larger region, take area average, then calculate monvar
# ---------------------------------- 

centpoint = [-28,60,]
itersize  = 0.5
klon,klat = proc.find_latlon(centpoint[0],centpoint[1],lon,lat)

expand_size = int((62-57)/itersize)
bboxes = []
for ii in range(expand_size):
    lonf,latf=centpoint
    
    boxexp = ii*itersize
    bbox = [lonf-boxexp , lonf+boxexp, latf-boxexp, latf+boxexp]
    
    bboxes.append(bbox)
    print(bbox)
    
bbox_spgne = [-40,-15,52,62]


#ax.set_xlim([-40,-15])
#ax.set_ylim([52,62])
#ax.set_extent(bbox_spgne)
coords = dict(lon=lon,lat=lat,time=np.arange(12*1000))

in_forcings_ds = [xr.DataArray(infrc,coords=coords,dims=coords) for infrc in in_forcings]

varforcings = [proc.area_avg_cosweight(ff).var('time') for ff in in_forcings_ds]

def calc_aavg_monvar(invar,bb):
    aavg = proc.sel_region_xr(invar,bb)
    aavg = proc.area_avg_cosweight(aavg)
    monv = aavg.data.reshape(1000,12).var(0)
    return monv

monv_bbexp = []
monv_bbrat = []
for bb in bboxes:
    monvbb_eof  = calc_aavg_monvar(in_forcings_ds[0],bb)
    monvbb_fstd = calc_aavg_monvar(in_forcings_ds[1],bb)
    
    monvbb_ratio = monvbb_eof / monvbb_fstd
    
    monv_bbrat.append(monvbb_ratio)
    monv_bbexp.append([monvbb_eof,monvbb_fstd])
    
    


fig,ax= plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
viz.plot_box(bbox_spgne,ax=ax,color='cyan')
viz.add_coast_grid(ax,bbox=bbox_spgne)
for bb in bboxes:
    viz.plot_box(bb,ax=ax)

    

#%% Plot the effect



for ii in range(3):
    
    if ii ==2:
        title   = title
        ylm     = [.25,1.25]
    else:
        
        ylm     = [0,0.75]
        
        title   = "Monthly Variance (%s)" % ttls[ii]
        
        
            
    fig,ax = viz.init_monplot(1,1,figsize=(8,4.5))

    
    for b,bb in enumerate(bboxes):
        
        label = "bbsize = $\pm$ %.2f deg" % (b*itersize)
            
        
        
        if ii == 2:
            plotvar = monv_bbrat[b]
        else:
            plotvar = monv_bbexp[b][ii]
            
    
        ax.plot(mons3,plotvar,label=label,lw=2.5)
        
    ax.legend(fontsize=8,ncol=2)
    ax.set_ylim(ylm)
    ax.set_title(title)
        
    

# =======================
#%% Test EOF-Forcing vs. Fstd at point, for many WN timeseries
# =======================

import tqdm

def asqrt(x):
    return np.sqrt(np.abs(x)) * np.sign(x)

a   = 22
o   = 35
im  = 6

#for im in range(12):
eofpt  = eof_uncombine[:,im,a,o]
fstdpt = fprime.data[im,a,o]
eofmag = stdsqsum(eofpt[:-1],0) + eofpt[-1]#np.sqrt(np.sum(eofpt**2))


eof_uncombine2 = np.concatenate([convda_byvar[2]['Fprime'].data,convda_byvar[2]['correction_factor'].data[None,...]],axis=0)
eofpt_sq       = eof_uncombine2[:,im,a,o] #eofpt**2  * np.sign(eofpt)


print(eofmag)
print(fstdpt)
print("\n")

mwnn = lambda : mwn(1000)

nmode           = len(eofpt)

eof_std_in      = eofpt[:-1]
#correction_std  = asqrt(eofpt[-1])

eof_std_in      = eofpt_sq

fpstds          = []
wnstds          = []
for mc in tqdm.tqdm(range(100)):
    #eof_sqrt  = [np.sqrt(np.abs(ii))*np.sign(ii) for ii in eofpt]
    #eofwnpt   = np.nansum(np.array([mwn(10000)*np.sqrt(np.abs(ii))*np.sign(ii) for ii in eofpt]),0)
    
    #eofwnpt   = np.nansum(np.array([mwn(1000)*np.sqrt(np.abs(ii))*np.sign(ii) for ii in eofpt]),0)
    
    # EOF Forcing (sqrt(eof_sq))
    #eofwnpt   = np.nansum(np.array([mwn(1000)*ii for ii in eof_std_in]),0)
    
    eofsumpt   = np.array([mwn(10000) * ii for ii in eof_std_in]) # Multiply each white noise timeseries by the variance
    # This part makes sense, it seems each compoennt has the sqrt same as eof_std_in....
    
    
    # Need to RECALCULATE correction factor... this time as variances rather than as standard deviations
    eofsumpt     = np.nansum(eofsumpt,0)
    vareoffilt   = np.var(eofsumpt)
    #updated_corr = fstdpt**2 - vareoffilt # THIS WAS THE ANSWER
    #corrpt     = mwn(10000) * np.sqrt(updated_corr)
    #updated_corr = eofpt_sq[-1]#fstdpt**2 - vareoffilt
    corrpt     = 0#mwn(10000) * eofpt_sq[-1] # In udpated case we just folded the correction factor it
    
    
    
    
    #eofwnpt   = np.sqrt(np.abs(np.nansum(eofwnpt,0))) * np.sign(eofpt) + mwn(1000)*correction_std
    
    
    eofwnpt    = eofsumpt + corrpt
    
    #eofwnpt   = eofwnpt * np.sign(eofpt)[:,None] # Apply Sign
    #eofwnpt   = eofwnpt.sum(0) # Sum modes
    #eofwnpt   = np.sqrt(np.abs(eofwnpt)) * np.sign(eofwnpt)
    
    
    
    # Single Forcing: N(0,1) * sigma_fstd
    fstdptts  = mwn(1000) * fstdpt#* np.sqrt(fstdpt) #np.sqrt(fstdpt)#np.sqrt(fstdpt)
    
    wnstds.append(np.var(eofwnpt))
    fpstds.append(np.var(fstdptts))

print(np.nanvar(eofwnpt))
print(np.nanvar(fstdptts))

#%

#%
fig,ax = plt.subplots(1,1)

ax.hist(wnstds,color='goldenrod',alpha=0.55,label="Reconstructed Forcing = %.4f" % (np.nanvar(eofwnpt)))
ax.hist(fpstds,color='r',alpha=0.55,label="Original Forcing = %.4f" % (np.nanvar(fstdptts)))
ax.axvline([fstdpt**2],c='k')

# 95% and Mean for EOF Forcing
ax.axvline(np.nanmean(wnstds),c='midnightblue')
ax.axvline(np.quantile(wnstds,[0.025]),c='midnightblue',ls='dotted')
ax.axvline(np.quantile(wnstds,[0.975]),c='midnightblue',ls='dotted')

# 95% and Mean for Fstd Forcing
ax.axvline(np.nanmean(fpstds),c='darkred')
ax.axvline(np.quantile(fpstds,[0.025]),c='darkred',ls='dotted')
ax.axvline(np.quantile(fpstds,[0.975]),c='darkred',ls='dotted')


ax.legend()
ax.set_title("mon=%02i" % (im+1))
ax.set_xlim([0.01,.15])


#%%
# This is the "perfect solution"? What are you even solving... (Which is the same as the basic solution)

eofmod = np.array([(asqrt(ii)**2)*np.sign(ii) for ii in eof_std_in])
                  
#eofsumpt   = eof_std_in #np.array([mwn(1000) * (asqrt(ii)**2)*np.sign(ii) for ii in eof_std_in])
eofsumpt = np.array([mwn(10000) * ii for ii in eof_std_in])


#eofsumpt = np.array([mwn(10000) * ii for ii in eofpt])



cswn  = []
cseof = []

csfix  = []
for ii in range(len(eof_std_in)):
    
    cswn.append( np.var(( eofsumpt[:ii,:] ).sum(0)) )
    cseof.append( (eof_std_in[:ii]**2).sum() )
    
    
    #csfix.append(np.var(( eofsumpt[:ii,:] ).sum(0)) +  np.var(mwn(1000))*eofpt[-1] )
    
    
# Basically correction factor is wrong because it is the difference of squares
corr_updated = fstdpt**2 - np.var(eofsumpt.sum(0))

csfix = np.var(eofsumpt.sum(0) +  mwn(10000) * np.sqrt(corr_updated) )#* eofpt[-1] #+ eofpt[-1]#+ mwn(10000)*eofpt[-1])


fig,ax = plt.subplots(1,1)

ax.axhline( (fstdpt - eofpt[-1])**2 ,c='orange',label="Target - Correction (Filtered EOF Amplitude)")
ax.plot(cswn,label='var timeseries')
ax.plot(cseof,color='blue',label='var EOF')

ax.axhline(csfix,color='gray',label="varEOF + correction")


ax.axhline( (fstdpt)**2,c='red',label="Target")
ax.axhline( stdsqsum(eofpt,0)**2 ,c='blue',label="Actual")
#ax.axhline(np.var(fstdptts))

ax.legend()

# It see svariance keeps growing but some is removed for the 

#%% Lets do a synthetic case

wnstds = []
fpstds = []



for ii in range(1000):
    
    sigma_e = np.array([np.sqrt(1.5),np.sqrt(0.5)]) # Sum of variance should add up (not stadard deviation)
    
    #sigma_e = np.array([1.5**2,.5**2])
    # i.e. if a = b+c, this ddoes not mean that sqrt(a) = sqrt(b) + sqrt(c)
    
    wne     = np.array([mwn(12000)*(e) for e in sigma_e])
    
    wntest  = wne.sum(0)# np.sqrt(np.abs(wne.sum(0))) * np.sign(wne.sum(0))
    
    #fptest  = np.random.normal(0,1,1000) * np.sqrt(2)
    
    fptest  = np.random.normal(0,1,1000) * np.sqrt(2)#**2 #* np.sqrt(2)
    
    wnstds.append(np.std(wntest))
    fpstds.append(np.std(fptest))
    
    
#%%

fig,ax = plt.subplots(1,1)

ax.hist(wnstds,color='goldenrod',alpha=0.55,label="Reconstructed Forcing")
ax.hist(fpstds,color='r',alpha=0.55,label="Original Forcing")



    # print("Std(wn) = %.2f" % (np.std(wntest)))
    # print("Std(fp) = %.2f" % (np.std(fptest)))
    # print('\n')




#%% Visualize Differences between Forcings

# Similar to above, lets first look at differences in forcing

stdev_fstd = stdevs[0].SST
stdev_eofs = stdevs[2].SST
var_ratio_pointwise = (stdev_eofs**2 / stdev_fstd**2).mean('run')
fig,ax     = plt.subplots(1,1,constrained_layout=True,figsize=(8,4.5))
cints      = np.arange(.7,1.225,.025)
cf = ax.contourf(var_ratio_pointwise,levels=cints)
cl = ax.contour(var_ratio_pointwise,levels=cints,colors="k",linewidths=0.5)
ax.clabel(cl)
cb = viz.hcbar(cf,fraction=0.055)

#var_ratio_pointwise.plot(vmin=.5,vmax=1.2)

#%% Look at monthly case
im = 6

for im in range(12):
    stdev_fstd = stdevs_mon[0].SST.isel(mon=im)
    stdev_eofs = stdevs_mon[2].SST.isel(mon=im)
    var_ratio_pointwise_mon = (stdev_eofs**2 / stdev_fstd**2).mean('run')
    
    fig,ax     = plt.subplots(1,1,constrained_layout=True,figsize=(8,4.5))
    cints      = np.arange(.7,1.225,.025)
    cf = ax.contourf(var_ratio_pointwise_mon,levels=cints)
    cl = ax.contour(var_ratio_pointwise_mon,levels=cints,colors="k",linewidths=0.5)
    ax.clabel(cl)
    cb = viz.hcbar(cf,fraction=0.055)
    ax.set_title("%s" % mons3[im])
    
    
    
    
    #plt.contourf(var_ratio_pointwise_mon,levels=np.arange(.5,1.25,.05))








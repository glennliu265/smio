#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Examine how the area-averaged variance changes for ERA5 SST
Also check for the CESM2 Hierarchy

This Includes:
    Approach 1: Shift southern limit of box up from 0N to 50N (keep all other limits same)
    Approach 2: Shift central latitude of bounding box, with specified window size



Created on Thu May 15 14:40:03 2025

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


#%% User Edits

dpath                   = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"

# Load GMSST For ERA5 Detrending
detrend_obs_regression  = True
nc_gmsst                = "ERA5_GMSST_1979_2024.nc"

# Load Ice Mask
ds_mask                 = dl.load_mask(expname="ERA5").mask


# Set Figure Path
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20250516/"
proc.makedir(figpath)

vnames                  = ['sst',]




#%% Load the variables

nvars   = len(vnames)
ncname  = "%sERA5_%s_NAtl_1979_2024.nc"
ncname2 = "%sERA5_%s_NAtl_1979to2024.nc"
dsall  = []
for vv in tqdm.tqdm(range(nvars)):
    
    vname = vnames[vv]
    try:
        ds    = xr.open_dataset(ncname % (dpath,vname)).load()
    except:
        
        ds    = xr.open_dataset(ncname2 % (dpath,vname)).load()
        
    dsall.append(ds[vname])
    
#%% Load GMSST and also detrend pointwise

# Detrend by Regression
dsall_anom    = [proc.xrdeseason(ds) for ds in dsall]

# Detrend by Regression to the global Mean
ds_gmsst      = xr.open_dataset(dpath + nc_gmsst).GMSST_MeanIce.load()
dtout         = [proc.detrend_by_regression(ds,ds_gmsst) for ds in dsall_anom]
dsall_anom_dt = [dtout[vv][vnames[vv]] for vv in range(nvars)]
ds_sst        = dsall_anom_dt[0]

# ===================================
#%% Approach 1 (Slide box southwards)
# ===================================

sst_mask = ds_sst * ds_mask

latlim_south = np.arange(0,51,1)
nboxes       = len(latlim_south)

aavgs = []
for ii in tqdm.tqdm(range(nboxes)):
    bbsel = [-80,0,latlim_south[ii],60]
    dsreg = proc.sel_region_xr(sst_mask,bbsel)
    aavgs.append(proc.area_avg_cosweight(dsreg))
    

aavgs_lp = [proc.lp_butter(aavg,120,6) for aavg in aavgs]
stds     = np.nanstd(np.array(aavgs),1)
stds_lp  = np.nanstd(np.array(aavgs_lp),1)

vratio   = stds_lp/stds * 100


#%% Plot Latitude versus standard deviation

use_bar = True
fig,ax  = plt.subplots(1,1,constrained_layout=True,figsize=(12.5,4.5))


if use_bar:
    braw = ax.bar(latlim_south,stds,color='gray',label="Raw")
    blp  = ax.bar(latlim_south,stds_lp,color='k',label="10-year LP Filtered")
    #ax.bar_label(braw,fmt="%.2f",c='gray')
    #ax.bar_label(blp,fmt="%.2f",c='k')
else:
    ax.plot(latlim_south,stds,c='gray',label="Raw",)#labelsize=8)
    ax.plot(latlim_south,stds_lp,c='k',label="10-year LP Filtered",)#fontsize=8,rotation=90)



ax.set_ylim([0,1])
ax.set_xlim([-1,51])
ax.set_xlabel("$X=$ Southern Limit of Bounding Box [$\degree$N]")
ax.set_ylabel("1$\sigma$(SST) [$\degree$C]")
ax.legend()

     
ax.axvline([20],ls='dotted',c="red")

ax2 = ax.twinx()
ax2.plot(latlim_south,vratio,c='orange')
ax2.set_ylim([0,100])
ax2.set_ylabel(r"Percent Low-Frequency Variability, $\frac{\sigma_{LP}}{\sigma}$ [%]",color='orange')

ax.set_title("1$\sigma$ of Area-Averaged SST [80 to 00$\degree$W, $X$ to 60$\degree$N]")

figname = "%sERA5_Stdev_Shift_southern_boundary.png" % (figpath)
plt.savefig(figname,dpi=150,bbox_inches='tight')


# ===================================
#%% Approach 2 (Sliding Bounding Box)
# ===================================


window    = 5
aavgs2    = []
centlats  = []
centlat   = 0 + window
npts      = []
npts_ok   = []
while (centlat-window >= 0) and (centlat+window < 65):
    
    bbsel = [-80,0,centlat-window,centlat+window]
    dsreg = proc.sel_region_xr(sst_mask,bbsel)
    
    aavgs2.append(proc.area_avg_cosweight(dsreg))
    centlats.append(centlat)
    print(centlat)
    centlat += 1
    
    # Also Save number of non NaN pts
    regdata = dsreg.data.sum((0,1))
    npts.append(len(regdata))
    npts_ok.append(np.sum(~np.isnan(regdata)))
    

ncentlats = len(centlats)



aavgs2_lp = [proc.lp_butter(aavg,120,6) for aavg in aavgs2]
stds2     = np.nanstd(np.array(aavgs2),1)
stds2_lp  = np.nanstd(np.array(aavgs2_lp),1)

vratio2   = stds2_lp/stds2 * 100

#%% make the same plot as above but for a sliding box

use_bar = True
fig,ax  = plt.subplots(1,1,constrained_layout=True,figsize=(12.5,4.5))


braw = ax.bar(centlats,stds2,color='gray',label="Raw")
blp  = ax.bar(centlats,stds2_lp,color='k',label="10-year LP Filtered")



ax.set_ylim([0,1])
ax.set_xlim([-1,61])
ax.set_xlabel("$X=$ Central Latitude of Bounding Box [$\degree$N]")
ax.set_ylabel("1$\sigma$(SST) [$\degree$C]")
ax.legend()

     
ax.axvline([20],ls='dotted',c="red")

ax2 = ax.twinx()
ax2.plot(centlats,vratio2,c='orange')
ax2.set_ylim([0,100])
ax2.set_ylabel(r"Percent Low-Frequency Variability, $\frac{\sigma_{LP}}{\sigma}$ [%]",color='orange')

ax.set_title("1$\sigma$ of Area-Averaged SST [80 to 00$\degree$W, $X \pm %i \degree N$]" % (window))

#ax.set_xticks(centlats[::2])

figname = "%sERA5_Stdev_Shift_box_win%02i.png" % (figpath,window)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Plot Number of points just to make sure it isnt this

fig,ax  = plt.subplots(1,1,constrained_layout=True,figsize=(12.5,4.5))

braw = ax.bar(centlats,npts,color='gray',label="Total # Pts in Longitude Band")
blp  = ax.bar(centlats,npts_ok,color='k',label="# Non-NaN Points")

ax.set_ylim([0,330])
ax.set_xlim([-1,61])
ax.set_xlabel("$X=$ Central Latitude of Bounding Box [$\degree$N]")
ax.set_ylabel("1$\sigma$(SST) [$\degree$C]")
ax.legend()

     
ax.axvline([20],ls='dotted',c="red")

ax2 = ax.twinx()
ax2.plot(centlats,vratio2,c='orange')
ax2.set_ylim([0,100])
ax2.set_ylabel(r"Percent Low-Frequency Variability, $\frac{\sigma_{LP}}{\sigma}$ [%]",color='orange')

ax.set_xticks(centlats[::2])

figname = "%sERA5_NaN_PointCount_Shift_box_win%02i.png" % (figpath,window)
plt.savefig(figname,dpi=150,bbox_inches='tight')

# =============================================================================
#%% Repeat Analysis for CESM2 Hierarchy
# =============================================================================

# Load the Data
dpath_cesm = "/Users/gliu/Downloads/02_Research/01_Projects/06_POM/01_Data/proc/"
ncs_cesm = [
    "SOM_0001_0360_TS_anom.nc",
    "POM03_0101-499_TS_anom.nc",
    'FCM_1600_2000_TS_anom.nc'
    ]

cesm_names = [
    "SOM",
    "POM",
    "FCM"
    ]

cesm_colors = [
    'violet',
    'cornflowerblue',
    'forestgreen'
    ]

ncesm = len(ncs_cesm)

# Load DataSets
ds_cesm = []
for nn in range(ncesm):
    ds = xr.open_dataset(dpath_cesm + ncs_cesm[nn]).TS.load()
    
    if "SOM" in ncs_cesm[nn]:
        
        print("Dropping first 60 years for SLAB simulation")
        ds = ds.sel(time=slice('0060-02-01','0361-01-01'))
    
    ds_cesm.append(ds)
    
# Load Ice Mask
ds_mask_cesm2 = dl.load_mask(expname="cesm2_pic").mask

#%% Check lat/lon

lonpom  = ds_cesm[1].lon
lonmask = ds_mask_cesm2.lon



latpom  = ds_cesm[1].lat
latmask = ds_mask_cesm2.lat

# Important, remap pencil ocean model to the right mask
ds_cesm[1]['lat'] = latmask

    
#%% Preprocess CESM

# Apply Ice Mask and Remove Seasonal Cycle
cesm_masked = [ds * ds_mask_cesm2 for ds in ds_cesm]
cesm_masked = [proc.format_ds(ds) for ds in cesm_masked]
cesm_anom   = [proc.xrdeseason(ds) for ds in cesm_masked]

# Compute the global mean
gmsst_cesm  = [proc.area_avg_cosweight(ds) for ds in cesm_anom]

# Select the north atlantic reiong
bbox_natl   = [-100,10,-10,70]
cesm_natl   = [proc.sel_region_xr(ds,bbox_natl) for ds in cesm_anom]
dtout       = [proc.detrend_by_regression(cesm_natl[ii].rename('sst'),gmsst_cesm[ii]) for ii in range(ncesm)]
cesmsst_out = [ds.sst for ds in dtout]

#%% 

# Make Above into a function
def calc_aavg_window_lon(ds,window,bbox,startlat=0,return_count=False):
    aavgs     = []
    centlats  = []
    centlat   = startlat + window
    npts      = []
    npts_ok   = []
    while (centlat-window >= 0) and (centlat+window < 65):
        
        bbsel = [bbox[0],bbox[1],centlat-window,centlat+window]
        dsreg = proc.sel_region_xr(ds,bbsel)
        
        aavgs.append(proc.area_avg_cosweight(dsreg))
        centlats.append(centlat)
        centlat += 1
        
        # Also Save number of non NaN pts
        regdata = dsreg.data.sum((0,1))
        npts.append(len(regdata))
        npts_ok.append(np.sum(~np.isnan(regdata)))
    if return_count:
        return aavgs,centlats,npts,npts_ok
    return aavgs,centlats

window = 5
bbox   = [-80,0,None,None]
avgwin_cesm = [calc_aavg_window_lon(ds,window,bbox) for ds in cesmsst_out]

centlats_cesm = [aa[1] for aa in avgwin_cesm]
#%% Now compute standard deviation of each

def calc_stds_sample(aavgs):
    
    aavgs_lp = [proc.lp_butter(aavg,120,6) for aavg in aavgs]
    stds     = np.nanstd(np.array(aavgs),1)
    stds_lp  = np.nanstd(np.array(aavgs_lp),1)
    vratio   = stds_lp/stds * 100
    return aavgs_lp,stds,stds_lp,vratio


metrics = [calc_stds_sample(aa[0]) for aa in avgwin_cesm]

#%% Quickly plot the results

plotname = 'std' ##'vratio'


fig,ax  = plt.subplots(1,1,constrained_layout=True,figsize=(12.5,4.5))

if plotname == 'std':
    for nn in range(ncesm):
        ax.plot(centlats_cesm[nn],metrics[nn][1],label=cesm_names[nn],lw=2.5,c=cesm_colors[nn])
    ax.plot(centlats,stds2,label='ERA5',c='k')
elif plotname == "std_lp":
    for nn in range(ncesm):
        ax.plot(centlats_cesm[nn],metrics[nn][2],label=cesm_names[nn],lw=2.5,c=cesm_colors[nn])
    ax.plot(centlats,stds2_lp,label='ERA5',c='k')
elif plotname == "vratio":
    for nn in range(ncesm):
        ax.plot(centlats_cesm[nn],metrics[nn][3],label=cesm_names[nn],lw=2.5,c=cesm_colors[nn])
    ax.plot(centlats,vratio2,label='ERA5',c='k')
    
ax.legend()
ax.axvline([20],ls='dotted',c="red")


ax2.set_ylabel(r"Percent Low-Frequency Variability, $\frac{\sigma_{LP}}{\sigma}$ [%]",color='orange')

ax.set_xticks(centlats[::2])

if plotname == 'vratio':
    ax.set_ylim([0,100])
else:
    ax.set_ylim([0,1])
    
    
#ax.set_ylim([0,1])
ax.set_xlim([0,65])

ax.set_ylabel("Standard Deviation")
ax.set_xlabel("Central Longitude")

# stds_cesm       = [np.nanstd(aa[0]) for aa in avgwin_cesm]
# avgwin_lp_cesm  = [proc.lp_butter(aa[0].data,120,6) for aa in avgwin_cesm]
# stds_lp_cesm    = [np.nanstd(aa) for aa in avgwin_lp_cesm]
    

#%% Make other function, taking area averages along certain dimension

def shift_box_1d(ds,bbox,limits,shiftdim):
    nboxes = len(limits)
    aavgs  = []
    for ii in tqdm.tqdm(range(nboxes)):
        bbsel = np.array(bbox).copy() #[-80,0,latlim_south[ii],60]
        bbsel[shiftdim] = limits[ii]
        dsreg           = proc.sel_region_xr(ds,bbsel)
        aavgs.append(proc.area_avg_cosweight(dsreg))
    return aavgs

# Same as Above
latlim_south       = np.arange(0,51,1)
bbin               = [-80,0,20,60]
cesm_south_aavgs   = [shift_box_1d(ds,bbin,latlim_south,2) for ds in cesmsst_out]
    
metrics_south_cesm = [calc_stds_sample(aavg) for aavg in cesm_south_aavgs]


#%% Investigate how the standard deviation compares to FCM, etc for southern sliding window case

plot_cesm = [1,]

use_bar = True
fig,ax  = plt.subplots(1,1,constrained_layout=True,figsize=(12.5,4.5))


if use_bar:
    braw = ax.bar(latlim_south,stds,color='gray',label="Raw")
    blp  = ax.bar(latlim_south,stds_lp,color='k',label="10-year LP Filtered")
    #ax.bar_label(braw,fmt="%.2f",c='gray')
    #ax.bar_label(blp,fmt="%.2f",c='k')
else:
    ax.plot(latlim_south,stds,c='gray',label="Raw",)#labelsize=8)
    ax.plot(latlim_south,stds_lp,c='k',label="10-year LP Filtered",)#fontsize=8,rotation=90)

# Plot the case for CESM2 Hierarchy
for nn in range(ncesm):
    if nn not in plot_cesm:
        continue
    
    # Plot Stds
    ax.plot(latlim_south,metrics_south_cesm[nn][1],label="%s Raw" % cesm_names[nn],lw=2,c=cesm_colors[nn],marker='x',)
    
    # Plot Stds_LP
    ax.plot(latlim_south,metrics_south_cesm[nn][2],label="%s 10-year LP Filtered" % cesm_names[nn],lw=2,c=cesm_colors[nn],ls='dashed',marker='+')
    
ax.set_ylim([0,1])
ax.set_xlim([-1,51])
ax.set_xlabel("$X=$ Southern Limit of Bounding Box [$\degree$N]")
ax.set_ylabel("1$\sigma$(SST) [$\degree$C]")
ax.legend()


ax.axvline([20],ls='dotted',c="red")

ax.set_title("1$\sigma$ of Area-Averaged SST [80 to 00$\degree$W, $X$ to 60$\degree$N]")

figname = "%sERA5_Stdev_Shift_southern_boundary_compare_cesm.png" % (figpath)
plt.savefig(figname,dpi=150,bbox_inches='tight')




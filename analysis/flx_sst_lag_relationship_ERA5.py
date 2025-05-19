#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Examine the Heat Flux SST Relationship in ERA5

Created on Thu May 15 10:12:08 2025

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

# Set Figure Path
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20250516/"
proc.makedir(figpath)

vnames                  = ['sst','qnet','rhflx','thflx']

#%% Load the variables

nvars  = len(vnames)
ncname = "%sERA5_%s_NAtl_1979_2024.nc"
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
dsall_anom      = [proc.xrdeseason(ds) for ds in dsall]

# Confirmed it is positive downwards
ds_scycle       = [ds.groupby('time.month').mean('time') for ds in dsall]

# Detrend by Regression to the global Mean
ds_gmsst        = xr.open_dataset(dpath + nc_gmsst).GMSST_MeanIce.load()
dtout           = [proc.detrend_by_regression(ds,ds_gmsst) for ds in dsall_anom]
dsall_anom_dt   = [dtout[vv][vnames[vv]] for vv in range(nvars)]

# #%% Look at Claude's Paper and select some similar points

# lonf                    = -30
# latf                    = 30
# dspt                    = [proc.selpt_ds(ds,lonf,latf) for ds in dsall_anom_dt]
# sst,qnet,rhflx,thflx    = dspt

# #dspt_predt = xr.merge([proc.selpt_ds(ds,lonf,latf) for ds in dsall])

#%% Do a simple lag covariance for each variable

# Read out anomalous variables
sst,qnet,rhflx,thflx = dsall_anom_dt

# Compute lead-lag coariance at each location
lags        = np.arange(12)
monwin      = 3

# Calculate Lag Covariance for each case
covar_qnet  = scm.calc_leadlagcovar(qnet,sst,lags,monwin)
covar_rhflx = scm.calc_leadlagcovar(rhflx,sst,lags,monwin)
covar_thflx = scm.calc_leadlagcovar(thflx,sst,lags,monwin)
autocov_sst = scm.calc_leadlagcovar(sst,sst,lags,monwin)



covar_qnet_flip = scm.calc_leadlagcovar(sst,qnet,lags,monwin)
#%% Set up for analysis

incovars      = [covar_qnet,covar_rhflx,covar_thflx,autocov_sst]

expnames      = ["QNET","RHFLX","THFLX","SST"]
expnames_long = ["$Q_{net}$","$Q_{Radiative}$","$Q_{Turbulent}$"]

expcols       = ["midnightblue",'violet','firebrick']
expls         = ['dashdot','dashed','solid']

nvs = len(incovars)

#%% Point Analysis
lonf                    = -30
latf                    = 30
dspt                    = [proc.selpt_ds(ds,lonf,latf) for ds in incovars]

leadlags = dspt[0].lag
locfn,loctitle = proc.make_locstring(lonf,latf,fancy=True)

#%% Plot the Covariance...

kmonth = 1
fig,ax = plt.subplots(1,1,figsize=(6,4.5),constrained_layout=True)

for nv in range(nvs-1):
    
    plotvar = dspt[nv].isel(mon=kmonth)
    ax.plot(plotvar.lag,plotvar,
            label=expnames_long[nv],ls=expls[nv],c=expcols[nv],
            marker="o",fillstyle='none')
    
ax.legend()

#ax.set_xlim([leadlags[0],leadlags[-1]])


ax.set_xticks(leadlags)
ax.set_xlim([-6,6])
ax.axhline([0],lw=.75,c='k')
ax.axvline([0],lw=.75,c='k')

ax.set_ylabel("SST-FLX Covariance [$W/m^2  \,\degree C$]")
ax.set_xlabel(r" $\leftarrow$ SST Leads | Lags (Months) | FLX Leads $\rightarrow$")

# ax2 = ax.twinx()
# nv  = -1
# plotvar = dspt[nv].isel(mon=kmonth)
# ax2.plot(plotvar.lag,plotvar,label=expnames[nv],marker="o",fillstyle='none')

ax.set_title("ERA5 1979-2024, Covariance with SST @ %s" % loctitle)


#%% Compute some of the heat flux


hff_rhflx   = covar_rhflx / autocov_sst

hff_thflx   = covar_thflx / autocov_sst

hff_qnet    = covar_qnet  / autocov_sst

hffs        = [hff_qnet,hff_rhflx,hff_thflx]
hffsname    = expnames_long#["$Q_{net}$","$Q_{Radiative}$","$Q_{Turbulent}$"]

#%% Plot Lag 1 Radiative heat flux

im        = 1
il        = -1
vmax      = 25

#cints     = np.arange(-50,60,10)

bboxplot  = [-90,10,-10,65]
fsz_tick  = 18
fsz_title = 28
fsz_axis  = 22
mons3     = proc.get_monstr()

# Make the Plot
#for im in range(12):
    
for il in np.arange(-10,10):
    fig,axs,_ = viz.init_orthomap(1,3,bboxplot,figsize=(28,12.5))
    proj      = ccrs.PlateCarree()
    
    
    for ii in range(3):
        ax      = axs[ii]
        
        plotvar = hffs[ii].sel(lag=il+1,mon=im+1)
        
        pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                                transform=proj,vmin=-vmax,vmax=vmax,cmap='cmo.balance')
        
        cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,colors="yellow",
                                transform=proj,levels=[0,])
        ax.clabel(cl,fontsize=fsz_tick)
        
    
    for ii,ax in enumerate(axs):
        ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color='lightgray',fontsize=fsz_tick)
        ax.set_title("%s Feedback" % hffsname[ii],fontsize=fsz_title)
    
    cb = viz.hcbar(pcm,ax=axs.flatten(),fontsize=fsz_tick)
    cb.set_label(r"%s Heat Flux Feedback [$W / m^2$ per $\degree C$]" % (mons3[im]),fontsize=fsz_axis)
    
    
    
    figname = "%sERA5_HFF_Estimate_withENSO_lag%0i_mon%02i.png" % (figpath,il+1,im+1)
    plt.savefig(figname,dpi=150,bbox_inches='tight')
#plotvar = hff_qnet.sel(lag=il+1,mon=im+1)
#plotvar.plot(vmin=-15,vmax=15,cmap='cmo.balance')

#hrr_qnet  = 

#%% Check Qnet, what is the difference between leading 1 and lagging 1

lonf = -60
latf = 35
kmonth =1 

qnf_pt  = proc.selpt_ds(covar_qnet_flip,lonf,latf).isel(mon=kmonth)
qn_pt   = proc.selpt_ds(covar_qnet,lonf,latf).isel(mon=kmonth)


fig,ax = plt.subplots(1,1)
ax.plot(leadlags,qnf_pt,label='flip (v1=SST,v2=Q)')
ax.plot(leadlags,qn_pt * -1,label='raw (v2=Q,v2=SST)')
ax.legend()






#%% Look at the basic covariance







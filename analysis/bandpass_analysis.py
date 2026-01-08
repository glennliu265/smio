#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Trying to understand source of 2-year BP filtered Peak for SPGNE SST

Copied Upper section of construct_ENSO_forcing

Created on Fri Dec 26 15:41:14 2025

@author: gliu

"""




import sys
import time
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import xarray as xr
import sys
import glob 
import scipy as sp
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
from scipy.io import loadmat
import matplotlib as mpl

import importlib
from tqdm import tqdm

#%% Additional Modules

# local device (currently set to run on Astraeus, customize later)
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import cvd_utils as cvd

ensopath = "/Users/gliu/Downloads/02_Research/01_Projects/07_ENSO/03_Scripts/ensobase/"
sys.path.append(ensopath)
import utils as ut

# ========================================================
#%% User Edits
# ========================================================



# Load GMSST For ERA5 Detrending
detrend_obs_regression  = True
dpath_gmsst             = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
nc_gmsst                = "ERA5_GMSST_1979_2024.nc"

bbox_spgne              = [-40,-15,52,62]

# Set Figure Path
figpath  = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20251218/"
proc.makedir(figpath)


# ========================================================
#%% Load ERA5 and preprocess
# ========================================================
# Copied largely from area_average_variance_ERA5

# # Load SST
dpath_era   = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
ncname_era  = dpath_era + "ERA5_sst_NAtl_1979to2024.nc"
ds_era      = xr.open_dataset(ncname_era).sst.load()


#%%
# Detrend by Regression
dsa_era         = proc.xrdeseason(ds_era)
#flxa_era        = proc.xrdeseason(ds_era_flx)

# Detrend by Regression to the global Mean
ds_gmsst        = xr.open_dataset(dpath_era + nc_gmsst).GMSST_MaxIce.load()
dtout           = proc.detrend_by_regression(dsa_era,ds_gmsst,regress_monthly=True)
sst_era         = dtout.sst


sst_spgne       = proc.sel_region_xr(sst_era,bbox_spgne)
spgne_aavg      = proc.area_avg_cosweight(sst_spgne)

#%% Apply Bandpass Filter around 2 years
order      = 6
cutoff_upper = 1.5*12
cutoff_lower = 3*12
sst_lp       = proc.lp_butter(spgne_aavg,cutoff_lower,order)

sst_lp2      = proc.lp_butter(spgne_aavg,cutoff_upper,order)
sst_hp       = spgne_aavg.data - sst_lp2

sst_bp       = spgne_aavg.data - sst_lp - sst_hp


#%% Check Spectra


ssts_in  = [spgne_aavg.data,sst_lp,sst_hp,sst_bp]
specout  = scm.quick_spectrum(ssts_in,nsmooth=2,pct=0.10,return_dict=True)    


#%% Plot Spectra


def init_specplot(ax,decadal_focus=False,xlab=True,ylab=True,xperlab=True,
                  fsz_axis=14,fsz_ticks=14):
    
    if decadal_focus:
        xper            = np.array([20,10,7,5,3,2,1,0.5])
    else:
        xper            = np.array([40,20,10,7,5,3,2,1,0.5])

    xper_ticks      = 1 / (xper*12)
    dtmon_fix       = 60*60*24*30
    
    ax.set_xlim([xper_ticks[0],0.5])
    ax.axvline([1/(6)],label="",ls='dotted',c='gray')
    ax.axvline([1/(12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(2*12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(3*12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(5*12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(7*12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(10*12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(20*12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(40*12)],label="",ls='dotted',c='gray')
    ax.set_xscale('log')
    
    if xlab:
        ax.set_xlabel("Frequency (1/month)",fontsize=fsz_axis)
    if ylab:
        ax.set_ylabel(r"Power [$\degree C ^2 / cycle \, per \, mon$]",fontsize=fsz_axis)

    ax2 = ax.twiny()
    ax2.set_xlim([xper_ticks[0],0.5])
    ax2.set_xscale('log')
    ax2.set_xticks(xper_ticks,labels=xper,fontsize=fsz_ticks)
    if xperlab:
        ax2.set_xlabel("Period (Years)",fontsize=fsz_ticks)
    
    return ax,ax2

#%% Check Bandpass Filter Through Power through Analysis

fig,ax   = plt.subplots(1,1,layout='constrained',figsize=(12.5,4.5))
ax,_     = init_specplot(ax)

vnames_out = ["SST Raw","%.1f-year LP-filtered" % (cutoff_lower/12),
              "%.1f-year HP-filtered" % (cutoff_upper/12),
              "BP Filter Output"]
simcols  = ['k','midnightblue','hotpink','orange']

dtmon_fix       = 60*60*24*30
for ii in range(4):
    plotspec        = specout['specs'][ii] / dtmon_fix
    plotfreq        = specout['freqs'][ii] * dtmon_fix
    CCs             = specout['CCs'][ii] / dtmon_fix
    
    
    color_in = simcols[ii]
    label=vnames_out[ii]
    if ii == 0:
        ls = 'solid'#'dashed'
    else:
        ls = 'dashed'
    
    
    ax.loglog(plotfreq,plotspec,lw=4,label=label,c=color_in,ls=ls)
    
    # if len(plotspec) != len(plotfreq):
    #     if len(plotspec) < len(plotfreq):
    #         plotfreq = plotfreq[:-1]
    #     else:
    #         plotspec = plotspec[:-1]
    
    ax.loglog(plotfreq,CCs[:,0],ls='solid',lw=0.5,c=color_in)
    ax.loglog(plotfreq,CCs[:,1],ls='dashed',lw=0.9,c=color_in)
ax.legend()

figname = "%sSPGNE_Bandpass_output_%.1f_to_%.1f_years.png" % (figpath,cutoff_upper/12,cutoff_lower/12)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Just plot the timeseries to see


fig,ax = plt.subplots(layout='constrained',figsize=(10,4.5))

for ii in range(4):
    ax.plot(ssts_in[ii],label=vnames_out[ii],c=simcols[ii],lw=2.5)

ax = viz.add_axlines(ax)

ax.set_xlim([0,552])
ax.legend()
ax.set_xlabel("Time (Months)")
ax.set_ylabel("SST Anomaly [$\degree$ C]")

figname = "%sSPGNE_Timeseries_output_%.1f_to_%.1f_years.png" % (figpath,cutoff_upper/12,cutoff_lower/12)
plt.savefig(figname,dpi=150,bbox_inches='tight')



#%% Perform Lag Regression (just to NATL SST first
#% Lag Regression 2d

coords      = dict(time=ds_era.time,)
ts_in       = xr.DataArray(sst_bp,coords=coords,dims=coords,name='sst')
leadlags    = np.arange(-12,13,1)
llout       = ut.calc_leadlag_regression_2d(ts_in,sst_era,leadlags,sep_mon=False)

#%% PLot Lead Lag Regression

bbox_plot   = [-80,-0,0,65]
bbox_spgne  = [-40,-15,52,62]
proj        = ccrs.PlateCarree()
cints_rho   = np.arange(-1,1.1,0.1)

ii          = 0
for il in tqdm(range(len(leadlags))):
        
    #il          = 0
    lag         = leadlags[il]
    plotvar     = llout.sst.isel(lag=il)
    
    # Initialize Map
    fig,ax,_    = viz.init_orthomap(1,1,bbox_plot,figsize=(12.5,4.5))
    ax          = viz.add_coast_grid(ax,bbox=bbox_plot,proj=proj,fill_color='k')
    
    ax.set_title("Lag %02i" % (plotvar.lag.data),fontsize=18)
    
    pcm         = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                        cmap='cmo.balance',vmin=-1.5,vmax=1.5,transform=proj,)
    
    plotsig     = llout.sig.isel(lag=il)
    viz.plot_mask(plotsig.lon,plotsig.lat,plotsig.data.T,
                  reverse=False,proj=proj,geoaxes=True,ax=ax,markersize=.5,color='gray',marker='.')
    
    
    viz.plot_box(bbox_spgne,ax=ax,proj=proj,color="yellow",linewidth=1.5)
        
    
    
    figname = "%sSPGNE_BP_Lag_Regression_%.1f_to_%.1f_years_iter%02i_lag%02i.png" % (figpath,cutoff_upper/12,cutoff_lower/12,ii,lag)
    plt.savefig(figname,dpi=150,bbox_inches='tight')
    
    ii+=1
    
    

#%% Save Colorbar

    
# Initialize Map
fig,ax,_    = viz.init_orthomap(1,1,bbox_plot,figsize=(12.5,4.5))
ax          = viz.add_coast_grid(ax,bbox=bbox_plot,proj=proj,fill_color='k')



figname = "%sSPGNE_BP_Lag_Regression_%.1f_to_%.1f_years_colorbar.png" % (figpath,cutoff_upper/12,cutoff_lower/12)
plt.savefig(figname,dpi=150,bbox_inches='tight')

viz.hcbar(pcm,ax=ax,fraction=0.045)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Variance/Prsoperties of Area-Averaged SSTs for sets of observational SSTs

Created on Tue May 13 21:53:09 2025

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

#%% indicate cropping region and check for a folder

# Indicate Path to Area-Average Files
dpath_aavg      = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/region_average/"
regname         = "SPGNE"
bbsel           = [-40,-15,52,62]

# regname         = "NNAT"
# bbsel           = [-80,0,20,60]


# Set up Output Path
locfn,locstring = proc.make_locstring_bbox(bbsel,lon360=True)
bbfn            = "%s_%s" % (regname,locfn)
outpath         = "%s%s/" % (dpath_aavg,bbfn)
proc.makedir(outpath)
print("Region is        : %s" % bbfn)
print("Output Path is   : %s" % outpath)

# String Format is <bbfn>/<expname>_<vname>_<ystart>_<yend>_<procstring>.nc

#%% Indicate Which Experiments to Load

expnames = ["ERA5 (1979-2024)",
            "OISST (1982-2020)",
            "HadISST (1870-2024)",
            "COBE2 (1850-2024)",
            "ERSST (1854-2024)",
            "DCENT (1850-2023)",
            #"HadISST (1920-2017)",
            #"",
            #"ERSST5",
            #"EN4"
            ]

ncnames = [
    'ERA5_sst_1979_2024_raw_IceMask5.nc',
    'OISST_sst_1982_2020_raw_IceMaskMax5.nc',
    "HadISST_sst_1870_2024_raw.nc",
    "COBE2_sst_1850_2024_raw.nc",
    "ERSST5_sst_1854_2024_raw.nc",
    "DCENT_EnsMean_sst_1850_2023_raw.nc",
    #'HadISST_SST_1920_2017_detrend_deseason.nc',
    #'ERSST5_sst_1854_2017_raw.nc',
    #'EN4_sst_1900_2021_raw.nc'
    ]



vname = "sst"

expcols = ['dimgray','blue','red',
           'cyan','violet','yellow']

restrict_time = True
ystart = 1945
yend   = 2024
if restrict_time is not None:
    restrict_str  = "%04ito%04i" % (ystart,yend)
else:
    restrict_str = "FullPeriod"

nexps = len(expnames)


# Set Figure Path
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20250730/"
proc.makedir(figpath)

#%% Load the experiments

dsall = []
for ex in range(nexps):
    #ncsearch = "%s%s*.nc" % (outpath,expnames[ex])
    #nclist   = glob.glob(ncsearch)
    ds = xr.open_dataset(outpath + ncnames[ex]).load()[vname]
    
    if np.any(np.isnan(ds)):
        print("NaN detected in %s" % (expnames[ex]))
        
        print("Replacing with zero")
        ds = xr.where(np.isnan(ds),0,ds)
        
    if restrict_time:
        try:
            ds = ds.sel(time=slice('%04i-01-01' % ystart,"%04i-12-31" % yend))
        except: # Current doesn't work for ERSST?
            ds = ds.sel(time=slice(str(ystart),str(yend)))
        
        
    dsall.append(ds)


    


#%% Preprocess Timeseries

def preprocess_ds(ds):
    dsa      = proc.xrdeseason(ds)
    #dsa      = xr.where(np.isnan(dsa),0,dsa)
    #dsa_dt  = sp.signal.detrend(dsa)
    dsa_dt   = proc.xrdetrend_1d(dsa,3)
    return dsa_dt

dsa         = [preprocess_ds(ds) for ds in dsall]
stds        = [ds.std('time') for ds in dsa]
dsa_lp      = [proc.lp_butter(ds,120,6) for ds in dsa]
stds_lp     = [np.nanstd(ts) for ts in dsa_lp]

#%% Check Spread

instd       = stds
instd_lp    = stds_lp

vratio      = np.array(instd_lp) / np.array(instd) * 100

xlabs       = ["%s\n%.2f" % (expnames[ii],vratio[ii])+"%" for ii in range(len(vratio))]
fig,ax      = plt.subplots(1,1,constrained_layout=True,figsize=(6,6))

braw        = ax.bar(np.arange(nexps),instd,color=expcols)
blp         = ax.bar(np.arange(nexps),instd_lp,color='k')

ax.bar_label(braw,fmt="%.02f",c='gray')#color=expcols)#c='gray')
ax.bar_label(blp,fmt="%.02f",c='k')

ax.set_xticks(np.arange(nexps),labels=xlabs,fontsize=12,rotation=45)
ax.set_ylabel("$\sigma$(SST) [$\degree$C]")
ax.set_ylim([0,1.0])
#ax.set_title("%s (%s)" % (bbname,bbstr))

ax.grid(True,ls='dotted',lw=0.55,c='gray')


#%% Check the Re-emergence
lags        = np.arange(61)
ssts_in     = [ds.data for ds in dsa]

#nsmooths = [4,4,20,20,20,20]
nsmooths    = 4
tsm         = scm.compute_sm_metrics(ssts_in,lags=lags,nsmooth=nsmooths,detrend_acf=False)

#%% Plot the ACF

xtks    = lags[::6]
    
kmonth = 2
#for kmonth in range(12):

    
fig,ax = plt.subplots(1,1,figsize=(10,4.5),constrained_layout=True)

ax,_ = viz.init_acplot(kmonth,xtks,lags,ax=ax)
for ii in range(nexps):
    plotvar = tsm['acfs'][kmonth][ii]
    
    # Replace the Year with specified range if needed
    labin = expnames[ii]
    if restrict_time is not None:
        kdate = labin.index(" ")
        labin = labin.replace(labin[kdate:]," (%04i-%04i)" % (ystart,yend))
        
        
        
    ax.plot(lags,plotvar,label=labin,c=expcols[ii],lw=2.5)
ax.legend()

ax.set_ylim([-.5,1.1])
ax.axhline([0],c='k',lw=0.55)


savename = "%sACF_%s_mon%02i_%s.png" % (figpath,bbfn,kmonth+1,restrict_str)
plt.savefig(savename,dpi=150)


#%%


#%% Plot Spectra

dfcol       = "k"
transparent = True

metrics_out = tsm

dtmon_fix       = 60*60*24*30
xper            = np.array([40,10,5,1,0.5])
xper_ticks      = 1 / (xper*12)

    
fig,ax      = plt.subplots(1,1,figsize=(8,4.5),constrained_layout=True)

for ii in range(nexps):

    col_in = expcols[ii]
    plotspec        = metrics_out['specs'][ii] / dtmon_fix
    plotfreq        = metrics_out['freqs'][ii] * dtmon_fix
    CCs = metrics_out['CCs'][ii] / dtmon_fix

    ax.loglog(plotfreq,plotspec,lw=2.5,label=expnames[ii],c=col_in)
    
    #ax.loglog(plotfreq,CCs[:,0],ls='dotted',lw=0.5,c=expcols[ii])
    #ax.loglog(plotfreq,CCs[:,1],ls='dashed',lw=0.9,c=expcols[ii])

ax.set_xlim([1/1000,0.5])
ax.axvline([1/(6)],label="",ls='dotted',c='gray')
ax.axvline([1/(12)],label="",ls='dotted',c='gray')
ax.axvline([1/(5*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(10*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(40*12)],label="",ls='dotted',c='gray')

ax.set_xlabel("Frequency (1/Month)",fontsize=14)
ax.set_ylabel("Power [$\degree C ^2 cycle \, per \, mon$]")

ax2 = ax.twiny()
ax2.set_xlim([1/1000,0.5])
ax2.set_xscale('log')
ax2.set_xticks(xper_ticks,labels=xper)
ax2.set_xlabel("Period (Years)",fontsize=14)


# Plot Confidence Interval (ERA5)
alpha           = 0.05
cloc_era        = [plotfreq[0],1e-1]
dof_era         = metrics_out['dofs'][-1]
cbnds_era       = proc.calc_confspec(alpha,dof_era)
proc.plot_conflog(cloc_era,cbnds_era,ax=ax,color=dfcol,cflabel=r"95% Confidence") #+r" (dof= %.2f)" % dof_era)

ax.legend()

#figname = "%sSpectra_LogLog_Obs.png" % (figpath)
#plt.savefig(figname,dpi=150,bbox_inches='tight',transparent=transparent)


#%%
mons3 = proc.get_monstr()
fsz_legend=12
fsz_axis = 16
fsz_tick = 14

fig,ax = viz.init_monplot(1,1,figsize=(8,4.5))

for ex in range(nexps):
    
    col_in = expcols[ex]

        
    plotvar = metrics_out['monvars'][ex]
    
    ax.plot(mons3,plotvar,label=expnames[ex],c=col_in,lw=2.5)


ax.set_ylim([-.2,.5])
ax.set_ylabel("SST Variance [$\degree C^2$]",fontsize=fsz_axis)
ax.tick_params(labelsize=fsz_tick)
ax.legend(fontsize=fsz_legend,ncol=2)


#%% Plot just the timeseries



if restrict_time is False:
    
    labint  = 60
    plotx   = dsa[0].time.data
    xtk     = np.arange(len(plotx))
    plotx   = [str(x)[:7] for x in plotx]
    plotx   = plotx[::labint]


fig,ax = plt.subplots(1,1,figsize=(12.5,4.5))

for ex in range(nexps):
    
    plotvar = dsa[ex]
    if restrict_time:
        ax.plot(xtk,plotvar,label=expnames[ex],c=expcols[ex])
    
        
    else:
        ax.plot(plotvar,label=expnames[ex],c=expcols[ex])
    

if restrict_time:
    ax.set_xticks(xtk[::labint],labels=plotx)
ax.legend(ncol=4)
# savename = "%sMonvar_%s.png" % (figpath,comparename)
# if darkmode:
#     figname = proc.addstrtoext(figname,"_darkmode")
# plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=transparent)


#stds = [ds.std('tim')]



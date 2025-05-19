#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Analyze FLX-SST Lag Covariance

Works with output from 
- area_average_obs
- area_average_output

Created on Thu May  8 18:58:49 2025

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

#%% Load stuff for each variable

expnames = ["ERA5","CESM2_FCM","CESM2_POM","CESM2_SOM"]
sstnames = ["sst" ,"SST"      ,"TS"      ,"TS",]
flxnames = ["qnet","SHF"      ,"SHF"      , "LHFLX"     ]

nexps = len(expnames)

# Load GMSST For ERA5 Detrending
detrend_obs_regression = True
dpath_gmsst     = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
nc_gmsst        = "ERA5_GMSST_1979_2024.nc"


# Set Figure Path
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20250516/"
proc.makedir(figpath)


#%%  Load 


ssts = []
flxs = []
for ex in range(nexps):
    
    # Load SSTs
    ncsearch = "%s%s*%s*.nc" % (outpath,expnames[ex],sstnames[ex])
    ncin     = glob.glob(ncsearch)[0]
    print(ncin)
    ds       = xr.open_dataset(ncin)[sstnames[ex]].load()#.rename('sst')
    
    if np.all(ds > 250).data.item():
        ds = ds - 273.15
    if "ERA5" not in expnames[ex]:
        ds = proc.fix_febstart(ds)
        
    if "SOM" in expnames[ex]:
            
        print("Dropping first 60 years for SLAB simulation")
        ds = ds.sel(time=slice('0060-02-01','0361-01-01'))
    ssts.append(ds.copy())
    
    # # Load Fluxes
    # if ex != 3:
    #     ncsearch = "%s%s*%s*.nc" % (outpath,expnames[ex],flxnames[ex])
    #     ncin     = glob.glob(ncsearch)[0]
    #     print(ncin)
    #     ds       = xr.open_dataset(ncin)[flxnames[ex]].load()#.rename('sst')
    #     if "ERA5" not in expnames[ex]:
    #         ds = proc.fix_febstart(ds)
    #     flxs.append(ds.copy())
    

# # Also Load other fluxes for SOM
# vnames      = ["LHFLX","SHFLX","FLNS","FSNS"]
# flxcomp_som = []
# for vv in range(4):
#     ncsearch = "%s%s*%s*.nc" % (outpath,expnames[-1],vnames[vv])
#     ncin     = glob.glob(ncsearch)[0]
#     print(ncin)
#     ds       = xr.open_dataset(ncin)[vnames[vv]].load()#.rename('sst')
#     ds       = proc.fix_febstart(ds)
#     flxcomp_som.append(ds)


# flxcsom = xr.merge(flxcomp_som)

# flxs.append((-flxcsom.FSNS + (flxcsom.FLNS + flxcsom.LHFLX + flxcsom.SHFLX))*-1)

#%%
#flxs[-1] = flxs[-1] * -1

# ssts = [proc.fix_febstart(ds) for ds in ssts]
# flxs = [proc.fix_febstart(ds) for ds in flxs]
# flxcomp_som = [proc.fix_febstart(ds) for ds in flxcomp_som]

#%% Check the ssts (Seasonal Cycle)

mons3  = proc.get_monstr()
fig,axs = viz.init_monplot(1,2,figsize=(12.5,4.5))

for ex in range(nexps):
    ax = axs[0]
    sstin = ssts[ex].groupby('time.month').mean('time').data
    ax.plot(mons3,sstin,label=expnames[ex])
    ax.legend()
    ax.set_title("SST Seasonal Cycle")
    
    ax = axs[1]
    sstin = flxs[ex].groupby('time.month').mean('time').data
    if ex == 3:
        sstin = sstin #* -1
    ax.plot(mons3,sstin,label=expnames[ex])
    
    ax.legend()
    ax.set_title("Qnet Seasonal Cycle")


#%% Detrend and Deseason

def preproc_ds(ds):
    dsa   = proc.xrdeseason(ds)
    dsadt = proc.xrdetrend(dsa)
    return dsadt

#flxas = [preproc_ds(ds) for ds in flxs]
sstas = [preproc_ds(ds.squeeze()) for ds in ssts]

cutoff   = 5*12
nyrs     = int(cutoff/12)

#flxas_lp = [proc.lp_butter(ds,cutoff,6) for ds in flxas]
sstas_lp = [proc.lp_butter(ds,cutoff,6) for ds in sstas]


#%% Detrend/preprocess ERA5 Differently

if "ERA5" in expnames and (detrend_obs_regression):
    # Assume it is the first
    sst_era5    = ssts[0]
    ssta_era5   = proc.xrdeseason(sst_era5)
    sst_era     = ssta_era5.expand_dims(dim=dict(lon=1,lat=1))
    
    # Add dummy lat lon
    #sst_era['lon'] = 1
    #sst_era['lat'] = 1
    
    # Load GMSST
    ds_gmsst     = xr.open_dataset(dpath_gmsst + nc_gmsst).GMSST_MeanIce.load()
    dsdtera5     = proc.detrend_by_regression(sst_era,ds_gmsst)
    sst_era_dt   = dsdtera5.sst.squeeze()
    
    sstas[0] = sst_era_dt
    sstas_lp[0] = proc.lp_butter(sstas[0],cutoff,6)
    
    
    

#%% Compute Lag relationship


ii = 0
lags = np.arange(61)


llcorrs = []
llcorrs_lp = []
for ex in range(4):
    
    leadlags,leadlagcorr = proc.leadlag_corr(sstas[ex],flxas[ex],lags)
    llcorrs.append(leadlagcorr)

    _,leadlagcorr2 = proc.leadlag_corr(sstas_lp[ex],flxas_lp[ex],lags)
    llcorrs_lp.append(leadlagcorr2)
    
#%% Plot it

expcols = ["k","darkblue","goldenrod","red"]

lagmax=12
if lagmax == 12:
    xtks    = leadlags[::3]
else:
    xtks   =  leadlags[::12]
plot_ll = [llcorrs,llcorrs_lp]
titles  = ["Raw","%i-Year LP Filter" % nyrs]
fig,axs = plt.subplots(1,2,figsize=(10,4.5),constrained_layout=True)

for ii in range(2):
    ax = axs[ii]
    for ex in range(nexps):
        plotll = plot_ll[ii][ex] * -1
        ax.plot(leadlags,plotll,label=expnames[ex],c=expcols[ex],lw=2)
        
    
    ax.legend()
    ax.set_title(titles[ii])
    ax.axhline([0],c="k",ls='solid',lw=0.5)
    ax.axvline([0],c="k",ls='solid',lw=0.5)
    
    
    ax.set_xticks(xtks)
    ax.set_xlim([-lagmax,lagmax])
    ax.set_ylim([-.7,.7])
    
    ax.set_xlabel("<< Qnet Leads | SST Leads >>  ")
    ax.set_ylabel("Correlation")

#%% Make Barplot

stds    = np.array([np.nanstd(ts) for ts in sstas])
stds_lp = np.array([np.nanstd(ts) for ts in sstas_lp])

instd    = stds
instd_lp = stds_lp
expcols  = ['k','cornflowerblue','forestgreen','violet']
expcols_bar  = ['gray','cornflowerblue','forestgreen','violet']

vratio   = (stds_lp  / stds) * 100

xlabs           = ["%s\n%.2f" % (expnames[ii],vratio[ii])+"%" for ii in range(len(vratio))]
fig,ax          = plt.subplots(1,1,constrained_layout=True,figsize=(6,6))

braw            = ax.bar(np.arange(nexps),instd,color=expcols_bar)
blp             = ax.bar(np.arange(nexps),instd_lp,color='k')

ax.bar_label(braw,fmt="%.04f",c='gray')
ax.bar_label(blp,fmt="%.04f",c='k')

#ax.bar_label(vratio,fmt="%.2f",c=w,label_type='bottom')

ax.set_xticks(np.arange(nexps),labels=xlabs)
ax.set_ylabel("$\sigma$(SST) [$\degree$C]")
ax.set_ylim([0,1.0])

#%% Calculate Metrics

ssts          = [ds.data for ds in sstas]
lags          = np.arange(61)
nsmooths      = [4,250,100,100]
metrics_out   = scm.compute_sm_metrics(ssts,nsmooth=nsmooths,lags=lags)
expnames_long = expnames

#%% Plot Spectra
dtmon_fix       = 60*60*24*30
xper            = np.array([40,10,5,1,0.5])
xper_ticks      = 1 / (xper*12)

    
fig,ax      = plt.subplots(1,1,figsize=(8,4.5),constrained_layout=True)

for ii in range(nexps):
    plotspec        = metrics_out['specs'][ii] / dtmon_fix
    plotfreq        = metrics_out['freqs'][ii] * dtmon_fix
    CCs = metrics_out['CCs'][ii] / dtmon_fix

    ax.loglog(plotfreq,plotspec,lw=2.5,label=expnames_long[ii],c=expcols[ii])
    
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
proc.plot_conflog(cloc_era,cbnds_era,ax=ax,color='k',cflabel=r"95% Confidence") #+r" (dof= %.2f)" % dof_era)

ax.legend()

#figname = "%sSpectra_LogLog_%s.png" % (figpath,comparename)
#plt.savefig(figname,dpi=150,bbox_inches='tight')
#metrics_out = 

#%%

ntime_sample = len(sstas[0])
niter        = 1000
invar        = sstas[1].data
# def calc_std(ts):
#     ts = np.std(ts)
#     return 

def mciter_stdev(ntime_sample,invar,niter,return_idx=False):
    ntime         = len(invar)
    nstarts       = np.arange(ntime-ntime_sample+1)
    sample_starts = []
    sample_out    = []
    for it in tqdm.tqdm(range(niter)):
        
        # Get sample indices
        istart     = np.random.choice(nstarts)
        iend       = istart + ntime_sample
        selrange   = np.arange(istart,iend)
        sample_starts.append(istart)
        
        # 
        itsample    = invar[selrange]
        output      = np.nanstd(itsample)
        sample_out.append(output)
    if return_idx:
        return np.array(sample_out),np.array(sample_starts)
    return np.array(sample_out)


def get_conf(samples,alpha=0.05,mu=None):
    nsamples    = len(samples)
    pct         = alpha/2
    id_upper    = int((1 - pct) * nsamples)
    id_lower    = int(pct * nsamples)
    lower_bnd   = np.sort(samples)[id_lower]
    upper_bnd   = np.sort(samples)[id_upper]
    
    if mu is not None:
        upper_bnd = upper_bnd - mu
        lower_bnd = mu - lower_bnd
    
    return lower_bnd,upper_bnd
    
    

#%%
st = time.time()
stds_distr       = [mciter_stdev(ntime_sample,sst,10000) for sst in sstas[1:]]
stds_distr_lp    = [mciter_stdev(ntime_sample,sst,10000) for sst in sstas_lp[1:]]

confs_val         = np.array([get_conf(ss) for ss in stds_distr])
confs_val_lp      = np.array([get_conf(ss) for ss in stds_distr_lp])


confs            = np.array([get_conf(stds_distr[ii],mu=stds[ii+1]) for ii in range(3)])
confs_lp         = np.array([get_conf(stds_distr_lp[ii],mu=stds_lp[ii+1]) for  ii in range(3)])

# Set up confs
# confs       = (confs - stds[1:,None]).T
# confs_lp    = (confs_lp - stds_lp[1:,None]).T

#%% Check Distribution of Stdev

titles  = ["1$\sigma$","1$\sigma$ (LP-Filtered)"]


mcex     = 0
ex       = mcex + 1 # Add 1 since ERA5 was skipped
nbins  = 25
fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(6,8))

ax     = axs[0]

insample = stds_distr[mcex]
ax.hist(insample,bins=nbins,color=expcols[ex],edgecolor="w",linewidth=0.55)

# Plot 95% Confidence
lower,upper = get_conf(insample)
label = "95% Confidence: [" + "%.2f, %.2f]" % (lower,upper)
ax.axvline(lower,c='red',ls='dashed',label=label)
ax.axvline(upper,c='red',ls='dashed',label="")

# Plot Sample Mean
mu = np.nanmean(insample)
ax.axvline(mu,c='firebrick',ls='solid',label="Sample Mean: %.2f" % mu)

# Plot Actual Value
plotstd = stds[ex]
ax.axvline(plotstd,c='orange',ls='solid',label="Actual Value: %.2f" % plotstd)

#Plot ERA5
plotstd = stds[0]
ax.axvline(plotstd,c='k',ls='solid',label="ERA5: %.2f" % plotstd)


# LP Filtered =================================================================
ax     = axs[1]

insample = stds_distr_lp[mcex]
ax.hist(insample,bins=nbins,color=expcols[ex],edgecolor="w",linewidth=0.55)

# Plot 95% Confidence
lower,upper = get_conf(insample)
label = "95% Confidence: [" + "%.2f, %.2f]" % (lower,upper)
ax.axvline(lower,c='red',ls='dashed',label=label)
ax.axvline(upper,c='red',ls='dashed',label="")

# Plot Sample Mean
mu = np.nanmean(insample)
ax.axvline(mu,c='firebrick',ls='solid',label="Sample Mean: %.2f" % mu)

# Plot Actual Value
plotstd = stds_lp[ex]
ax.axvline(plotstd,c='orange',ls='solid',label="Actual Value: %.2f" % plotstd)


#Plot ERA5
plotstd = stds_lp[0]
ax.axvline(plotstd,c='k',ls='solid',label="ERA5: %.2f" % plotstd)

for a,ax in enumerate(axs):
    ax.set_xlim([0,1])
    ax.set_title(titles[a])
    ax.legend()
    ax.set_ylim([0,1200])

plt.suptitle("%s Stdev Distribution (%.0e Trials)" % (expnames[ex],1e4))

figname = "%s%s_MCiter1e5_Stdev_Distribution.png" % (figpath,expnames[ex])
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Replot the barplots, but with the confidence intervals



instd    = stds
instd_lp = stds_lp
expcols  = ['k','cornflowerblue','forestgreen','violet']
expcols_bar  = ['gray','cornflowerblue','forestgreen','violet']

vratio          = (stds_lp  / stds) * 100

xlabs           = ["%s\n%.2f" % (expnames[ii],vratio[ii])+"%" for ii in range(len(vratio))]
fig,ax          = plt.subplots(1,1,constrained_layout=True,figsize=(6,6))

braw            = ax.bar(np.arange(nexps),instd,color=expcols_bar)
blp             = ax.bar(np.arange(nexps),instd_lp,color='k')

ax.bar_label(braw,fmt="%.02f",c='gray')
ax.bar_label(blp,fmt="%.02f",c='k')

#ax.bar_label(vratio,fmt="%.2f",c=w,label_type='bottom')

ax.set_xticks(np.arange(nexps),labels=xlabs)
ax.set_ylabel("$\sigma$(SST) [$\degree$C]")
ax.set_ylim([0,1.0])

# plot Error Bars
ax.errorbar(np.arange(1,nexps)-.125,instd[1:],yerr=confs.T,
            linestyle="none",color='lightgray',marker="d",fillstyle='none')

ax.errorbar(np.arange(1,nexps)+.125,instd_lp[1:],yerr=confs_lp.T,
            linestyle="none",color='dimgray',marker="o",fillstyle='none')

# plot vertical lines for ERA5
ax.axhline([stds[0]],c='gray',lw=0.55,ls='solid')
ax.axhline([stds_lp[0]],c='dimgray',ls='solid',lw=0.55)


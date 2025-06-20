#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Looking at area average variance for CESM2 Hierarchy and ERA5

Copied upper section of flx_sst_lag_relationship

Created on Mon May 19 12:53:17 2025

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

# Indicate Paths (processed output by crop_natl_CESM2.py)
dpath           = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/NAtl/"

# Load GMSST For ERA5 Detrending
detrend_obs_regression = True
dpath_gmsst     = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
nc_gmsst        = "ERA5_GMSST_1979_2024.nc"

# Set Figure Path
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20250523/"
proc.makedir(figpath)

# For simplicity, load ice mask (can compare this later)
dsmask = dl.load_mask(expname='cesm2 pic').mask
dsmask180_cesm = proc.lon360to180_xr(dsmask)


#%% Functions

def calc_stds_sample(aavgs):
    
    aavgs_lp = [proc.lp_butter(aavg,120,6) for aavg in aavgs]
    stds     = np.array([np.nanstd(ss) for ss in aavgs])#np.nanstd(np.array(aavgs),1)
    stds_lp  = np.array([np.nanstd(ss) for ss in aavgs_lp])# np.nanstd(np.array(aavgs_lp),1)
    vratio   = stds_lp/stds * 100
    return aavgs_lp,stds,stds_lp,vratio

# ========================================================
#%% Load TS, CESM2
# ========================================================

st        = time.time()
cesmnames = ["CESM2_SOM","CESM2_POM3","CESM2_FCM"]
ncesm     = len(cesmnames)
dscesms   = []
dscesms_flxs = []
for vv in range(2):
    
    if vv == 0:
        vname = "TS"
    else:
        vname = "SHF"
    
    for cc in tqdm.tqdm(range(ncesm)):
        cname    = cesmnames[cc]
        ncsearch = "%s%s*_%s_*.nc" % (dpath,cname,vname)
        nclist   = glob.glob(ncsearch)
        print(nclist[0])
        
        ds       = xr.open_dataset(nclist[0])[vname]
        
        if "SOM" in cesmnames[cc]:   # Drop first 60 years
            ds = ds.sel(time=slice('0061-01-01',None))
        elif "POM" in cesmnames[cc]: # Drop first 100 years
            ds = ds.sel(time=slice('0100-01-01',None))
        elif "FCM" in cesmnames[cc]: # Drop first 200 years
            ds = ds.sel(time=slice('0200-01-01',None))
        
        if vv == 0:
            dscesms.append(ds.load())
        else:
            dscesms_flxs.append(ds.load())
             
    
#%% Do some preprocessing

def preproc_cesm(ds):
    ds      = proc.fix_febstart(ds)
    dsa     = proc.xrdeseason(ds)
    dsa_dt  = proc.xrdetrend(dsa)
    
    return dsa_dt

dscesms_proc        = [preproc_cesm(ds) for ds in dscesms]
dscesms_flxs_proc   = [preproc_cesm(ds) for ds in dscesms_flxs]

# NOTE the latitudes are different, need to be careful when doing computations
# (from area-average-variance-era5)
latpom  = dscesms_proc[1].lat
bbox_natproc = proc.get_bbox(dscesms_proc[0])
maskreg = proc.sel_region_xr(dsmask180_cesm,bbox_natproc)
latmask = maskreg.lat
if not np.all(latpom.data == latmask.data):
    print("Reassignining latitudes for pencil ocean model")
    dscesms_proc[1]['lat'] = latmask # Important, remap pencil ocean model to the right mask
    dscesms_flxs_proc[1]['lat'] = latmask

# Debug Plot, just to check they are different (variance)
#%Check Plot (debugging)
# fig,axs = plt.subplots(1,3)
# for cc in range(ncesm):
#     plotvar = dscesms_proc[cc].var('time') * dsmask180_cesm
#     pcm = axs[cc].pcolormesh(plotvar.lon,plotvar.lat,plotvar,vmin=0,vmax=1,cmap='cmo.balance')
#     viz.hcbar(pcm,ax=axs[cc])

# ========================================================
#%% Load ERA5 and preprocess
# ========================================================
# Copied largely from area_average_variance_ERA5

# Load SST
dpath_era   = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
ncname_era  = dpath_era + "ERA5_sst_NAtl_1979to2024.nc"
ds_era      = xr.open_dataset(ncname_era).sst.load()


# Load Flux
ncname_era_flx = dpath_era + "ERA5_qnet_NAtl_1979to2024.nc"
ds_era_flx = xr.open_dataset(ncname_era_flx).qnet.load()


# Load Mask
dsmask_era  = dl.load_mask(expname='ERA5')

#%% Load GMSST and also detrend pointwise

# Detrend by Regression
dsa_era       = proc.xrdeseason(ds_era)
flxa_era      = proc.xrdeseason(ds_era_flx)

# Detrend by Regression to the global Mean
ds_gmsst      = xr.open_dataset(dpath_era + nc_gmsst).GMSST_MeanIce.load()
dtout         = proc.detrend_by_regression(dsa_era,ds_gmsst)
sst_era       = dtout.sst
dtout_flx     = proc.detrend_by_regression(flxa_era,ds_gmsst)
flx_era       = dtout_flx.qnet

proc.printtime(st,print_str="Loaded and procesed data") #15.86s

# =======================================================
#%% Now Select Regions and make plots
# =======================================================

bbox_spgne  = [-40,-15,52,62]
bbox_nnat   = [-80,0,20,60]
bboxes      = [bbox_spgne,bbox_nnat]

bbnames           = ["SPGNE","NNAT"]
bbnames_long     = ["Northeastern Subpolar Gyre",
                    "Extratropical North Atlantic"]



dsall       =  dscesms_proc + [sst_era,]
expnames    =  cesmnames  + ["ERA5",]
maskall     = [dsmask180_cesm,]*3 + [dsmask_era.mask]


dsall_flx   = dscesms_flxs_proc + [flx_era,]


expnames_short = ["SOM","PenOM","FCM","ERA5"]
expnames_long  = ["Slab Ocean Model (60-360)","Pencil Ocean Model (100-400)","Fully Coupled Model (200-2000)","ERA5 (1979-2024)"]

nexps = len(expnames)
expcols         = ['violet','forestgreen','cornflowerblue','k']
expcols_bar     = ['violet','forestgreen','cornflowerblue','gray']


#%% Select Regions and Take area averages

lags        = np.arange(61)
nsmooths    = [100,100,250,4]


nregs               = len(bboxes)
aavgs_byreg         = []
metrics_byreg       = []
stds_metrics        = []

flx_aavgs_byreg     = []
stds_metrics_flx    = []
for rr in range(nregs):
    
    # Preprocess SSTs
    bbsel   = bboxes[rr]
    sstregs = [proc.sel_region_xr(dsall[ii] * maskall[ii],bbsel) for ii in range(nexps)]
    aavgs   = [proc.area_avg_cosweight(ds) for ds in sstregs]
    aavgs_byreg.append(aavgs)
    
    ssts    = [aa.data for aa in aavgs]
    tsms = scm.compute_sm_metrics(ssts,nsmooth=nsmooths,lags=lags)
    metrics_byreg.append(tsms)
    
    smms = calc_stds_sample(aavgs)
    stds_metrics.append(smms)
    
    
    # Preprocess Fluxes
    flxregs     = [proc.sel_region_xr(dsall_flx[ii] * maskall[ii],bbsel) for ii in range(nexps)]
    aavgs_flx   = [proc.area_avg_cosweight(ds) for ds in flxregs]
    flx_aavgs_byreg.append(aavgs_flx)
    
    smms_flx = calc_stds_sample(aavgs_flx)
    stds_metrics_flx.append(smms_flx)


#%% 


#%% Make the barplots

for rr in range(nregs):

    instd           = stds_metrics[rr][1]
    instd_lp        = stds_metrics[rr][2]
    vratio          = stds_metrics[rr][3]
    xlabs           = ["%s\n%.2f" % (expnames_short[ii],vratio[ii])+"%" for ii in range(len(vratio))]
    
    fig,ax          = plt.subplots(1,1,constrained_layout=True,figsize=(6,6))
    
    braw            = ax.bar(np.arange(nexps),instd,color=expcols_bar)
    blp             = ax.bar(np.arange(nexps),instd_lp,color='k')
    
    ax.bar_label(braw,fmt="%.02f",c='gray')
    ax.bar_label(blp,fmt="%.02f",c='k')
    
    #ax.bar_label(vratio,fmt="%.2f",c=w,label_type='bottom')
    
    ax.set_xticks(np.arange(nexps),labels=xlabs)
    ax.set_ylabel("$\sigma$(SST) [$\degree$C]")
    ax.set_ylim([0,1.0])
    
    ax.set_title(bbnames_long[rr])
    
    
    figname = "%s%s_NASST_AMV_Stdev.png" % (figpath,bbnames[rr])
    plt.savefig(figname,transparent=True,bbox_inches='tight')


#%% Make the Spectra Plots

rr = 0

ylim = [1e-3,1e2]
for rr in range(2):
    
    metrics_out = metrics_byreg[rr]
    
    dtmon_fix       = 60*60*24*30
    xper            = np.array([40,10,5,1,0.5])
    xper_ticks      = 1 / (xper*12)
    
        
    fig,ax      = plt.subplots(1,1,figsize=(8,4.5),constrained_layout=True)
    
    for ii in range(nexps):
        plotspec        = metrics_out['specs'][ii] / dtmon_fix
        plotfreq        = metrics_out['freqs'][ii] * dtmon_fix
        CCs             = metrics_out['CCs'][ii] / dtmon_fix
    
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
    
    ax.set_ylim(ylim)
    
    ax.legend()
    figname = "%s%s_NASST_Spectra.png" % (figpath,bbnames[rr])
    plt.savefig(figname,transparent=True,bbox_inches='tight')

#%% Investigate Flx-SST Lag Relationship
rr          = 0
monwin      = 3
lagcc       = np.arange(0,12,1)
dumll       = dict(lon=1,lat=1)
lpcutoff    = 60

calc_corr   = False


lpfilt      = lambda x: proc.lp_butter(x,lpcutoff,6)

covars      = []
covars_lp   = []
for ex in range(nexps):
    
    var1      = aavgs_byreg[rr][ex].expand_dims(dumll).transpose('time','lat','lon')
    var2      = flx_aavgs_byreg[rr][ex].expand_dims(dumll).transpose('time','lat','lon')
    out       = scm.calc_leadlagcovar(var1,var2,lagcc,monwin=monwin,calc_corr=calc_corr)
    
    covars.append(out.squeeze())
    
    var1_lp   = xr.apply_ufunc(lpfilt,var1,input_core_dims=[['time'],],output_core_dims=[['time'],],vectorize=True)
    var2_lp   = xr.apply_ufunc(lpfilt,var2,input_core_dims=[['time'],],output_core_dims=[['time'],],vectorize=True)
    
    out_lp    = scm.calc_leadlagcovar(var1_lp.transpose('time','lat','lon'),
                                      var2_lp.transpose('time','lat','lon'),
                                      lagcc,monwin=monwin,calc_corr=calc_corr)
    
    covars_lp.append(out_lp.squeeze())
    #var1_lp   = aavgs_byreg[rr][ex].expand_dims(dumll).transpose('time','lat','lon')

    
#%% Plot each case 

imon   = 6 # Indicatesthe Month
mons3  =  proc.get_monstr()

def get_monlag(basemonth,leadlags):
    mons3       = proc.get_monstr()
    nlags       = len(leadlags)
    leadlagmons = []
    for ll in range(nlags):
        # First determine the month
        #m  = basemonth + np.sign(leadlags[ll]) * (leadlags[ll]%12) 
        m  = basemonth + (leadlags[ll]%12) 
        im = m%12 - 1
        if im == (-1):
            im = 11
        leadlagmons.append(mons3[im])
    return leadlagmons


for imon in range(12):
    
    leadlags = covars[ex].lag.data
    lllab    = get_monlag(imon+1,leadlags) 
    nlags    = len(leadlags)
    fig,axs = plt.subplots(2,1,figsize=(8,6),constrained_layout=True)
    
    for ii,ax in enumerate(axs):
        
        if ii == 0:
            incov = covars
        else:
            incov = covars_lp
        
        for ex in range(nexps):
            # if ex >0:
            #     continue
            plotvar = incov[ex].isel(mon=imon)
            ax.plot(plotvar.lag.data,plotvar,c=expcols[ex],label=expnames_long[ex],marker='o')
        
        #ax.legend()
        
        ax.axhline([0],lw=0.75,c="k")
        ax.axvline([0],lw=0.75,c="k")
        
        # Label X Axis
        
        xlabels = ["%s\n%s" % (leadlags[ll],lllab[ll]) for ll in range(nlags)]
        ax.set_xticks(leadlags,labels=xlabels)
        
        
        if ii == 1:
            if rr == 0:
                ax.set_ylim([-3,3])
            else:
                ax.set_ylim([-.2,.2])
            ax.set_xlabel("SST Lag relative to Qnet (Months)\n  << SST Leads | Qnet Leads >>")
            
        else:
            ax.legend()
            if rr == 0:
                ax.set_ylim([-20,20])
            else:
                ax.set_ylim([-2,2])
                

            
            title = "%s Lag Covariance, %s" % (mons3[imon],bbnames[rr])
            ylab  = r"SST-Qnet Covariance [$\degree C \, \frac{W}{m^2}$]"
            calcname = "LagCovar"
            
        
        if calc_corr:
            title = "%s Lag Correlation, %s" % (mons3[imon],bbnames[rr])
            ylab  = r"SST-Qnet Cross-correlation [$\degree C \, \frac{W}{m^2}$]"
            ax.set_ylim([-1,1])
            calcname = "crosscorr"
            
        ax.set_title(title)
        ax.set_ylabel(ylab)
    
    figname = "%sFLX_SST_%s_CESM2Hierarchy_%s_lpf%i_mon%02i.png" % (figpath,calcname,bbnames[rr],lpcutoff,imon+1)
    plt.savefig(figname,dpi=150,bbox_inches='tight')
    
    
#%% Try all month case


def calc_leadlagcovar_allmon(var1,var2,lags,dim=0,return_da=True,ds_flag=False):
    
    if type(var1) == xr.DataArray:
        ds_flag = True
        lat     = var1.lat
        lon     = var1.lon
        var1    = var1.data
    if type(var2) == xr.DataArray:
        ds_flag = True
        lat     = var2.lat
        lon     = var2.lon
        var2    = var2.data
        
    # Assume time is in the first dimension
    lagcovar,winlens       = proc.calc_lag_covar_ann(var1,var2,lags,dim,0,)
    lagcovar_lead,_        = proc.calc_lag_covar_ann(var2,var1,lags,dim,0,)
    leadlags               = np.hstack([-1*np.flip(lags)[:-1],lags])
    
    # Flip Along Lead Dimension and drop Lead 0
    cov_lead_flip  = np.flip(lagcovar_lead[1:,:,:],0) 
    lagcovar_out   = np.concatenate([cov_lead_flip,lagcovar],axis=0)
    
    if return_da:
        if ds_flag:
            coords     = dict(lag=leadlags,lat=lat,lon=lon)
        else:
            coords = dict(lag=leadlags)
        da_out = xr.DataArray(lagcovar_out,coords=coords,dims=coords,name='cov')
        return da_out
    return lagcovar_out,leadlags
    
    
    
lagsall            = np.arange(61)

covars_allmon      = []
covars_allmon_lp   = []
for ex in range(nexps):
    
    var1      = aavgs_byreg[rr][ex].expand_dims(dumll).transpose('time','lat','lon')
    var2      = flx_aavgs_byreg[rr][ex].expand_dims(dumll).transpose('time','lat','lon')
    da_out    = calc_leadlagcovar_allmon(var1,var2,lagsall,dim=0,return_da=True)

    
    covars_allmon.append(da_out.squeeze())
    
    var1_lp   = xr.apply_ufunc(lpfilt,var1,input_core_dims=[['time'],],output_core_dims=[['time'],],vectorize=True)
    var2_lp   = xr.apply_ufunc(lpfilt,var2,input_core_dims=[['time'],],output_core_dims=[['time'],],vectorize=True)
    
    da_out_lp    = calc_leadlagcovar_allmon(var1_lp.transpose('time','lat','lon'),
                                      var2_lp.transpose('time','lat','lon'),
                                      lagsall)
    
    covars_allmon_lp.append(da_out_lp)
    
#%% Plot each case (all months)

mons3  =  proc.get_monstr()
calc_corr = False

def get_monlag(basemonth,leadlags):
    mons3       = proc.get_monstr()
    nlags       = len(leadlags)
    leadlagmons = []
    for ll in range(nlags):
        # First determine the month
        #m  = basemonth + np.sign(leadlags[ll]) * (leadlags[ll]%12) 
        m  = basemonth + (leadlags[ll]%12) 
        im = m%12 - 1
        if im == (-1):
            im = 11
        leadlagmons.append(mons3[im])
    return leadlagmons

    
leadlags = covars[ex].lag.data
lllab    = get_monlag(imon+1,leadlags) 
nlags    = len(leadlags)
fig,axs = plt.subplots(2,1,figsize=(8,6),constrained_layout=True)

for ii,ax in enumerate(axs):
    
    if ii == 0:
        incov = covars_allmon
    else:
        incov = covars_allmon_lp
    
    for ex in range(nexps):
        # if ex >0:
        #     continue
        plotvar = incov[ex].squeeze()#.isel(mon=imon)
        ax.plot(plotvar.lag.data,plotvar,c=expcols[ex],label=expnames_long[ex],marker='o')
    
    #ax.legend()
    
    ax.axhline([0],lw=0.75,c="k")
    ax.axvline([0],lw=0.75,c="k")
    
    # Label X Axis
    
    #xlabels = ["%s\n%s" % (leadlags[ll],lllab[ll]) for ll in range(nlags)]
    ax.set_xticks(leadlags,)#labels=xlabels)
    
    
    if ii == 1: # LP limits
        if rr == 0:
            ax.set_ylim([-1,1])
        else:
            ax.set_ylim([-1,1])
        ax.set_xlabel("SST Lag relative to Qnet (Months)\n  << SST Leads | Qnet Leads >>")
        
    else:
        ax.legend()
        if rr == 0:
            ax.set_ylim([-.5,.5])
        else:
            ax.set_ylim([-0.5,0.5])
            

        
        title = "All-Month Lag Covariance, %s" % (bbnames[rr])
        ylab  = r"SST-Qnet Covariance [$\degree C \, \frac{W}{m^2}$]"
        calcname = "LagCovar"
        
    
    if calc_corr:
        title = "All-Month Lag Correlation, %s" % (bbnames[rr])
        ylab  = r"SST-Qnet Cross-correlation [$\degree C \, \frac{W}{m^2}$]"
        ax.set_ylim([-1,1])
        calcname = "crosscorr"
        
    ax.set_title(title)
    ax.set_ylabel(ylab)
    
    ax.set_xlim([-11,11])

figname = "%sFLX_SST_%s_CESM2Hierarchy_%s_lpf%i_ALLMON.png" % (figpath,calcname,bbnames[rr],lpcutoff)
plt.savefig(figname,dpi=150,bbox_inches='tight')
    
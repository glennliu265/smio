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
import matplotlib.gridspec as gridspec

from scipy.io import loadmat
import matplotlib as mpl

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
dpath           = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/NAtl/Old/"

# Load GMSST For ERA5 Detrending
detrend_obs_regression = True
dpath_gmsst     = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
nc_gmsst        = "ERA5_GMSST_1979_2024.nc"

# Set Figure Path
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20250625/"
proc.makedir(figpath)

# For simplicity, load ice mask (can compare this later)
dsmask = dl.load_mask(expname='cesm2 pic').mask
dsmask180_cesm = proc.lon360to180_xr(dsmask)

#%%
# Set Plotting Options
darkmode = False
if darkmode:
    dfcol = "w"
    bgcol = np.array([15,15,15])/256
    sp_alpha = 0.05
    transparent = True
    plt.style.use('dark_background')
    mpl.rcParams['font.family']     = 'Avenir'
else:
    dfcol = "k"
    bgcol = "w"
    sp_alpha = 0.75
    transparent = False
    plt.style.use('default')
    


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
expnames_short_new = ["SOM","MCOM","FOM","ERA5"]
expnames_long  = ["Slab Ocean Model (60-360)","Multi-Column Ocean Model (100-400)","Full Ocean Model (200-2000)","ERA5 (1979-2024)"]

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

fsz_axis    = 18
fsz_ticks   = 16
label_ratio = False
for rr in range(nregs):

    instd           = stds_metrics[rr][1]
    instd_lp        = stds_metrics[rr][2]
    vratio          = stds_metrics[rr][3]
    if label_ratio:
        xlabs           = ["%s\n%.2f" % (expnames_short[ii],vratio[ii])+"%" for ii in range(len(vratio))]
    else:
        xlabs           = expnames_short
    
    fig,ax          = plt.subplots(1,1,constrained_layout=True,figsize=(6,6))
    
    braw            = ax.bar(np.arange(nexps),instd,color=expcols_bar)
    blp             = ax.bar(np.arange(nexps),instd_lp,color=dfcol)
    
    ax.bar_label(braw,fmt="%.02f",c='gray',fontsize=fsz_axis)
    ax.bar_label(blp,fmt="%.02f",c=dfcol,fontsize=fsz_axis)
    
    #ax.bar_label(vratio,fmt="%.2f",c=w,label_type='bottom')
    
    ax.set_xticks(np.arange(nexps),labels=xlabs)
    ax.set_ylabel("$\sigma$(SST) [$\degree$C]",fontsize=fsz_axis)
    ax.set_ylim([0,1.0])
    
    ax.set_title(bbnames_long[rr],fontsize=fsz_axis)
    
    ax.tick_params(labelsize=fsz_ticks)
    
    figname = "%s%s_NASST_AMV_Stdev.png" % (figpath,bbnames[rr])
    if darkmode:
        figname = proc.darkname(figname)
        
    plt.savefig(figname,bbox_inches='tight',transparent=transparent,dpi=150)


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
    
    
        color_in = expcols[ii]
        if color_in == "k" and darkmode:
            color_in = dfcol
        ax.loglog(plotfreq,plotspec,lw=2.5,label=expnames_long[ii],c=color_in)
        
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
    
    ax.set_ylim(ylim)
    
    ax.legend()
    figname = "%s%s_NASST_Spectra.png" % (figpath,bbnames[rr])
    if darkmode:
        figname = proc.darkname(figname)
    plt.savefig(figname,transparent=transparent,bbox_inches='tight',dpi=150)
    
# ======================
#%% Make the Combined Barplot and Spectra for Paper Outline
# ======================


remove_topright=True



rr              = 0
metrics_out     = metrics_byreg[rr]

fig             = plt.figure(figsize=(16,4))
gs              = gridspec.GridSpec(4,12)

# --------------------------------- # Barplot
ax11            = fig.add_subplot(gs[:,:3],)
ax              = ax11


fsz_axis        = 18
fsz_ticks       = 16
fsz_title       = 26
fsz_legend      = 14
label_ratio     = False

instd           = stds_metrics[rr][1]
instd_lp        = stds_metrics[rr][2]
vratio          = stds_metrics[rr][3]
if label_ratio:
    xlabs           = ["%s\n%.2f" % (expnames_short[ii],vratio[ii])+"%" for ii in range(len(vratio))]
else:
    xlabs           = expnames_short_new


braw            = ax.bar(np.arange(nexps),instd,color=expcols_bar)
blp             = ax.bar(np.arange(nexps),instd_lp,color=dfcol)

ax.bar_label(braw,fmt="%.02f",c='gray',fontsize=fsz_ticks)
ax.bar_label(blp,fmt="%.02f",c=dfcol,fontsize=fsz_ticks)

ax.set_xticks(np.arange(nexps),labels=xlabs,rotation=45)
ax.set_ylabel("$\sigma$(SST) [$\degree$C]",fontsize=fsz_axis)
ax.set_ylim([0,1.25])

ax.tick_params(labelsize=fsz_ticks)

viz.label_sp(0,alpha=0.15,ax=ax,fontsize=fsz_title,y=1.17,x=-.3,
             fontcolor=dfcol)

colorsf = {'Raw':'gray','10-year Low-pass':'k',}         
labels = list(colorsf.keys())
handles = [plt.Rectangle((0,0),1,1, color=colorsf[label]) for label in labels]
ax.legend(handles, labels,fontsize=fsz_legend,framealpha=0)


if remove_topright:
    ax.spines[['right', 'top']].set_visible(False)
    
    
# --------------------------------- # Power Spectra
ax22       = fig.add_subplot(gs[:,4:])

obs_cutoff = 10 # in years
obs_cutoff = 1/(obs_cutoff*12)

ax         = ax22

decadal_focus = False
if decadal_focus:
    xper            = np.array([20,10,5,1,0.5])
else:
    xper            = np.array([40,10,5,1,0.5])
xper_ticks      = 1 / (xper*12)

dtmon_fix       = 60*60*24*30

for ii in range(nexps):
    plotspec        = metrics_out['specs'][ii] / dtmon_fix
    plotfreq        = metrics_out['freqs'][ii] * dtmon_fix
    CCs             = metrics_out['CCs'][ii] / dtmon_fix


    color_in = expcols[ii]
    if color_in == "k" and darkmode:
        color_in = dfcol
    
    if ii == 3:
        
        iplot_hifreq = np.where(plotfreq > obs_cutoff)[0]
        ax.loglog(plotfreq,plotspec,label="",c=color_in,ls='dashed',lw=1.5)
        plotfreq     = plotfreq[iplot_hifreq]
        plotspec     = plotspec[iplot_hifreq]
        
        ax.loglog(plotfreq,plotspec,lw=4,label=expnames_long[ii],c=color_in)
        
    else:
        
        ax.loglog(plotfreq,plotspec,lw=4,label=expnames_long[ii],c=color_in)
    
    #ax.loglog(plotfreq,CCs[:,0],ls='dotted',lw=0.5,c=expcols[ii])
    #ax.loglog(plotfreq,CCs[:,1],ls='dashed',lw=0.9,c=expcols[ii])

ax.set_xlim([xper_ticks[0],0.5])
ax.axvline([1/(6)],label="",ls='dotted',c='gray')
ax.axvline([1/(12)],label="",ls='dotted',c='gray')
ax.axvline([1/(5*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(10*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(20*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(40*12)],label="",ls='dotted',c='gray')

ax.set_xlabel("Frequency (1/month)",fontsize=fsz_axis)
ax.set_ylabel("Power [$\degree C ^2 / cycle \, per \, mon$]",fontsize=fsz_axis)

ax2 = ax.twiny()
ax2.set_xlim([xper_ticks[0],0.5])
ax2.set_xscale('log')
ax2.set_xticks(xper_ticks,labels=xper,fontsize=fsz_ticks)
ax2.set_xlabel("Period (Years)",fontsize=fsz_ticks)

# Plot Confidence Interval (ERA5)
alpha           = 0.05
cloc_era        = [8e-2,1e-2]
dof_era         = metrics_out['dofs'][-1]
cbnds_era       = proc.calc_confspec(alpha,dof_era)
proc.plot_conflog(cloc_era,cbnds_era,ax=ax,color=dfcol,cflabel=r"95% Confidence") #+r" (dof= %.2f)" % dof_era)

ax.set_ylim(ylim)

ax.legend(fontsize=fsz_legend,framealpha=0.5,edgecolor='none')

for ax in [ax,ax2]:
    ax.tick_params(labelsize=fsz_ticks)
    
viz.label_sp(1,alpha=0.15,ax=ax,fontsize=fsz_title,y=1.17,x=-.1,
             fontcolor=dfcol)


figname = "%s%s_NASST_Spectra_Barplot.png" % (figpath,bbnames[rr])
if darkmode:
    figname = proc.darkname(figname)
plt.savefig(figname,transparent=transparent,bbox_inches='tight',dpi=150)

#%% Investigate Flx-SST Lag Relationship
rr          = 0
monwin      = 3
lagcc       = np.arange(0,12,1)
dumll       = dict(lon=1,lat=1)
lpcutoff    = 60

calc_corr   = True


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


def calc_leadlagcovar_allmon(var1,var2,lags,dim=0,return_da=True,ds_flag=False,calc_corr=False):
    
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

# ------
#%% Maybe Try to Reproduce the O'Reilly et al 2016 Calculations using Annual Averages
rr = 1

def calc_ann_avg(ds):
    return ds.groupby('time.year').mean('time')

def movmean(ds,win):
    return np.convolve(ds.data,np.ones(win)/win,mode='same')
    
win = 11

sst_ann     = [calc_ann_avg(ds) for ds in aavgs_byreg[rr]]
sst_ann_lp  = [movmean(ds,win) for ds in sst_ann]

flx_ann     = [calc_ann_avg(ds) for ds in flx_aavgs_byreg[rr]]
flx_ann_lp  = [movmean(ds,win) for ds in flx_ann]

#%% Compute Lag-Relationships with annual data

movmean10 = lambda x: movmean(x,win)

lagsall_ann     = np.arange(24)
corrs_ann      = []
corrs_ann_lp   = []
corrs_ann_hp   = []
for ex in range(nexps):
    
    # Compute for Raw
    var1    = sst_ann[ex].expand_dims(dumll).transpose('year','lat','lon')
    var2    = flx_ann[ex].expand_dims(dumll).transpose('year','lat','lon')
    da_out  = calc_leadlagcovar_allmon(var1,
                                       var2,
                                      lagsall_ann)
    corrs_ann.append(da_out.squeeze())
    
    # Compute for LPFiltered
    var1lp   = xr.apply_ufunc(movmean10,var1,input_core_dims=[['year'],],output_core_dims=[['year'],],vectorize=True)
    var2lp   = xr.apply_ufunc(movmean10,var2,input_core_dims=[['year'],],output_core_dims=[['year'],],vectorize=True)
    da_out_lp  = calc_leadlagcovar_allmon(var1lp.transpose('year','lat','lon'),
                                          var2lp.transpose('year','lat','lon'),
                                          lagsall_ann)
    corrs_ann_lp.append(da_out_lp.squeeze())
    
    
    # Compute for HPFiltered
    var1hp   = var1 - var1lp
    var2hp   = var2 - var2lp
    da_out_hp  = calc_leadlagcovar_allmon(var1hp.transpose('year','lat','lon'),
                                          var2hp.transpose('year','lat','lon'),
                                          lagsall_ann)
    corrs_ann_hp.append(da_out_hp.squeeze())
    

    
#%% Plot the Results

xtks = np.arange(-10,11,1)

fig,axs = plt.subplots(1,2,figsize=(12.5,4.5),constrained_layout=True)

for aa in range(2):
    
    ax = axs[aa]
    if aa == 1:
        corrin = corrs_ann_hp
        title  = "Short-Term Component"
    else:
        corrin = corrs_ann_lp
        title  = "%i-Year Running Mean" % win
        
        
    
    for ex in range(nexps):
        plotcorr = corrin[ex]
        ax.plot(plotcorr.lag,plotcorr,c=expcols[ex],label=expnames_long[ex],marker='o')
        
    ax.legend()
    ax.set_title(title)
    
    ax.axhline([0],lw=0.75,c="k")
    ax.axvline([0],lw=0.75,c="k")
        
    ax.set_xlim([-10,10])
    ax.set_xticks(xtks)
        



#%% Check the monthly phasing of sea surface temperature variance

rr            = 1
plot_variance = True

monvars       =  [ ds.groupby('time.month').var('time') for ds in aavgs_byreg[rr]]
monstds       =  [ ds.groupby('time.month').std('time') for ds in aavgs_byreg[rr]]


fig,ax        = viz.init_monplot(1,1,figsize=(8,4.5))

for ex in range(nexps):
    
    if plot_variance:
        label="Monthly Variance [$\degree C^2$]"
        analysisname = "var"
        plotvar = monvars[ex]
    else:
        label="Monthly Standard Deviation [$\degree C$]"
        analysisname = "std"
        plotvar = monstds[ex]
    
    ax.plot(mons3,plotvar,c=expcols[ex],label=expnames_long[ex],marker='o')
    
ax.set_ylabel(label)
ax.legend()
    
ax.set_ylim([0,0.50])
ax.set_xlabel("Month")

figname = '%sCESM2_Hierarchy_Monthly_%s_%s.png' % (figpath,analysisname,bbnames[rr])
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Apply Pointwise Lowpass Filter

lpfilter10 = lambda x: proc.lp_butter(x,120,6)
dsall_lpf  = []
for ex in tqdm.tqdm(range(nexps)):
    st = time.time()
    dslpf = xr.apply_ufunc(
        lpfilter10,
        dsall[ex],
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True,
        )
    print("LPF in  in %.2fs" % (time.time()-st))
    dsall_lpf.append(dslpf)
    
    
stdall_lp  = [ds.std('time') for ds in dsall_lpf]
stdall_reg = [ds.std('time') for ds in dsall]

#%% Check Variance Ratio (PenOM / FCM)

bboxplot = [-80,0,0,65]
fsz_tick = 24
fsz_title =35
cints    = [0.80,1,1.20]

fig,axs,_ = viz.init_orthomap(1,2,bboxplot,figsize=(28,12.5))
proj      = ccrs.PlateCarree()
    

for ii in range(2):
    ax = axs[ii]
    if ii == 0:
        plotvar = stdall_reg[1] / stdall_reg[2]
        title = ""
    else:
        plotvar = stdall_lp[1] / stdall_lp[2]
        title = "Low-Pass Filtered"
    
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                        transform=proj,vmin=0,vmax=2,cmap='cmo.balance')
    
    cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                        transform=proj,levels=cints,colors="k")
    ax.clabel(cl,fontsize=fsz_tick)
    
    ax.set_title(r"%s Ratio ($\frac{\sigma(PenOM)}{\sigma(FOM)}$)" % (title),fontsize=fsz_title)

    
for ii,ax in enumerate(axs):
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color='lightgray',fontsize=fsz_tick)
    #ax.set_title("%s Feedback" % hffsname[ii],fontsize=fsz_title)

cb = viz.hcbar(pcm,ax=axs.flatten(),fontsize=fsz_tick)

#%% Slab vs PenOM


bboxplot  = [-80,0,0,65]
fsz_tick  = 24
fsz_title = 35
cints     = np.arange(0,2.1,0.2)#[0.80,1,1.20]

fig,axs,_ = viz.init_orthomap(1,2,bboxplot,figsize=(28,12.5))
proj      = ccrs.PlateCarree()

for ii in range(2):
    ax = axs[ii]
    if ii == 0:
        plotvar = stdall_reg[1] / stdall_reg[0]
        title = ""
    else:
        plotvar = stdall_lp[1] / stdall_lp[0]
        title = "Low-Pass Filtered"
    
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                        transform=proj,vmin=0,vmax=2,cmap='cmo.balance')
    
    cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                        transform=proj,levels=cints,colors="k")
    ax.clabel(cl,fontsize=fsz_tick)
    
    ax.set_title(r"%s Ratio ($\frac{\sigma(PenOM)}{\sigma(SLAB)}$)" % (title),fontsize=fsz_title)

    
for ii,ax in enumerate(axs):
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color='lightgray',fontsize=fsz_tick)
    #ax.set_title("%s Feedback" % hffsname[ii],fontsize=fsz_title)

cb = viz.hcbar(pcm,ax=axs.flatten(),fontsize=fsz_tick)


#%%

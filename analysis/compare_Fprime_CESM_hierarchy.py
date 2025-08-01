#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Copied upper section of calc_Fprime_CESM2_hierarchy (also uses output from that script)


Created on Tue Jul  8 11:25:56 2025

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

#%% Settings

# Calculation Options
expnames        = ["SOM","MCOM","FOM"]
sstnames        = ["TS","TS","TS"]
flxnames        = ["SHF","SHF","SHF"]
expnames_long   = ["Slab","Multi-Column","Full"]
expyears        = ["60-360","100-500","200-1800"]
dpath           = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/NAtl/"

tstarts         = ['0060-01-01','0100-01-01','0200-01-01']
tends           = ['0360-12-31','0500-12-31','2000-12-31']

figpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20250710/"
proc.makedir(figpath)

#%% Some helper functions

def load_cesm2(expname,sstname): # Search and load for a file
    ncsearch = dpath + "CESM2_%s_%s_*.nc" % (expname,sstname) 
    ncname   = glob.glob(ncsearch)
    if len(ncname) > 1:
        print("Warning, more than 1 file found...")
        print(ncname)
    elif len(ncname) == 0:
        print("No files found for %s (%s)...." % (expname,sstname) )
    return xr.open_dataset(ncname[0]).load()

def preproc(ds,tstart,tend):
    dsa = proc.xrdeseason(ds)
    dsadt = proc.xrdetrend(dsa)
    dsadt = dsadt.sel(time=slice(tstart,tend))
    return dsadt



#%% Load the SST, Qnet, Fprime for CESM2 Hierarchy

nexps   = len(expnames)

bbox_natl = [-80,0,0,65]

vars_byexp = []
for ex in tqdm.tqdm(range(nexps)):
    
    # Load SST
    ds_sst = load_cesm2(expnames[ex],sstnames[ex])
    ds_sst = preproc(ds_sst[sstnames[ex]],tstarts[ex],tends[ex])
    
    # Load FLX
    ds_flx = load_cesm2(expnames[ex],flxnames[ex])
    # Check if upwards Positive for Flux
    bbox_gs     = [-80,-60,20,40]
    ds_flx      = proc.check_flx(ds_flx,flxname=flxnames[ex],bbox_gs=bbox_gs)
    ds_flx      = preproc(ds_flx[flxnames[ex]],tstarts[ex],tends[ex])
    
    # Load Fprime 
    ds_fprime = load_cesm2(expnames[ex],"Fprime_upwards")
    ds_fprime = preproc(ds_fprime['Fprime'],tstarts[ex],tends[ex])
    
    # Merge and append
    ds_exp = xr.merge([ds_sst,ds_flx,ds_fprime],join='override')
    ds_exp = proc.sel_region_xr(ds_exp,bbox_natl)
    vars_byexp.append(ds_exp)

#%% Also load no ENSO case
enso_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/anom/"

enso_ncs = ["CESM2_SOM_SHF_NAtl_0060to0360_detrend1_ENSOrem_lag1_pcs3_monwin3.nc",
            "CESM2_MCOM_SHF_NAtl_0100to0500_detrend1_ENSOrem_lag1_pcs3_monwin3.nc",
            "CESM2_FOM_SHF_NAtl_0200to2000_detrend1_ENSOrem_lag1_pcs3_monwin3.nc"]

flx_noenso = [xr.open_dataset(enso_path + nc).load() for nc in enso_ncs]
#"CESM2_SOM_SHF_detrend1_ENSOcmp_lag1_pcs3_monwin3_0060to0360.nc"
#"CESM2_MCOM_SHF_detrend1_ENSOcmp_lag1_pcs3_monwin3_0100to0500.nc"
#"CESM2_FOM_SHF_detrend1_ENSOcmp_lag1_pcs3_monwin3_0200to2000.nc"


# Preprocess this
ds_flx_noenso      = [proc.check_flx(flx_noenso[ex],flxname=flxnames[ex],bbox_gs=bbox_gs) for ex in range(3)]
ds_flx_noenso      = [preproc(ds_flx_noenso[ex][flxnames[ex]],tstarts[ex],tends[ex]) for ex in range(3)]


#%% Load Ice Mask for analysis

maskpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/masks/"
masknc    = "cesm2_pic_limask_0.3p_0.05p_0200to2000.nc"
dsmask    = xr.open_dataset(maskpath+masknc).load()
dsmask180 = proc.lon360to180_xr(dsmask)
bbreg     = proc.get_bbox(ds_sst)
dsmask180 = proc.sel_region_xr(dsmask180,bbreg).mask


#%% Also open view of ERA5 for analysis (this might get a bit large...)

ncflx_era       = "ERA5_qnet_NAtl_1979to2024.nc"
pathflx_era     = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
flx_era         = xr.open_dataset(pathflx_era+ncflx_era)['qnet']

ncfp_era        = "ERA5_Fprime_QNET_timeseries_QNETpilotObsAConly_nroll0_NAtl.nc"
pathfp_era      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/reanalysis/proc/NATL_proc_obs/"
fprime_era      = xr.open_dataset(pathfp_era+ncfp_era)['Fprime']#.load()

ncsst_era       = "ERA5_sst_NAtl_1979to2024.nc"
pathsst_era     = pathflx_era
sst_era         = xr.open_dataset(pathsst_era+ncsst_era)['sst']

# Load GMSST For ERA5 Detrending
detrend_obs_regression = True
dpath_gmsst     = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
nc_gmsst        = "ERA5_GMSST_1979_2024.nc"
ds_gmsst        = xr.open_dataset(dpath_gmsst + nc_gmsst).GMSST_MaxIce.load()

#%% Load Lbd_a for conversion/amplitude comparison

hffpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/hff/"
ncs_lbda = [
    "CESM2_SOM_SHF_damping_NAtl_0060to0360_ensorem1_detrend1.nc",
    "CESM2_MCOM_SHF_damping_NAtl_0100to0500_ensorem1_detrend1.nc",
    "CESM2_FOM_SHF_damping_NAtl_0200to2000_ensorem1_detrend1.nc",
    ]

dshff = [xr.open_dataset(hffpath+nc).load() for nc in ncs_lbda]

def preproc_hff(hffin,sstreg_in,bbsel):
    hffreg      = proc.sel_region_xr(hffin.SHF_damping.isel(lag=0),bbsel)
    ntime,_,_   = sstreg_in.shape
    nyrs        = int(ntime/12)
    hfftile     = np.tile(hffreg.data.transpose(1,2,0),nyrs).transpose(2,0,1)
    coordsin    = dict(time=sstreg_in.time,lat=hffreg.lat,lon=hffreg.lon)
    hfftile     = xr.DataArray(hfftile,dims=coordsin,coords=coordsin,name='SHF_damping')
    return hfftile
    #hff_aavg = proc.area_avg_cosweight(hff)
    
    
    

# ============================
#%% Part (1): Point Analysis
# ============================


# ============================
#%% Try Area-Average Analysis?
# ============================

bbsel        = [-40,-15,52,62]
#bbsel = [-60,-20,40,60]
bbfn,bbtitle = proc.make_locstring_bbox(bbsel)


dsreg_byvar = []
aavg_byvar  = []
for ex in range(nexps):
    dsreg           = proc.sel_region_xr(vars_byexp[ex],bbsel)
    dsreg['TS_lbda'] = preproc_hff(dshff[ex],dsreg.TS,bbsel)
    dsreg_mask  = dsreg * dsmask180
    aavg        = proc.area_avg_cosweight(dsreg_mask)
    aavg_byvar.append(aavg)
    dsreg_byvar.append(dsreg)

#% Process and compute ERA5 =======

def preproc_era(ds,ds_gmsst):
    dsa   = proc.xrdeseason(ds)
    dtout = proc.detrend_by_regression(dsa,ds_gmsst)
    return dtout[str(dsa.name)]
    

era_vars     = [sst_era,flx_era,fprime_era]
st           = time.time()
era_reg      = [proc.sel_region_xr(ds,bbsel).load() for ds in era_vars]
proc.printtime(st)
eraanom      = [preproc_era(ds.squeeze(),ds_gmsst) for ds in era_reg]
eraanom      = [proc.area_avg_cosweight(ds) for ds in eraanom]

aavg_era = xr.merge([eraanom[0].rename("TS"),
                     eraanom[1].rename("SHF"),
                     eraanom[2]],join='override')

aavg_byvar.append(aavg_era)

#%%  Check the Monthly Variance

monvars     = [aavgs.groupby('time.month').var('time') for aavgs in aavg_byvar]

vnames      = ["TS","SHF","Fprime"]
#vnames      = ["TS_lbda","SHF","Fprime"]
vnames_plot = ["SST","Net Heat Flux","Stochastic Heat Flux"]
vunits      = [r"\degree C",r"\frac{W}{m^2}",r"\frac{W}{m^2}"]
mons3       = proc.get_monstr()
expcols     = ['violet','forestgreen','cornflowerblue','k']
expcols_bar = ['violet','forestgreen','cornflowerblue','gray']

expnames_plot  = expnames_long + ["ERA5",]
nexps_loop          = nexps + 1

sstnames = sstnames + ["TS",]
flxnames = flxnames + ["SHF",]

fig,axs     = viz.init_monplot(3,1,figsize=(6,10))


for vv in range(3):
    ax = axs[vv]
    ax.set_ylabel(u"Var(%s) [$%s ^2$]" % (vnames_plot[vv],vunits[vv]))
    for ex in range(nexps_loop):
        
        plotvar = monvars[ex][vnames[vv]]
        ax.plot(mons3,plotvar,c=expcols[ex],label=expnames_plot[ex],lw=2.5,marker="o")
    
    if vv == 0:
        ax.legend()
        
        
figname = "%sMonthly_Variance_CESM2_Hierarchy_%s.png" % (figpath,bbfn)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Check the overall variance of each variable

nexps_loop = nexps+1
expnames_plot  = expnames_long + ["ERA5",]

fsz_axis = 14
ssts    = [aavg_byvar[ex][sstnames[ex]].data for ex in range(nexps_loop)]
flxs    = [aavg_byvar[ex][flxnames[ex]].data for ex in range(nexps_loop)]
fprimes = [aavg_byvar[ex]['Fprime'].data for ex in range(nexps_loop)]

ssts_conv =  [aavg_byvar[ex].TS_lbda.data for ex in range(nexps_loop-1)]


plot_conv = False

def lpfilter_ignorenan(ts):
    if np.any(np.isnan(ts)):
        print("Warning, NaNs present. Filling with zero")
        ts[np.isnan(ts)] = 0.
    return proc.lp_butter(ts,120,order=6)

if plot_conv:
    invars = [ssts_conv,flxs[:-1],fprimes[:-1]]
    nexps_loop = nexps
    expnames_plot = expnames_plot[:-1]
    
else:
    invars = [ssts,flxs,fprimes]
    nexps_loop = nexps + 1
    
fig,axs = plt.subplots(3,1,figsize=(6,10))

for vv in range(3):
    ax       = axs[vv]
    
    invar    = invars[vv]
    stds     = [np.nanstd(ts) for ts in invar]
    invar_lp = [lpfilter_ignorenan(ts) for ts in invar]
    stds_lp  = [np.nanstd(ts) for ts in invar_lp]
    
    
    braw            = ax.bar(np.arange(nexps_loop),stds,color=expcols_bar)
    blp             = ax.bar(np.arange(nexps_loop),stds_lp,color='k')
    
    ax.bar_label(braw,fmt="%.02f",c='gray',fontsize=fsz_axis)
    ax.bar_label(blp,fmt="%.02f",c='k',fontsize=fsz_axis)
    
    
    ax.set_xticks(np.arange(nexps_loop),labels=expnames_plot,)
    ax.set_ylabel(u"Stdev(%s) [$%s$]" % (vnames_plot[vv],vunits[vv]))
    
    if vv == 0 and plot_conv is False:
        ax.set_ylim([0,1.0])
    else:
        ax.set_ylim([0,50])
        
figname = "%sArea_Avg_Stdev_CESM2_Hierarchy_%s.png" % (figpath,bbfn)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Power Spectra

nsmooths       = [20,50,150,6]
lags           = np.arange(61)
metrics_sst    = scm.compute_sm_metrics(ssts,nsmooth=nsmooths,lags=lags)
metrics_flx    = scm.compute_sm_metrics(flxs,nsmooth=nsmooths,lags=lags)
metrics_fprime = scm.compute_sm_metrics(fprimes,nsmooth=nsmooths,lags=lags)

inmetrics      = [metrics_sst,metrics_flx,metrics_fprime]

#%% Plot the power spectra for eacn one


dtmon_fix       = 60*60*24*30
xper            = np.array([40,10,5,1,0.5])
xper_ticks      = 1 / (xper*12)

#dof_dummy = [3,300,3000]#[38.882768453113854, 95.73408899440915, 285.2811815853462]
invars          = [ssts,flxs,fprimes]
fig,axs         = plt.subplots(3,1,figsize=(6,10),constrained_layout=True)

for vv in range(3):
    ax          = axs[vv]
    invar       = invars[vv]
    
    metrics_out = inmetrics[vv]
    
    # if vv == 0:
    #     cfloc = 1e-1
    # else:
    #     cfloc = 1e3
    
    ax = axs[vv]
    #fig,ax      = plt.subplots(1,1,figsize=(8,4.5),constrained_layout=True)
    
    for ii in range(nexps_loop):
        plotspec        = metrics_out['specs'][ii] / dtmon_fix
        plotfreq        = metrics_out['freqs'][ii] * dtmon_fix
        CCs             = metrics_out['CCs'][ii] / dtmon_fix
    
    
        color_in = expcols[ii]
        ax.loglog(plotfreq,plotspec,lw=2.5,label=expnames_plot[ii],c=color_in,marker="x",markersize=2.5)
        
        # # Plot Confidence Interval (ERA5)
        # idof            = ii
        # alpha           = 0.05
        # cloc_era        = [1e-2+(ii*.25e-2),cfloc]
        # dof_fom         = metrics_out['dofs'][idof] # dof_dummy[ii] #
        # cbnds_era       = proc.calc_confspec(alpha,dof_fom)
        # proc.plot_conflog(cloc_era,cbnds_era,ax=ax,color=expcols[idof],cflabel=r"95% Confidence") #+r" (dof= %.2f)" % dof_era)
        
        
        
        
        #ax.loglog(plotfreq,CCs[:,0],ls='dotted',lw=0.5,c=expcols[ii])
        #ax.loglog(plotfreq,CCs[:,1],ls='dashed',lw=0.9,c=expcols[ii])
    
    ax.set_xlim([1/1000,0.5])
    ax.axvline([1/(6)],label="",ls='dotted',c='gray')
    ax.axvline([1/(12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(5*12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(10*12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(40*12)],label="",ls='dotted',c='gray')
    
    
    ax.set_ylabel("Power [$%s ^2 cycle \, per \, mon$]" % vunits[vv])
    
    ax2 = ax.twiny()
    ax2.set_xlim([1/1000,0.5])
    ax2.set_xscale('log')
    ax2.set_xticks(xper_ticks,labels=xper)
    
    
    if vv == 0:
        ax.set_ylim([1e-3,1e2])
        ax2.set_xlabel("Period (Years)",fontsize=14)
        cfloc = 1e-1
    else:
        ax.set_ylim([1e1,1e4])
        cfloc = 1e3
    
    if vv == 2:
        ax.set_xlabel("Frequency (1/Month)",fontsize=14)
        ax.legend()
        
    # # Plot Confidence Interval (ERA5)
    # idof            = 2
    # alpha           = 0.05
    # cloc_era        = [1e-2,cfloc]
    # dof_fom         = metrics_out['dofs'][idof]
    # cbnds_era       = proc.calc_confspec(alpha,dof_fom)
    # proc.plot_conflog(cloc_era,cbnds_era,ax=ax,color=expcols[idof],cflabel=r"95% Confidence") #+r" (dof= %.2f)" % dof_era)
    
        

figname = "%sPower_Spectra_CESM2_Hierarchy_%s.png" % (figpath,bbfn)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Do Monte Carlo Testing for a selected spectra

sample_len = len(flxs[-1])
niter      = 10000
spec_byexp = []

start_jan = False #  Current issue with jan starts.... (exceed indices)

nsmooth_era =6
spec_byexp = []
for ex in range(nexps):
    
    
    tsfull = flxs[ex]
    if start_jan: 
        nstarts = np.arange(0,len(tsfull)-int(sample_len/12)-12,12)
    else:
        nstarts = np.arange(0,len(tsfull)-sample_len-1)
    
    
    spec_all = []
    
    istarts = np.random.choice(nstarts,niter)
    for mc in tqdm.tqdm(range(niter)):
        
        seg_id = np.arange(istarts[mc],istarts[mc]+sample_len)
        seg    = tsfull[seg_id]
        mout = scm.quick_spectrum([seg],nsmooth_era,0.10,return_dict=True)
        spec_all.append(mout['specs'][0])
        
        #mout   = scm.compute_sm_metrics([seg],nsmooth=nsmooth_era)
        #spec_all.append(mout['specs'][0])
    
    spec_byexp.append(np.array(spec_all))

#%% Plot same as above but with error bars

def get95conf(samples,axis):
    return np.quantile(samples,[0.025,0.975],axis=axis)

plot_separate = True

# Select variable to plot
invars      = [ssts,flxs,fprimes]
vv          = 1
metrics_out = inmetrics[vv]

if not plot_separate:
    fig,ax = plt.subplots(1,1,figsize=(10,4.5),constrained_layout=True)

    for ii in range(nexps_loop):
        plotspec = metrics_out['specs'][ii] / dtmon_fix
        plotfreq = metrics_out['freqs'][ii] * dtmon_fix
        CCs      = metrics_out['CCs'][ii] / dtmon_fix
        
        color_in = expcols[ii]
        
        if ii <  3:
            inconf          = spec_byexp[ii].squeeze()
            conf95          = get95conf(inconf,0) / dtmon_fix
            conffreq        = metrics_out['freqs'][-1] * dtmon_fix
            muconf          = inconf.mean(0) / dtmon_fix
            ax.plot(conffreq,muconf,c=color_in,marker="x",markersize=2.5,ls="dashed")
            ax.fill_between(conffreq,conf95[0,:],conf95[1,:],alpha=0.2,color=color_in,zorder=-1)
        
        
        ax.loglog(plotfreq,plotspec,lw=2.5,label=expnames_plot[ii],c=color_in,marker="x",markersize=2.5)
        
        
        
        # # Plot Confidence Interval (ERA5)
        # idof            = ii
        # alpha           = 0.05
        # cloc_era        = [1e-2+(ii*.25e-2),cfloc]
        # dof_fom         = metrics_out['dofs'][idof] # dof_dummy[ii] #
        # cbnds_era       = proc.calc_confspec(alpha,dof_fom)
        # proc.plot_conflog(cloc_era,cbnds_era,ax=ax,color=expcols[idof],cflabel=r"95% Confidence") #+r" (dof= %.2f)" % dof_era)
        
        
        
        
        #ax.loglog(plotfreq,CCs[:,0],ls='dotted',lw=0.5,c=expcols[ii])
        #ax.loglog(plotfreq,CCs[:,1],ls='dashed',lw=0.9,c=expcols[ii])
    
    ax.set_xlim([1/1000,0.5])
    ax.axvline([1/(6)],label="",ls='dotted',c='gray')
    ax.axvline([1/(12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(5*12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(10*12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(40*12)],label="",ls='dotted',c='gray')
    
    
    ax.set_ylabel("Power [$%s ^2 cycle \, per \, mon$]" % vunits[vv])
    
    ax2 = ax.twiny()
    ax2.set_xlim([1/1000,0.5])
    ax2.set_xscale('log')
    ax2.set_xticks(xper_ticks,labels=xper)
    
    
    if vv == 0:
        ax.set_ylim([1e-3,1e2])
        ax2.set_xlabel("Period (Years)",fontsize=14)
        cfloc = 1e-1
    else:
        ax.set_ylim([1e2,1e4])
        cfloc = 1e3
    
    if vv == 2:
        ax.set_xlabel("Frequency (1/Month)",fontsize=14)
        ax.legend()
    
    # Plot Confidence Interval (ERA5)
    idof            = 2
    alpha           = 0.05
    cloc_era        = [1e-2,cfloc]
    dof_fom         = metrics_out['dofs'][idof]
    cbnds_era       = proc.calc_confspec(alpha,dof_fom)
    proc.plot_conflog(cloc_era,cbnds_era,ax=ax,color=expcols[idof],cflabel=r"95% Confidence") #+r" (dof= %.2f)" % dof_era)
    
    
    figname = "%sPower_Spectra_Conf_CESM2_Hierarchy_%s.png" % (figpath,bbfn)
    plt.savefig(figname,dpi=150,bbox_inches='tight')
else:
    print("Plotting_Separately)")
    always_plot_ex = [0,3]
    loop_ex        = [1,2]
    
    for li in range(len(loop_ex)): # Initialize plot for each experiment
    
        ii_targ         = loop_ex[li]
        loop_indices    = always_plot_ex + [ii_targ,]
        
    
        fig,ax = plt.subplots(1,1,figsize=(10,4.5),constrained_layout=True)
        
        for nn in range(len(loop_indices)):
            ii       = loop_indices[nn]
            
            plotspec = metrics_out['specs'][ii] / dtmon_fix
            plotfreq = metrics_out['freqs'][ii] * dtmon_fix
            CCs      = metrics_out['CCs'][ii] / dtmon_fix
            
            color_in = expcols[ii]
            
            
                
            if ii < 3: # Plot Confidence Intervals
                inconf          = spec_byexp[ii].squeeze()
                conf95          = get95conf(inconf,0) / dtmon_fix
                conffreq        = metrics_out['freqs'][-1] * dtmon_fix
                muconf          = inconf.mean(0) / dtmon_fix
                ax.plot(conffreq,muconf,c=color_in,ls="dashed")
                #if ii ==
                
                ax.plot(conffreq,conf95[0,:],color=color_in,zorder=-1,ls='dashed',alpha=0.5,lw=.5)
                ax.plot(conffreq,conf95[1,:],color=color_in,zorder=-1,ls='dashed',alpha=0.5,lw=.5)
                ax.fill_between(conffreq,conf95[0,:],conf95[1,:],alpha=0.1,color=color_in,zorder=-1)
                
                
            ax.loglog(plotfreq,plotspec,lw=2.5,label=expnames_plot[ii],c=color_in)
            
            
        
        
        ax.set_xlim([1/1000,0.5])
        ax.axvline([1/(6)],label="",ls='dotted',c='gray')
        ax.axvline([1/(12)],label="",ls='dotted',c='gray')
        ax.axvline([1/(5*12)],label="",ls='dotted',c='gray')
        ax.axvline([1/(10*12)],label="",ls='dotted',c='gray')
        ax.axvline([1/(40*12)],label="",ls='dotted',c='gray')
    
    
        ax.set_ylabel("Power [$%s ^2 cycle \, per \, mon$]" % vunits[vv])
        
        ax2 = ax.twiny()
        ax2.set_xlim([1/1000,0.5])
        ax2.set_xscale('log')
        ax2.set_xticks(xper_ticks,labels=xper)
        
    
        #if vv == 1:
        ax.set_ylim([1e-3,1e2])
        ax2.set_xlabel("Period (Years)",fontsize=14)
        cfloc = 1e-1
        
        ax.set_ylim([1e2,1e5])
        cfloc = 1e3
        
        ax.set_xlabel("Frequency (1/Month)",fontsize=14)
        
        
                
        # Plot Confidence Interval (ERA5)
        idof            = 3
        alpha           = 0.05
        cloc_era        = [5e-3,2e4]
        dof_fom         = metrics_out['dofs'][idof]
        cbnds_era       = proc.calc_confspec(alpha,dof_fom)
        proc.plot_conflog(cloc_era,cbnds_era,ax=ax,color=expcols[idof],cflabel=r"95% Confidence (ERA5)") #+r" (dof= %.2f)" % dof_era)
        
        ax.legend()
        
        
        figname = "%sPower_Spectra_Conf_CESM2_Hierarchy_%s_loop%i.png" % (figpath,bbfn,li)
        plt.savefig(figname,dpi=150,bbox_inches='tight')
        
    



#%% ===========================================================================


#%% Briefly Check the Persistence

kmonth = 1
xtks   = lags[::3]

for kmonth in range(12):
    fig,axs = plt.subplots(3,1,constrained_layout=True,figsize=(8,10))
    
    for vv in range(3):
        ax = axs[vv]
        ax,_   = viz.init_acplot(kmonth,xtks,lags,ax=ax,title="")
        
        for ex in range(nexps_loop):
            
            plotvar = inmetrics[vv]['acfs'][kmonth][ex]
            
            ax.plot(lags,plotvar,
                    label=expnames_plot[ex],color=expcols[ex],lw=2.5)
        
        ax.set_ylabel("%s %s\nAutocorrelation" % (mons3[kmonth],vnames[vv]))
        ax.set_ylim([-.25,1.25])
        ax.axhline([0],ls='solid',lw=.55,c="k")
        
        if vv == 2:
            ax.legend()
            ax.set_xlabel("Correlation with %s Anomalies" % mons3[kmonth])
        else:
            ax.set_xlabel("")
        
        #plt.suptitle("%s Autocorrelation" % )
    
    
    figname = "%sACF_CESM2_Hierarchy_%s_mon%02i.png" % (figpath,bbfn,kmonth+1)
    plt.savefig(figname,dpi=150,bbox_inches='tight')

# ============================
#%% Part (2.5) : Lag Correlation Analysis
# ============================
# Copied from cesm2_hierarchy_v_obs

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

def calc_ann_avg(ds):
    return ds.groupby('time.year').mean('time')

def movmean(ds,win):
    lpts = np.convolve(ds.data,np.ones(win)/win,mode='same')
    lpts = xr.DataArray(lpts,coords=ds.coords,dims=ds.dims,name=ds.name)
    return lpts


def get_leadlags(sstin,flxin):
    
    dumll       = dict(lon=1,lat=1)
    
    movmean10 = lambda x: movmean(x,win)

    lagsall_ann     = np.arange(24)
    
    # Compute 
    var1    = sstin.expand_dims(dumll).transpose('year','lat','lon')
    var2    = flxin.expand_dims(dumll).transpose('year','lat','lon')
    da_out  = calc_leadlagcovar_allmon(var1,
                                       var2,
                                      lagsall_ann)
    return da_out.squeeze()





#%% Calculate Annual Average

win          = 11
ann_byvar    = []
ann_byvar_lp = []
ann_byvar_hp = []
for vv in range(3):
    vname     = vnames[vv]
    dsin      = [ds[vname] for ds in aavg_byvar]
    dsann     = [calc_ann_avg(ds) for ds in dsin]
    dsann_lp  = [movmean(ds,win) for ds in dsann]
    dsann_hp  = [dsann[ex] - dsann_lp[ex] for ex in range(4)]
    
    ann_byvar.append(dsann)
    ann_byvar_lp.append(dsann_lp)
    ann_byvar_hp.append(dsann_hp)
    


leadlags_qnet       = [get_leadlags(ann_byvar[1][ex],ann_byvar[0][ex]) for ex in range(4)]
leadlags_qnet_lp    = [get_leadlags(ann_byvar_lp[1][ex],ann_byvar_lp[0][ex]) for ex in range(4)]
leadlags_qnet_hp    = [get_leadlags(ann_byvar_hp[1][ex],ann_byvar_hp[0][ex]) for ex in range(4)]

leadlags_fprime     = [get_leadlags(ann_byvar[2][ex],ann_byvar[0][ex]) for ex in range(4)]
leadlags_fprime_lp  = [get_leadlags(ann_byvar_lp[2][ex],ann_byvar_lp[0][ex]) for ex in range(4)]
leadlags_fprime_hp  = [get_leadlags(ann_byvar_hp[2][ex],ann_byvar_hp[0][ex]) for ex in range(4)]

#%% Make Some Plots

usehp   = True
varplot = "Qnet"#"Fprime"#
xtks    = np.arange(-10,11,1)

fig,axs = plt.subplots(1,2,figsize=(12.5,4.5),constrained_layout=True)

for aa in range(2):
    
    ax = axs[aa]
    
    if aa == 0:
        if varplot == "Fprime":
            if usehp:
                corrin = leadlags_fprime_hp
            else:
                corrin = leadlags_fprime
        elif varplot == "Qnet":
            if usehp:
                corrin = leadlags_qnet_hp
            else:
                corrin = leadlags_qnet
        #corrin = leadlags_qnet
        
        title  = "Raw"
    else:
        if varplot == "Fprime":
            corrin = leadlags_fprime_lp
        else:
            corrin = leadlags_qnet_lp
            
        title  = "%i-Year Running Mean" % win
        
        
    
    for ex in range(nexps_loop):

        plotcorr = corrin[ex]
        
        plotcorr = plotcorr #* -1
        # if ex < 3:
        #     plotcorr = plotcorr * -1
        ax.plot(plotcorr.lag,plotcorr,c=expcols[ex],label=expnames_plot[ex],marker='o')
        
    ax.legend()
    ax.set_title(title)
    
    ax.axhline([0],lw=0.75,c="k")
    ax.axvline([0],lw=0.75,c="k")
        
    ax.set_xlim([-10,10])
    ax.set_xticks(xtks)
        

figname = "%sLeadLagCorr_CESM2_Hierarchy_%s_%s_usehp%i.png" % (figpath,bbfn,varplot,usehp)
plt.savefig(figname,dpi=150,bbox_inches='tight')


#%% [2025.07.24] Check Where Low Frequency Variability is weak for Qnet

qnets_cesm = [ds.SHF for ds in dsreg_byvar]
qnets_cesm = [proc.xrdeseason(ds) for ds in qnets_cesm]

qnets_reg = qnets_cesm + [preproc_era(era_reg[2].isel(ens=0).squeeze(),ds_gmsst)]
qnet_aavg = [proc.area_avg_cosweight(ds) for ds in qnets_reg]

def movmean_arr(ds,win):
    lpts = np.convolve(ds,np.ones(win)/win,mode='same')
    #lpts = xr.DataArray(lpts,coords=ds.coords,dims=ds.dims,name=ds.name)
    return lpts

def calc_slope_lf(x):
    #ts10  = movmean_arr(x,12*10)
    #ts40  = movmean_arr(x,12*40)
    ts10 = proc.lp_butter(x,12*10,6)
    ts40 = proc.lp_butter(x,12*0,6)
    var10 = np.var(ts10)
    var40 = np.var(ts40)
    return var40/var10

vratios40_all = []
for ii in range(4):
    ds = qnets_reg[ii]
    vratios40 = xr.apply_ufunc(
        calc_slope_lf,
        ds,
        input_core_dims=[['time']],
        output_core_dims=[[]],
        vectorize=True,
        )
    vratios40_all.append(vratios40)


#%% Plot the Ratios

fsz_tick = 36
proj     = ccrs.PlateCarree()




for ex in range(4):
    plotvar = vratios40_all[ex]
    fig,ax,_ = viz.init_regplot(regname="SPGE")
    pcm=ax.pcolormesh(plotvar.lon,plotvar.lat,np.log(plotvar),transform=proj,
                      cmap='cmo.balance')
    
    cl = ax.contour(plotvar.lon,plotvar.lat,np.log(plotvar),levels=[0,],transform=proj,
                      colors="k")
    cb = fig.colorbar(pcm,ax=ax)
    cb.ax.tick_params(labelsize=fsz_tick)
    ax.set_title(expnames_plot[ex],fontsize=fsz_tick)
    
    

#%% Compare with values from the area-averaged timeseries


vratios_aavg = [np.log(calc_slope_lf(ds)) for ds in flxs]

#%% Maybe just try the spectrum?


def pointwise_spectra(tsens,nsmooth=1, opt=1, dt=None, clvl=[.95], pct=0.1):
    calc_spectra = lambda x: proc.point_spectra(x,nsmooth=nsmooth,opt=opt,
                                                dt=dt,clvl=clvl,pct=pct)
    
    # Change NaN to Zeros for now
    tsens_nonan = xr.where(np.isnan(tsens),0,tsens)
    
    # Compute Spectra
    specens = xr.apply_ufunc(
        proc.point_spectra,  # Pass the function
        tsens_nonan,  # The inputs in order that is expected
        # Which dimensions to operate over for each argument...
        input_core_dims=[['time'],],
        output_core_dims=[['freq'],],  # Output Dimension
        exclude_dims=set(("freq",)),
        vectorize=True,  # True to loop over non-core dims
    )
    
    # # Need to Reassign Freq as this dimension is not recorded
    #ts1  = tsens.isel(ens=0).values
    freq            = proc.get_freqdim(tsens)
    #specens['freq'] = freq
    return specens,freq # Note.. freq outpout is not working

spec_allpt = []
freq_allpt = []
nsmooths       = [20,50,150,6]
for ii in range(4):
    
    specexp,freq = pointwise_spectra(qnets_reg[ii],nsmooth=nsmooths[ii])
    spec_allpt.append(specexp)
    freq_allpt.append(freq)

# Reassign frequencies manually from an earlier calculation
# (calc freq_dim does not appear to be working)
freqs  = [metrics_flx['freqs'][ex] for ex in range(4)]

for ii in range(4):
    
    spec_allpt[ii]['freq'] = freqs[ii]
    
    #test = test.swap_dims(dict(freq=dict(freq:freq_allpt[0])))
    #test = test.rename(dict(freq="freqcount"))
    
    
da_freq = [ds.freq for ds in spec_allpt]

#%%

def getfreq(yr):
    dtyr  = 3600*24*30*12
    return 1/(dtyr*yr)


def get_id_xr(ds,dim,val):
    return ds.indexes[dim].get_loc(val, method="nearest")

# def get_id_xr(ds,dim,val):
#     return ds[dim].get_indexer(val, method="nearest")

#id49 = [get_id_xr(ds,'freq',nsec40) for ds in da_freq]
    
dtmon           = 3600*24*30
nsec40          = getfreq(40)
nsec10          = getfreq(10)
id40            = [get_id_xr(ds,'freq',nsec40) for ds in da_freq] # Get the Index using # sec
p40_aavg        = [np.array(metrics_flx['specs'][ex][id40[ex]])/dtmon for ex in range(4)] # Check power at that level

id10            = [get_id_xr(ds,'freq',nsec10) for ds in da_freq]       
p10_aavg        = [np.array(metrics_flx['specs'][ex][id10[ex]])/dtmon for ex in range(4)]

power40_aavg    = [ds.sel(freq=nsec40,method='nearest')/dtmon for ds in spec_allpt] # 
power10_aavg    = [ds.sel(freq=nsec10,method='nearest')/dtmon for ds in spec_allpt] # 

power_diff      = [power40_aavg[ex] - power10_aavg[ex] for ex in range(4)]

#%% Check power decrease by region

fsz_tick    = 36
fsz_title   = 48
proj        = ccrs.PlateCarree()

for ex in range(4):
    
    plotvar  = power_diff[ex]
    fig,ax,_ = viz.init_regplot(regname="SPGE")
    pcm      = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                             cmap='cmo.balance',vmin=-10000,vmax=10000)
    
    # cl = ax.contour(plotvar.lon,plotvar.lat,np.log(plotvar),levels=[0,],transform=proj,
    #                   colors="k")
    cb = fig.colorbar(pcm,ax=ax)
    cb.ax.tick_params(labelsize=fsz_tick)
    ax.set_title(expnames_plot[ex],fontsize=fsz_title)


#%% Plot Power Change for each one

iinames = ["(A) Power at 10 years$-1$",
           "(B) Power at 20 years$-1$",
           "Difference (B) - (A)"
           ]

iiplot = [power10_aavg,
          power40_aavg,
          power_diff]

bboxplot   = [-50,-10,50,65]
ex = 1
fig,axs,_  = viz.init_orthomap(1,3,bboxplot,figsize=(26,12.5),centlon=-30)




for ii in range(3):
    ax        = axs[ii]
    ax        = viz.add_coast_grid(ax,bbox=bboxplot,fill_color='lightgray',fontsize=fsz_tick)
    
    if ii == 2:
        cmap = 'cmo.balance'
        vlms = [-1e4,1e4]
    else:
        
        if ii == 0:
            vlms = [0,10000]
            cmap = 'cmo.thermal'
        else:
            vlms = [0,10000]
            cmap = 'cmo.thermal'
    
    ax.set_title(iinames[ii],fontsize=fsz_title)
    plotvar   = iiplot[ii][ex]
    
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,cmap=cmap,transform=proj,
                        vmin=vlms[0],vmax=vlms[1])

    cb = viz.hcbar(pcm,ax=ax)
    cb.ax.tick_params(labelsize=fsz_tick)
    
    if ii == 0:
        viz.add_ylabel(expnames_plot[ex],ax=ax,fontsize=fsz_title)

figname = "%sPower_Comparison_Qnet__40minus10_%s.png" % (figpath,expnames_plot[ex])
plt.savefig(figname,dpi=150)
#for ex in range(4):
    
    

    

    
#%%
# def calc_spec(x,nsmooth):
#     specout = 
    


# specs_pointwise = []
# for ii in range(4):
    
#     nsmooth = 2
    
    
    
    
#     ds = qnets_reg[ii]
#     vratios40 = xr.apply_ufunc(
#         calc_spec,
#         ds,
#         input_core_dims=[['time']],
#         output_core_dims=[[]],
#         vectorize=True,
#         )
#     vratios40_all.append(vratios40)




#vratios40 = [ds.reduce(calc_slope_lf,dim='time') for ds in qnets_reg]


    
    
    #movmean10 = lambda x: movmean(x,12*10)
    #movmean40 = lambda x: movmean(x,12*40)






# Apply Low Pass Filter (40)
# Apply Low Pass Filter (10)
# Calculate Slope...


#%% Recreate plots from O'Reilly 2016, qnet and sst

# ex      = 0
# winlens = np.arange(3,16,1)
# nwin    = len(winlens)


# for ww in range(nwin):
    
#     ds_in     = aavg_byvar[ex]
#     dsann     = [calc_ann_avg(ds) for ds in dsin]
#     dsann_lp  = [movmean(ds,win) for ds in dsann]
#     dsann_hp  = [dsann[ex] - dsann_lp[ex] for ex in range(4)]
    



#%% Briefly Check the spatial pattern

vv    = 1
itime = 5

fig,axs= plt.subplots(4,1,subplot_kw={'projection':ccrs.PlateCarree()})

for ex in range(4):
    
    
    if ex < 3:
        plotvar = dsreg_byvar[ex][vnames[vv]].isel(time=itime)#mean('time')
    else:
        plotvar = proc.sel_region_xr(era_reg[vv],bbsel)#.mean('time')
        plotvar = proc.xrdeseason(era_reg[vv]) # Note: NOT DETRENDED
        
        plotvar = plotvar.isel(time=itime)#mean('time')
    
    ax = axs[ex]
    ax.coastlines()
    ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,vmin=-50,vmax=50,cmap='cmo.balance')


#%% Try regression 






#%% Compute Lag Relationships





#ds_ssts = [ds['TS'] for ds in ]
#sst_ann = [p(ds) for ds in ]


#%%



for ex in range(3):
    
    


# ============================
#%% Part (3): Spatial Patterns
# ============================


# Compute Standard Deviation
stdevs_cesm = [ds.std('time') for ds in vars_byexp]


#%% Compare Ratio of SST vs Fprime
#>> Focus on comparison of MCOM --> FOM

#sst_ratio    = np.log(stdevs_cesm[2].TS / stdevs_cesm[1].TS)
#fprime_ratio = np.log(stdevs_cesm[2].Fprime / stdevs_cesm[1].Fprime)


iidenom   = 1
iinumer   = 2
bboxplot  = [-80,0,0,65]
fsz_tick  = 24
fsz_title = 35
fsz_axis  = 36
proj      = ccrs.PlateCarree()

lon       = stdevs_cesm[2].lon.data
lat       = stdevs_cesm[2].lat.data

cints     = np.arange(-1,1.05,.05)

#vv        = 0
for vv in range(3):
    
    vname     = vnames[vv]
    logratio  = np.log(stdevs_cesm[iinumer][vname].data / stdevs_cesm[iidenom][vname].data)
    fig,ax,_  = viz.init_orthomap(1,1,bboxplot,figsize=(28,12.5))
    ax        = viz.add_coast_grid(ax,bbox=bboxplot,fill_color='lightgray',fontsize=fsz_tick)
    
    plotvar   = logratio
    #pcm       = ax.contourf(lon,lat,plotvar,cmap='cmo.balance',transform=proj,zorder=-1,levels=cints)
    pcm       = ax.pcolormesh(lon,lat,plotvar,cmap='cmo.balance',transform=proj,zorder=-1,vmin=-.5,vmax=.5)
    
    cb        = viz.hcbar(pcm,ax=ax,fontsize=fsz_tick,fraction=0.04)
    cb.set_label(r"%s Log Ratio $(\frac{%s}{%s})$" % (vnames[vv],expnames_plot[iinumer],expnames_plot[iidenom]),fontsize=fsz_axis)
    
    figname = "%sLog_ratio_%s_%s_to_%s_%s.png" % (figpath,bbfn,expnames[iinumer],expnames[iidenom],vnames[vv])
    plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Do a quick EOF Analysis (on Fprime)

dsmask_natl = proc.sel_region_xr(dsmask180,bbox_natl)

vv         = 2
vname      = vnames[vv]
invars_eof = [ds[vname].data * dsmask_natl.data[None,:,:] for ds in vars_byexp]

ex         = 0

eofpats    = []
pcs        = []
varexps    = []
for ex in range(3):

    eof_out    = scm.calc_enso(invars_eof[ex],lon,lat,3,bbox=bbox_natl)
    eofpats.append(eof_out[0])
    pcs.append(eof_out[1])
    varexps.append(eof_out[2])
    
    
#%% Plot EOF

nmode = 0
im    = 1

if vv == 0:
    vmax = 1
elif vv == 2:
    vmax = 25
else:
    vmax = 25#50
    
    
for nmode in range(3):
    for im in range(12):
    
        fig,axs,_= viz.init_orthomap(1,3,bboxplot,figsize=(28,12.5))
        
        for ex in range(3):
            ax        = axs[ex]
            ax        = viz.add_coast_grid(ax,bbox=bboxplot,fill_color='lightgray',fontsize=fsz_tick)
            
            title     = "%s\nVar. Expl.: %.2f" % (expnames_plot[ex],varexps[ex][im,nmode]*100) + "%"
            ax.set_title(title,fontsize=fsz_title)
            
            plotvar   = eofpats[ex][:,:,im,nmode]
            # # if ex == 2:
            # #     plotvar = plotvar*-1
            # if ex == 0:
            #     plotvar = plotvar*-1   
            
            pcm       = ax.pcolormesh(lon,lat,plotvar,cmap='cmo.balance',vmin=-vmax,vmax=vmax,transform=proj)
            
            
        cb = viz.hcbar(pcm,ax=axs.flatten(),fontsize=fsz_tick)
        cb.set_label("EOF%i, %s %s [$%s$]" % (nmode+1,mons3[im],vnames[vv],vunits[vv]),fontsize=fsz_axis)
        
        figname = "%sEOF_Pattern_%s_mode%02i_mon%02i.png" % (figpath,vnames[vv],nmode+1,im+1)
        plt.savefig(figname,dpi=150,bbox_inches='tight')


#%% Compute the power spectra of Certain Months and plot

def reshape_pc(pc,transpose=False):
    nyr,_,nmode = pc.shape
    if transpose:
        pc = pc.transpose(1,0,2)
    return pc.reshape(nyr*12,nmode)
#im       = 0
nmode    = 0
#in_pcs   = [pc[:,im,nmode] for pc in pcs]
in_pcs = [reshape_pc(pc,transpose=False)[...,nmode] for pc in pcs]
nsmooths = [25,]*3#[24,24,200]

specout_byexp = scm.quick_spectrum(in_pcs,nsmooths,0.10,return_dict=True)


# Plot Spectra 
dtmon_fix       = 60*60*24*30
xper            = np.array([40,10,5,1,0.5])
xper_ticks      = 1 / (xper*12)


fig,ax = plt.subplots(1,1,figsize=(10,4.5),constrained_layout=True)


for ex in range(3):
    els           = ['dashed','solid']
    
    dtmon_fix       = 60*60*24*30
    xper            = np.array([40,10,5,1,0.5])
    xper_ticks      = 1 / (xper*12)

    
        
    plotspec = specout_byexp['specs'][ex] / dtmon_fix
    plotfreq = specout_byexp['freqs'][ex] * dtmon_fix
    
    if len(plotspec) < len(plotfreq):
        plotfreq = plotfreq[1:]
    elif len(plotspec) > len(plotfreq):
        plotspec = plotspec[1:]
        
        
    
    color_in = expcols[ex]
    
    ax.loglog(plotfreq,plotspec,lw=2,label=expnames[ex],c=color_in,
              ls='solid',markersize=2.5)
    
        
    ax.set_xlim([1/1000,0.5])
    ax.axvline([1/(6)],label="",ls='dotted',c='gray')
    ax.axvline([1/(12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(5*12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(10*12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(40*12)],label="",ls='dotted',c='gray')
    
    
    ax.set_ylabel("Power [$%s ^2 cycle \, per \, mon$]" % vunits[vv])
    
    ax2 = ax.twiny()
    ax2.set_xlim([1/1000,0.5])
    ax2.set_xscale('log')
    ax2.set_xticks(xper_ticks,labels=xper)
    
    ax.legend()
    ax.set_title("%s Power Spectra, nsmooth=%i" % (vnames[vv],nsmooths[ex]))
    
    



#%% ===================================

#%% Check Qnet spectra with and without ENSO


bbsel_enso          = [-40,-15,52,62]
nsmooths            = [24,24,200]

vv                  = 1


specout_byexp = []
for ex in range(3):
    nsmooth = nsmooths[ex]
    
    flx0            = ds_flx_noenso[ex]
    flx1            = vars_byexp[ex].SHF
    invars          = [flx0,flx1]
    invars_reg      = [proc.sel_region_xr(ff,bbsel_enso) for ff in invars]
    invars_reg      = [proc.area_avg_cosweight(ff) for ff in invars_reg]
    invars_arr      = [iv.data for iv in invars_reg]
    
    specout         = scm.quick_spectrum(invars_arr,nsmooth,pct=0.10,return_dict=True)

    specout_byexp.append(specout)

#%%


fig,axs = plt.subplots(3,1,figsize=(8,8),constrained_layout=True)


for ex in range(3):
    ax = axs[ex]
    expnames_enso = ["No ENSO","With ENSO"]
    els           = ['dashed','solid']
    
    dtmon_fix       = 60*60*24*30
    xper            = np.array([40,10,5,1,0.5])
    xper_ticks      = 1 / (xper*12)
    specout         = specout_byexp[ex]
    
    for ii in range(2):
        
        plotspec = specout['specs'][ii] / dtmon_fix
        plotfreq = specout['freqs'][ii] * dtmon_fix
        
        
        color_in = expcols[ex]
        
        ax.loglog(plotfreq,plotspec,lw=1,label=expnames_enso[ii],c=color_in,
                  ls=els[ii],markersize=2.5)
        
        
    ax.set_xlim([1/1000,0.5])
    ax.axvline([1/(6)],label="",ls='dotted',c='gray')
    ax.axvline([1/(12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(5*12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(10*12)],label="",ls='dotted',c='gray')
    ax.axvline([1/(40*12)],label="",ls='dotted',c='gray')
    
    ax.set_ylabel("Power [$%s ^2 cycle \, per \, mon$]" % vunits[vv])
    
    ax2 = ax.twiny()
    ax2.set_xlim([1/1000,0.5])
    ax2.set_xscale('log')
    ax2.set_xticks(xper_ticks,labels=xper)
    
    ax.legend()
    ax.set_title("Power Spectra, %s, nsmooth=%i" % (expnames_long[ex],nsmooths[ex]))
    
    
    
    
#%%







#%% Below is srap from older script


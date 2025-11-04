#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""

Test sensitivity to area averaging at different steps in the calculation

Step (1): Preprocess Inputs        (1) Estimate from Area-averaged variables
Step (2): Estimate Parameters      (2) Run with Area-averaged parameters
Step (3): Run stochastic model     (3) Area-averaged SST output
Step (4): Compute metrics          (4) Area-averaged metrics (ACF, spectra)

Created on Mon Apr 21 14:27:20 2025

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

# local device

amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import cvd_utils as cvd

#%% Plotting Inputs

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

bboxplot    = [-80, 0, 20, 65]
mpl.rcParams['font.family'] = 'Avenir'
mons3       = proc.get_monstr(nletters=3)

fsz_tick    = 18
fsz_axis    = 20
fsz_title   = 16

rhocrit     = proc.ttest_rho(0.05, 2, 86)
proj        = ccrs.PlateCarree()
mons3       = proc.get_monstr()

#%% Set Paths, User Edits

# Set Bounding Path output path
outpath         = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/region_average/"
bbsel           = [-40,-15,52,62]
bbfn,bbtitle = proc.make_locstring_bbox(bbsel)
bbname          = "SPGne"
outpathbb       = outpath + "%s_%s/" % (bbname,bbfn)
proc.makedir(outpathbb)

# Indicate Other Paths
procpath        = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
smoutput_path   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/"
figpath         = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20250523/"
proc.makedir(figpath)

# Indicate Experiment and Comparison Name 
comparename     = "paperoutline"
expnames        = ["SST_Obs_Pilot_00_Tdcorr0_qnet","SST_Obs_Pilot_00_Tdcorr1_qnet","SST_ERA5_1979_2024"]
expnames_long   = ["Stochastic Model (with re-emergence)","Stochastic Model","ERA5"]
expcols         = ["turquoise","goldenrod","k"]
expls           = ["dotted","dashed",'solid']

# (12) Draft 2 Edition (using case with GMSST Mon Detrend)
comparename     = "Draft02"
expnames        = ["SST_ORAS5_avg_GMSST_EOFmon_usevar_NoRem_NATL","SST_ORAS5_avg_GMSST_EOFmon_usevar_NATL","SST_ERA5_1979_2024"]
expnames_long   = ["Stochastic Model (no re-emergence)","Stochastic Model (with re-emergence)","ERA5"]
expnames_short  = ["SM","SM_REM","ERA5"]
expcols         = ["goldenrod","turquoise","k"]
expls           = ["dashed","dotted",'solid']
detect_blowup   = True



nexps           = len(expnames)

#%% Method (1), Area averaged inputs


"""

Method (1), Area averaged inputs

"""

simname = "SPGNE"
simpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/sm_point_run/"
tmax    = int(12 * 1e5) # Restrict maximum Time
# Load Parameters and SST
m1_input = xr.open_dataset("%s%s_Method1_inputs.nc" % (simpath,simname)).load()
m1_sst   = xr.open_dataset("%s%s_Method1_output.nc" % (simpath,simname)).load()


# Compute metrics
irun        = 0 # Just select 1 run
m1in        = [m1_sst.sst.data[irun,:tmax]]
lags        = np.arange(61)
nsmooths    = [250,250,4]
tsm_m1      = scm.compute_sm_metrics(m1in,nsmooth=nsmooths,lags=lags)




#%% Method (2), Area-averaged parameter estimates


# Load Parameters and SST
m2_input = xr.open_dataset("%s%s_Method2_inputs.nc" % (simpath,simname)).load()
m2_sst   = xr.open_dataset("%s%s_Method2_output.nc" % (simpath,simname)).load()

# Compute Metrics
irun        = 0 # Just select 1 run
m2in        = [m2_sst.sst.data[irun,:tmax]]
tsm_m2      = scm.compute_sm_metrics(m2in,nsmooth=nsmooths,lags=lags)



#%% Method (3), Area averaged SST

"""
Method (3), Area averaged SST

- Read in SST 
- Preprocess pointwise
- Then take annual average

"""

recompute = False

if recompute:
    # Load in SSTs
    ds_ssts  = []
    for ex in tqdm.tqdm(range(nexps)):
        expname = expnames[ex]
        dsview  = dl.load_smoutput(expname,smoutput_path,load=False)
        dsreg   = proc.sel_region_xr(dsview,bbsel).load()
        ds_ssts.append(dsreg)
    
    # Preprocess SSTs (anomalize, take area average)
    ssts     = [ds.SST for ds in ds_ssts]
    def preproc_ds(ds):
        # Remove seasonal cycle
        dsanom = proc.xrdeseason(ds)
        # Remove (linear) trend
        dsanom_dt = proc.xrdetrend(dsanom)
        return dsanom_dt
    sstas    = [preproc_ds(ds) for ds in ssts]
    ssts_reg = [proc.area_avg_cosweight(ds,) for ds in sstas]
    
    
    
    # For stochastic model output, flatten the simulation
    for ex in range(nexps):
        
        insst = ssts_reg[ex]
        if 'run' in insst.dims: #len(insst.run) > 1:
            print("Flattening %s" % expnames[ex])
            # reshape the file
            ssttemp  = insst.data # run x nyr
            nrun,ntime = ssttemp.shape
            times_sim   = xr.cftime_range(start='0000',periods=int(nrun*ntime),freq="MS",calendar="noleap")
            
            coords_new = dict(time=times_sim)
            da_new     = xr.DataArray(ssttemp.flatten(),coords=coords_new,dims=coords_new,name="SST")
            
            ssts_reg[ex] = da_new
            
    # #%% Save SSTas (not recommended, is ~3GB for )
    # for ex in range(nexps): 
    #     expname = expnames[ex]
    #     outname = "%sSSTA_%s.nc" % (outpathbb,expname)
    #     dsout   = sstas[ex]
    #     edict   = proc.make_encoding_dict(dsout)
    #     dsout.to_netcdf(outname,encoding=edict)
    
    #% Save Detrended SSTs
    for ex in range(nexps): 
        expname = expnames[ex]
        outname = "%sSSTA_Area_Avg_%s.nc" % (outpathbb,expname)
        dsout   = ssts_reg[ex]
        edict   = proc.make_encoding_dict(dsout)
        dsout.to_netcdf(outname,encoding=edict)
else:
    
    #% Load
    ssts_reg = []
    for ex in range(nexps): 
        expname = expnames[ex]
        outname = "%sSSTA_Area_Avg_%s.nc" % (outpathbb,expname)
        ds = xr.open_dataset(outname).load().SST
        ssts_reg.append(ds)
        
        #dsout   = ssts_reg[ex]
        #edict   = proc.make_encoding_dict(dsout)
        #dsout.to_netcdf(outname,encoding=edict)
    
#%% Plot detrended timeseries
ssts_arr    = [ds.data for ds in ssts_reg]

era_sst   = ssts_arr[-1]
ntime_era = len(era_sst)

# Find time range in stochastic model with smallest rmse
indices_plot = []
for nn in range(2):
    
    sst_in = ssts_arr[nn]
    
    


        
    

    
#%% Plot timeseries

istart     = 12 #None # If None, just plot last n timesteps
fsz_legend = 14
fsz_axis   = 18
fsz_ticks  = 14
xplot      = np.arange(ntime_era)
ylims      = [-1.75,1.75]
xlims      = [xplot[0],xplot[-1]]

# Get era5 times
times      = ssts_reg[-1].time.data
years      = [str(t)[:4] for t in times]
plotint    = 36

times_sm   = ssts_reg[0].time.data[-ntime_era:]
years_sm   = [str(t)[:4] for t in times_sm]

# Initialize Fig
fig,axs    = plt.subplots(2,1,constrained_layout=True,figsize=(12,8))

# Plot Era
ax         = axs[0]
ax.plot(era_sst,label=expnames[-1],c=expcols[-1])


# Plot SM
ax = axs[1]
if istart is None:
    ax.plot(ssts_arr[0][-ntime_era:],label=expnames_long[0],c=expcols[0])
    ax.plot(ssts_arr[1][-ntime_era:],label=expnames_long[1],c=expcols[1])
else:
    ax.plot(ssts_arr[0][istart:istart+ntime_era],label=expnames_long[0],c=expcols[0])
    ax.plot(ssts_arr[1][istart:istart+ntime_era],label=expnames_long[1],c=expcols[1])
ax.legend(fontsize=fsz_legend)

# Label Axes
for ax in axs:
    ax.legend(fontsize=fsz_legend)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_ylabel("SST Anomaly [$\degree$C]",fontsize=fsz_axis)
    ax.axhline([0],ls='dotted',c='k',lw=0.75)
    ax.tick_params(labelsize=fsz_ticks)
ax = axs[0]
ax.set_xticks(xplot[::plotint],labels=years[::plotint])

ax = axs[1]
ax.set_xticks(xplot[::plotint],labels=years_sm[::plotint])


figname = "%s%sSST_SM_ERA5_Timeseries_Comparison_istart%05i.png" % (figpath,comparename,istart,)
if darkmode:
    figname = proc.addstrtoext(figname,"_darkmode")
plt.savefig(figname,dpi=150,bbox_inches='tight',transparent=transparent)


#%% Plot just the ERA5 Timeseries

# Initialize Fig
fig,ax    = plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))

# Plot Era
ax.plot(era_sst,label=expnames[-1],c=dfcol)

#ax.legend(fontsize=fsz_legend)
ax.set_xlim(xlims)
ax.set_ylim(ylims)
ax.set_ylabel("SST Anomaly [$\degree$C]",fontsize=fsz_axis)
ax.axhline([0],ls='dotted',c=dfcol,lw=0.75)
ax.tick_params(labelsize=fsz_ticks,)
ax.set_xticks(xplot[::plotint],labels=years[::plotint])

# Hide the right and top spines
ax.spines[['right', 'top']].set_visible(False)

figname = "%s%sERA5_Timeseries_Comparison_istart%05i.png" % (figpath,comparename,istart,)
if darkmode:
    figname = proc.addstrtoext(figname,"_darkmode")
plt.savefig(figname,dpi=150,bbox_inches='tight',transparent=transparent)

#%% Find time with the best fit

lpcutoff  = 240
ntime_sm  = len(ssts_arr[0])

nsegments = ntime_sm - ntime_era
rmse_all  = np.zeros((2,nsegments)) * np.nan
for ii in tqdm.tqdm(range(nsegments)):
    idxin = np.arange(ii,ii+ntime_era)
    
    for ex in range(2):
        
        sstin            = ssts_arr[ex][idxin]
        sstref           = ssts_arr[-1]
        if lpcutoff is not None:
            sstin = proc.lp_butter(sstin,lpcutoff,6)
            sstref = proc.lp_butter(sstref,lpcutoff,6)
        
        rmse_all[ex,ii]  = np.sqrt(np.nanmean(sstin - sstref)**2)
        
#%%% Plot time with the smallest value
fig,ax = plt.subplots(1,1,figsize=(12.5,4.5))

imins  = []
istart = np.arange(nsegments) 
for ex in range(2):
    #rmsein = rmse_all[ex,:]
    rmsein = proc.lp_butter(rmse_all[ex,:],12,6)
    ax.plot(istart,rmsein,label=expnames_long[ex],c=expcols[ex])
    
    imin = np.nanargmin(rmsein)
    ax.plot(imin,rmsein[imin],marker="d",c="k",label="istart=%i" % (imin))
    
    imins.append(imin)
    
    
ax.legend()

#%% Plot resulting timeseries

remove_topright   = True
label_actual_year = True
ylims             = [-2,2]

# Initialize Fig
fig,axs           = plt.subplots(2,1,constrained_layout=True,figsize=(12,8))

imins_sel = 0

# Plot Era
ax         = axs[0]
ax.plot(era_sst,label=expnames_long[-1],c=dfcol)

era_lp  = proc.lp_butter(era_sst,120,6)

#from : https://stackoverflow.com/questions/64068659/bar-chart-in-matplotlib-using-a-colormap
# cmapin  = plt.get_cmap("RdBu_r")
# rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
# ax.bar(xplot,era_lp,color=cmapin(rescale(era_lp)),edgecolor="w")


if label_actual_year:
    times_sm   = ssts_reg[0].time.data[imins[imins_sel]:imins[imins_sel]+ntime_era]
else:
    times_sm   = ssts_reg[0].time.data[:ntime_era]
years_sm   = [str(t)[:4] for t in times_sm]


# Plot SM
ax = axs[1]
ax.plot(ssts_arr[0][imins[imins_sel]:imins[imins_sel]+ntime_era],label=expnames_long[0],c=expcols[0])
ax.plot(ssts_arr[1][imins[imins_sel]:imins[imins_sel]+ntime_era],label=expnames_long[1],c=expcols[1])
ax.legend(fontsize=fsz_legend)

# Label Axes
for ax in axs:
    ax.legend(fontsize=fsz_legend)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_ylabel("SST Anomaly [$\degree$C]",fontsize=fsz_axis)
    ax.axhline([0],ls='dotted',c=dfcol,lw=0.75)
    ax.tick_params(labelsize=fsz_ticks)
    if remove_topright:
        ax.spines[['right', 'top']].set_visible(False)
        
ax = axs[0]
ax.set_xticks(xplot[::plotint],labels=years[::plotint])

ax = axs[1]
ax.set_xticks(xplot[::plotint],labels=years_sm[::plotint])

figname = "%sSST_SM_ERA5_Timeseries_Comparison_%s_istart%05i_lpf%02i.png" % (figpath,comparename,imins[1],lpcutoff)
if darkmode:
    figname = proc.addstrtoext(figname,"_darkmode")
    
plt.savefig(figname,dpi=150,bbox_inches='tight',transparent=transparent)


#%% Compute Metrics for each case

ssts_arr    = [ds.data for ds in ssts_reg]
lags        = np.arange(61)
nsmooths    = [250,250,4]
metrics_out = scm.compute_sm_metrics(ssts_arr,nsmooth=nsmooths,lags=lags)

tsm_m3 = metrics_out

#%% Option to save metrics (works fine)
outname     = "%sMetrics_%s_Method03.npz" % (outpathbb,comparename)
np.savez(outname,*metrics_out,allow_pickle=True)

# def repack_metrics(tsm):
    
#     mons        = np.arange(1,13,1)
    
#     # Get ACFs
#     acfs           = np.array(metrics_out['acfs']) # [mon,exp,lags]
#     nmon,nexp,nlag = acfs.shape
#     coords_acf     = dict(mon=mons,exp=np.arange(nexp),lags=np.arange(nlag))
#     da_acf         = xr.DataArray(acfs,coords=coords_acf,dims=coords_acf,name='acfs')
    
#     # Get Monthly Variances
#     monvars       = np.array(metrics_out['monvars']) # [exp,mon]
#     coords        = dict(exp=np.arange(nexp),mon=mons)
#     da_monvar     = xr.DataArray(monvars,coords=coords,dims=coords,name='monvars') 
    
    
#     # Get Spectra
#     specs        = np.array(metrics_out['specs'])

#%% Plot ACF for comparison

kmonth = 1
xtks   = lags[::6]

for kmonth in range(12):
    fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,4))
    ax,_   = viz.init_acplot(kmonth,xtks,lags,ax=ax)
    
    for ex in range(nexps):
        
        plotvar = metrics_out['acfs'][kmonth][ex]
        ax.plot(lags,plotvar,
                label=expnames_long[ex],color=expcols[ex],ls=expls[ex],lw=2.5)
    ax.legend()
    
    ax.set_title("%s SST Autocorrelation" % mons3[kmonth])
    ax.set_xlabel("Lag (Months)")
    ax.set_ylabel("Correlation with %s. Anomalies" % (mons3[kmonth]))
    
    
    figname = "%sACF_%s_Method3_mon%02i.png" % (figpath,comparename,kmonth+1)
    plt.savefig(figname,dpi=150)

#%% Plot ACF Map across all months


cints = np.arange(-1,1.05,0.05)
fig,axs = plt.subplots(3,1,constrained_layout=True,figsize=(8,6))
nlags = len(lags)

for ex in range(nexps):
    ax  = axs[ex]
    
    acfs_map = np.zeros((nlags,12)) * np.nan
    for kmonth in range(12):
        acfs_map[:,kmonth] = metrics_out['acfs'][kmonth][ex]
        
    cf = ax.contourf(lags,mons3,acfs_map.T,levels=cints,cmap='cmo.balance')
    
    ax.set_xticks(xtks)
    ax.set_title(expnames_long[ex],fontsize=fsz_axis)
    
cb = fig.colorbar(cf,ax=axs.flatten(),pad=0.05,fraction=0.025)
cb.set_label("Correlation")
    
    #ax.grid(True)
    # ax.plot(lags,plotvar,
    #         label=expnames_long[ex],color=expcols[ex],ls=expls[ex],lw=2.5)



#%% Plot Monthly Variance

fig,ax = viz.init_monplot(1,1,constrained_layout=True,figsize=(4.5,4))

for ex in range(nexps):
    
    plotvar = metrics_out['monvars'][ex]
    ax.plot(mons3,plotvar,
            label=expnames_long[ex],color=expcols[ex],ls=expls[ex],lw=2.5)
ax.legend()
ax.set_ylabel("Interannual Variance [$\degree$C$^2$]")

#%% plot Log spectra


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

# # Plot Confidence Interval (Stochastic Model)
# cloc_sm        = [plotfreq[0],1e-2]
# dof_sm         = metrics_out['dofs'][0]
# cbnds_sm       = proc.calc_confspec(alpha,dof_sm)
# proc.plot_conflog(cloc_sm,cbnds_sm,ax=ax,color=expcols[0])

#%% Add Barplot (similar in pom/exploratory/)

cutoff      = 120 # In Months
order       = 6
lpfilter_ts = lambda x: proc.lp_butter(x,cutoff,order)

instd       = [ds.std('time') for ds in ssts_reg]
ssts_reg_lp = [lpfilter_ts(ds) for ds in ssts_reg]
instd_lp    = [np.std(ts) for ts in ssts_reg_lp]



expnames_short = ["SM (REM)","SM","ERA5"]

vratio      = np.array(instd_lp) / np.array(instd) * 100

xlabs       = ["%s\n%.2f" % (expnames_short[ii],vratio[ii])+"%" for ii in range(len(vratio))]
fig,ax      = plt.subplots(1,1,constrained_layout=True,figsize=(6,6))
braw        = ax.bar(np.arange(nexps),instd,color='gray')
blp         = ax.bar(np.arange(nexps),instd_lp,color='k')

ax.bar_label(braw,fmt="%.04f",c='gray')
ax.bar_label(blp,fmt="%.04f",c='k')

#ax.bar_label(vratio,fmt="%.2f",c=w,label_type='bottom')

ax.set_xticks(np.arange(nexps),labels=xlabs)
ax.set_ylabel("$\sigma$(SST) [$\degree$C]")

ax.set_ylim([0,1.0])

#%% Method (4), Area averaged inputs

"""

Method (4), Area averaged Metrics

-- Note stuck for now because need to compuxte acf for TdCorr1 run

"""

# Load the (pointwise) ACFs


# Load for stochastic model
ds_acfs = []
for ex in tqdm.tqdm(range(nexps)):
    expname = expnames[ex]
    ncname  = "%s/%s/Metrics/ACF_Pointwise.nc" % (smoutput_path,expname,)
    ds = xr.open_dataset(ncname)#.load()
    dsreg = proc.sel_region_xr(ds,bbsel)
    ds_acfs.append(dsreg.acf.load())



#%% Plot ACFs for each region

kmonth      = 1
ds_acfsmon  = [ds.isel(mons=kmonth,thres=0) for ds in ds_acfs]
_,nlon,nlat,_ = ds_acfsmon[-1].shape

acf_aavg    = [proc.area_avg_cosweight(ds) for ds in ds_acfsmon]

#%% First, just check sensitivity to different ensemble members


fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,4))
ax,_   = viz.init_acplot(kmonth,xtks,lags,ax=ax)

for rr in range(10):
    ax.plot(lags,acf_aavg[0].isel(ens=rr),alpha=0.75,label="Run %0i" % (rr+1))
ax.plot(lags,acf_aavg[0].mean('ens'),alpha=1,label="Mean",c="k")

ax.legend(ncol=4)

#%% Next, let's compare the run averaged values between the stochastic model, etc

acf_aavg_ensmean = acf_aavg.copy()
for ex in range(nexps):
    acfin = acf_aavg_ensmean[ex]
    if 'ens' in acfin.dims:
        print("Taking the ensemble mean for %s" % (expnames_long[ex]))
        acf_aavg_ensmean[ex] = acf_aavg_ensmean[ex].mean('ens')

#%% Now plot the same figure as above but using these new valuews

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,4))
ax,_   = viz.init_acplot(kmonth,xtks,lags,ax=ax)

for ex in range(nexps):
    
    plotvar = acf_aavg_ensmean[ex]
    ax.plot(lags,plotvar,
            label=expnames_long[ex],color=expcols[ex],ls=expls[ex],lw=2.5)
ax.legend()

ax.set_title("%s SST Autocorrelation" % mons3[kmonth])
ax.set_xlabel("Lag (Months)")
ax.set_ylabel("Correlation with %s. Anomalies" % (mons3[kmonth]))


#%% Now check if this is due to pointwise spread in the stochastic model


fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,4))
ax,_   = viz.init_acplot(kmonth,xtks,lags,ax=ax)

for ex in range(nexps):
    
    for o in tqdm.tqdm(range(nlon)):
        for a in range(nlat):
            
            plotvar = ds_acfsmon[ex].isel(lon=o,lat=a).mean('ens').squeeze()
            ax.plot(lags,plotvar,alpha=0.05,
                    label="",color=expcols[ex],ls=expls[ex],lw=1.5)
            
    plotvar = acf_aavg_ensmean[ex]
    ax.plot(lags,plotvar,
            label=expnames_long[ex],color=expcols[ex],ls=expls[ex],lw=2.5)
    
ax.legend()
ax.set_title("%s SST Autocorrelation" % mons3[kmonth])
ax.set_xlabel("Lag (Months)")
ax.set_ylabel("Correlation with %s. Anomalies" % (mons3[kmonth]))

#%% Map the Autocorrelation Structure Across all months

# ===================================
#%% Plot Comparison between methods
# ===================================

kmonth = 1

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,4))
ax,_   = viz.init_acplot(kmonth,xtks,lags,ax=ax)

# Stochastic Model ---

# Plot Method 1
plotvar = tsm_m1['acfs'][kmonth][0]
ax.plot(lags,plotvar,label="Method 1 (Area-Average before Parameter Estimate)",c=expcols[0],lw=2.5,ls='dotted')

# Plot Method 2
plotvar = tsm_m2['acfs'][kmonth][0]
ax.plot(lags,plotvar,label="Method 2 (Area-Average Pointwise Parameters)",c=expcols[0],lw=1.5,ls='dashed')

# Plot Method 3
plotvar = tsm_m3['acfs'][kmonth][0]
ax.plot(lags,plotvar,label="Method 3 (Area-Average SST)",c=expcols[0],lw=1.5,ls='solid')

# Plot Method 4
plotvar = acf_aavg_ensmean[0]
ax.plot(lags,plotvar,label="Method 4 (Area-Average ACFs)",c=expcols[0],lw=1,ls='dashdot')


# ERA 5 ---

# Plot Method 3
plotvar = tsm_m3['acfs'][kmonth][-1]
ax.plot(lags,plotvar,label="ERA5 (Area-Average SST)",c="k",lw=1.5,ls='solid',zorder=-1)

# Plot Method 4
plotvar = acf_aavg_ensmean[-1]
ax.plot(lags,plotvar,label="ERA5 (Area-Average ACFs)",c="k",lw=1,ls='dashdot',zorder=-1)

ax.legend()
ax.set_title("")
ax.set_xlabel("Lag (Months)")
ax.set_ylabel("Correlation with %s. Anomalies" % (mons3[kmonth]))


figname = "%sMethod_Comparison_ACF_%s_mon%02i.png" % (figpath,comparename,kmonth+1)
plt.savefig(figname,dpi=150)

# =====================================
# %% Plot differences in the parameters
# =====================================

# Load in parameters for the region
ncparams = simpath + "SPGNE_Pointwise_Parameters.nc"
paramsreg  = xr.open_dataset(ncparams).load()

_,nlat,nlon = paramsreg.lbd_a.shape


lbd_a_withnan = xr.where(paramsreg.lbd_a == 0.,np.nan,paramsreg.lbd_a)

#%% Differences in Damping


use_nan=True

fig,ax = viz.init_monplot(1,1)


# Plot each point
for o in tqdm.tqdm(range(nlon)):
    for a in range(nlat):
        if (o == 0) and (a == 0):
            lbl = "Indv. Point"
            alph = 0.05
        else:
            lbl = ""
            alph = 0.005
          
        plotvar = lbd_a_withnan.isel(lon=o,lat=a)
        #plotvar = paramsreg.lbd_a.isel(lon=0,lat=a)
        ax.plot(mons3,plotvar,c='gray',alpha=alph,label=lbl)

# Plot Method (1) Area Averaged Estimate
plotvar = m1_input.lbd_a
ax.plot(mons3,plotvar,ls="dotted",label="Method 1 (Estimate from Area-Averaged Input)",c="midnightblue")

# Method (2)
plotvar = m2_input.lbd_a
ax.plot(mons3,plotvar,ls="dashed",label="Method 2 (Area-Averaged $\lambda^a$)",c='k')



#plotvar = m3_input.lbd_a
ax.set_xlabel("Month")
ax.set_ylabel("$Q_{net}$ Damping [W/ (m$^2$ $\degree$C)]")

ax.legend()

ax.set_ylim([0,40])


#%% Differences in forcing

use_nan=True

fig,ax = viz.init_monplot(1,1)


# Plot each point
for o in tqdm.tqdm(range(nlon)):
    for a in range(nlat):
        if (o == 0) and (a == 0):
            lbl = "Indv. Point"
            alph = 0.5
        else:
            lbl = ""
            alph = 0.005
          
        #plotvar = lbd_a_withnan.isel(lon=0,lat=a)
        plotvar = paramsreg.Fprime.isel(lon=o,lat=a)
        ax.plot(mons3,plotvar,c='gray',alpha=alph,label=lbl)

# Plot Method (1) Area Averaged Estimate
plotvar = m1_input.Fprime
ax.plot(mons3,plotvar,ls="dotted",label="Method 1 (Estimate from Area-Averaged Input)",c="midnightblue")

# Method (2)
plotvar = m2_input.Fprime
ax.plot(mons3,plotvar,ls="dashed",label="Method 2 (Area-Averaged $\lambda^a$)",c='k')


#plotvar = m3_input.lbd_a
ax.set_xlabel("Month")
ax.set_ylabel("F' Forcing [W/m$^2$]")

ax.legend()

ax.set_ylim([0,95])

#%% Mixed-Layer Depth



fig,ax = viz.init_monplot(1,1)

# Plot Method (1) Area Averaged Estimate
plotvar = paramsreg.h.mean('lat').mean('lon')
ax.plot(mons3,plotvar,ls="dotted",label="Method 1 (Area-Average MLD)",c="k")

# Plot each point
for o in tqdm.tqdm(range(nlon)):
    for a in range(nlat):
        if (o == 0) and (a == 0):
            lbl = "Indv. Point"
            alph = 0.5
        else:
            lbl = ""
            alph = 0.005
          
        #plotvar = lbd_a_withnan.isel(lon=0,lat=a)
        plotvar = paramsreg.h.isel(lon=o,lat=a)
        ax.plot(mons3,plotvar,c='gray',alpha=alph,label=lbl)
        
ax.invert_yaxis()
        
#plotvar = m3_input.lbd_a
ax.set_xlabel("Month")
ax.set_ylabel("Mixed-Layer Depth [m]")


#%% Combine the figures (just plot the pointwise and average of each one)

dampcol     = "hotpink"
forcecol    = "teal"
hcol        = 'cornflowerblue'

fig,axs = viz.init_monplot(3,1,figsize=(4,8))

# Plot Damping --- --- 

ax = axs[0]
ax.set_ylim([0,40])

# Plot each point
allvalues = []
for o in tqdm.tqdm(range(nlon)):
    for a in range(nlat):
        if (o == 0) and (a == 0):
            lbl = "Indv. Point"
            alph = 0.05
        else:
            lbl = ""
            alph = 0.005
          
        plotvar = lbd_a_withnan.isel(lon=o,lat=a)
        #plotvar = paramsreg.lbd_a.isel(lon=0,lat=a)
        ax.plot(mons3,plotvar,c=dampcol,alpha=alph,label=lbl)
        
        allvalues.append(plotvar.data)

std = np.nanstd(np.array(allvalues),0)
mu  = np.nanmean(np.array(allvalues),0)
plotvar = proc.area_avg_cosweight(lbd_a_withnan)
ax.plot(mons3,plotvar,c="k",alpha=1,label="Region-Average Damping",lw=2.5)
ax.plot(mons3,mu+std,c="k",alpha=1,label="$\pm \, 1 \, \sigma$",ls='dotted',lw=.75)
#ax.plot(mons3,mu,c='k',alpha=1,label="",ls='dashed',lw=0.75)
ax.plot(mons3,mu-std,c="k",alpha=1,label="",ls='dotted',lw=.75)

ax.set_ylabel(r"Net Heat Flux Damping [$\frac{W}{m^2 \degree C}$]")
ax.legend()

# Plot Forcing --- --- 
ax = axs[1]
ax.set_ylim([0,95])


# Plot each point
allvalues_frc = []
for o in tqdm.tqdm(range(nlon)):
    for a in range(nlat):
        if (o == 0) and (a == 0):
            lbl = "Indv. Point"
            alph = 0.05
        else:
            lbl = ""
            alph = 0.005
          
        #plotvar = lbd_a_withnan.isel(lon=0,lat=a)
        plotvar = paramsreg.Fprime.isel(lon=o,lat=a)
        ax.plot(mons3,plotvar,c=forcecol,alpha=alph,label=lbl)
        
        allvalues_frc.append(plotvar.data)

std = np.nanstd(np.array(allvalues_frc),0)
mu  = np.nanmean(np.array(allvalues_frc),0)

ax.plot(mons3,mu,c="k",alpha=1,label="Region-Average Forcing",lw=2.5)
ax.plot(mons3,mu+std,c="k",alpha=1,label="$\pm \, 1 \, \sigma$",ls='dotted',lw=.75)
ax.plot(mons3,mu-std,c="k",alpha=1,label="",ls='dotted',lw=.75)

ax.set_ylabel("F' Forcing [W/m$^2$]")

# Plot Mixed-Layer Depth --- --- 
ax = axs[2]



# # Plot each point
allvalues_h = []
# Plot each point
for o in tqdm.tqdm(range(nlon)):
    for a in range(nlat):
        if (o == 0) and (a == 0):
            lbl = "Indv. Point"
            alph = 0.05
        else:
            lbl = ""
            alph = 0.005
          
        #plotvar = lbd_a_withnan.isel(lon=0,lat=a)
        plotvar = paramsreg.h.isel(lon=o,lat=a)
        ax.plot(mons3,plotvar,c=hcol,alpha=alph,label=lbl)
        allvalues_h.append(plotvar.data)

std = np.nanstd(np.array(allvalues_h),0)
mu  = np.nanmean(np.array(allvalues_h),0)

ax.plot(mons3,mu,c="k",alpha=1,label="Region-Average Forcing",lw=2.5)
ax.plot(mons3,mu+std,c="k",alpha=1,label="$\pm \, 1 \, \sigma$",ls='dotted',lw=.75)
ax.plot(mons3,mu-std,c="k",alpha=1,label="",ls='dotted',lw=.75)

ax.set_ylim([0,500])
ax.set_xlabel("Month")
ax.set_ylabel("Mixed-Layer Depth [m]")

ax.invert_yaxis()








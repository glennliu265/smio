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

#%% Load Ice Mask for analysis

maskpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/masks/"
masknc   = "cesm2_pic_limask_0.3p_0.05p_0200to2000.nc"
dsmask   = xr.open_dataset(maskpath+masknc).load()
dsmask180 = proc.lon360to180_xr(dsmask)
bbreg     = proc.get_bbox(ds_sst)
dsmask180 = proc.sel_region_xr(dsmask180,bbreg).mask


#%% Also open view of ERA5 for analysis (this might get a bit large...)

ncflx_era   = "ERA5_qnet_NAtl_1979to2024.nc"
pathflx_era = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
flx_era     = xr.open_dataset(pathflx_era+ncflx_era)['qnet']

ncfp_era    = "ERA5_Fprime_QNET_timeseries_QNETpilotObsAConly_nroll0_NAtl.nc"
pathfp_era  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/reanalysis/proc/NATL_proc_obs/"
fprime_era  = xr.open_dataset(pathfp_era+ncfp_era)['Fprime']#.load()

ncsst_era   = "ERA5_sst_NAtl_1979to2024.nc"
pathsst_era = pathflx_era
sst_era     = xr.open_dataset(pathsst_era+ncsst_era)['sst']

# Load GMSST For ERA5 Detrending
detrend_obs_regression = True
dpath_gmsst     = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
nc_gmsst        = "ERA5_GMSST_1979_2024.nc"
ds_gmsst        = xr.open_dataset(dpath_gmsst + nc_gmsst).GMSST_MeanIce.load()


# ============================
#%% Part (1): Point Analysis
# ============================


# ============================
#%% Try Area-Average Analysis?
# ============================

bbsel        = [-40,-15,52,62]
bbfn,bbtitle = proc.make_locstring_bbox(bbsel)

aavg_byvar = []
for ex in range(nexps):
    dsreg       = proc.sel_region_xr(vars_byexp[ex],bbsel)
    dsreg_mask  = dsreg * dsmask180
    aavg        = proc.area_avg_cosweight(dsreg_mask)
    aavg_byvar.append(aavg)

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

fsz_axis = 14
ssts    = [aavg_byvar[ex][sstnames[ex]].data for ex in range(nexps_loop)]
flxs    = [aavg_byvar[ex][flxnames[ex]].data for ex in range(nexps_loop)]
fprimes = [aavg_byvar[ex]['Fprime'].data for ex in range(nexps_loop)]


def lpfilter_ignorenan(ts):
    if np.any(np.isnan(ts)):
        print("Warning, NaNs present. Filling with zero")
        ts[np.isnan(ts)] = 0.
    return proc.lp_butter(ts,120,order=6)

invars = [ssts,flxs,fprimes]
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
    
    if vv == 0:
        ax.set_ylim([0,1.0])
    else:
        ax.set_ylim([0,50])
        
figname = "%sArea_Avg_Stdev_CESM2_Hierarchy_%s.png" % (figpath,bbfn)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Power Spectra

nsmooths = [20,50,150,6]
lags     = np.arange(61)
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
        ax.set_ylim([1e2,1e4])
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
for vv in range(3):
    vname     = vnames[vv]
    dsin      = [ds[vname] for ds in aavg_byvar]
    dsann     = [calc_ann_avg(ds) for ds in dsin]
    dsann_lp  = [movmean(ds,win) for ds in dsann]
    
    ann_byvar.append(dsann)
    ann_byvar_lp.append(dsann_lp)
    
    


leadlags_qnet       = [get_leadlags(ann_byvar[0][ex],ann_byvar[1][ex]) for ex in range(4)]
leadlags_qnet_lp    = [get_leadlags(ann_byvar_lp[0][ex],ann_byvar_lp[1][ex]) for ex in range(4)]

leadlags_fprime     = [get_leadlags(ann_byvar[0][ex],ann_byvar[2][ex]) for ex in range(4)]
leadlags_fprime_lp  = [get_leadlags(ann_byvar_lp[0][ex],ann_byvar_lp[2][ex]) for ex in range(4)]

#%% Make Some Plots

xtks    = np.arange(-10,11,1)

fig,axs = plt.subplots(1,2,figsize=(12.5,4.5),constrained_layout=True)

for aa in range(2):
    
    ax = axs[aa]
    
    if aa == 0:
        #corrin = leadlags_qnet
        corrin = leadlags_fprime
        title  = "Raw"
    else:
        #corrin = leadlags_qnet_lp
        corrin = leadlags_fprime_lp
        title  = "%i-Year Running Mean" % win
        
        
    
    for ex in range(nexps_loop):

        plotcorr = corrin[ex]
        
        plotcorr = plotcorr * -1
        # if ex < 3:
        #     plotcorr = plotcorr * -1
        ax.plot(plotcorr.lag,plotcorr,c=expcols[ex],label=expnames_plot[ex],marker='o')
        
    ax.legend()
    ax.set_title(title)
    
    ax.axhline([0],lw=0.75,c="k")
    ax.axvline([0],lw=0.75,c="k")
        
    ax.set_xlim([-10,10])
    ax.set_xticks(xtks)
        


    
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

lon = stdevs_cesm[2].lon.data
lat = stdevs_cesm[2].lat.data

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

eofpats = []
pcs     = []
varexps = []
for ex in range(3):

    eof_out    = scm.calc_enso(invars_eof[ex],lon,lat,3,bbox=bbox_natl)
    eofpats.append(eof_out[0])
    pcs.append(eof_out[1])
    varexps.append(eof_out[2])
    
    
#%% Plot EOF

nmode = 0
im    = 7

if vv == 0:
    vmax = 2
else:
    vmax = 25#50
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

#%%

#%%




#%% Below is srap from older script


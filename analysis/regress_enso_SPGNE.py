#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Examine relationship of ENSO to SPGNE SST

Created on Wed Dec  3 14:12:29 2025

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


#%%



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

ensopath = "/Users/gliu/Downloads/02_Research/01_Projects/07_ENSO/03_Scripts/ensobase/"
sys.path.append(ensopath)
import utils as ut


#%% User Edits

# SST Files (Anomalized and Detrended)
exppath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/SST_ERA5_1979_2024/Output/"
expnc       = "SST_runid00.nc"

# ENSO Files
ensonc      = "ERA5_ensotest_ENSO_detrendGMSSTmon_pcs3_1979to2024.nc"
ensopath    = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/enso/"

# SST Files (from ENSO calculations)
noensonc    = "ERA5_sst_detrendGMSSTmon_ENSOcmp_lag1_pcs3_monwin3_1979to2024.nc"

# Set Figure Path
figpath     = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20251218/"
proc.makedir(figpath)

# Load GMSST For ERA5 Detrending
detrend_obs_regression  = True
dpath_gmsst             = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
nc_gmsst                = "ERA5_GMSST_1979_2024.nc"

bbox_spgne    = [-40,-15,52,62]

#%% Load relevant data

ds_sst        = xr.open_dataset(exppath + expnc).load()
ds_sst_noenso = xr.open_dataset(ensopath + noensonc).load()
ensoid        = xr.open_dataset(ensopath + ensonc).load()



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
dsa_era         = proc.xrdeseason(ds_era)
flxa_era        = proc.xrdeseason(ds_era_flx)

# Detrend by Regression to the global Mean
ds_gmsst        = xr.open_dataset(dpath_era + nc_gmsst).GMSST_MaxIce.load()
dtout           = proc.detrend_by_regression(dsa_era,ds_gmsst,regress_monthly=True)
sst_era         = dtout.sst

# Detrend Flux as Well
dtout_flx       = proc.detrend_by_regression(flxa_era,ds_gmsst,regress_monthly=True)
flx_era         = dtout_flx.qnet

#proc.printtime(st,print_str="Loaded and procesed data") #15.86s


sst_spgne       = proc.sel_region_xr(sst_era,bbox_spgne)
spgne_aavg      = proc.area_avg_cosweight(sst_spgne)

flx_spgne       = proc.sel_region_xr(flx_era,bbox_spgne)
spgne_aavg_flx  = proc.area_avg_cosweight(flx_spgne)

#sst_spgne_noenso = proc.sel_region_xr(sst_era_noenso,bbox_spgne)
#spgne_aavg_noenso = proc.area_avg_cosweight(sstspg_noenso)

#%% Calculate Lead Lag relationship

# Get the PCs

pc1     = ensoid.pcs.isel(pc=0).data.flatten()
tcoords = dict(time=sst_spgne.time)
pc1     = xr.DataArray(pc1,coords=tcoords,dims=tcoords,name="enso")

pc2     = ensoid.pcs.isel(pc=1).data.flatten()
pc2     = xr.DataArray(pc2,coords=tcoords,dims=tcoords,name="enso")


lags    = np.arange(-60,61,1)


#llcorr_pc1  = ut.calc_lag_regression_1d(sp.signal.detrend(pc1.data),spgne_aavg.data,lags,correlation=True)
llcorr_pc1  = ut.calc_lag_regression_1d(pc1.data,spgne_aavg.data,lags,correlation=True)
llcorr_pc1  = np.array(llcorr_pc1)
llcorr_pc2  = ut.calc_lag_regression_1d(pc2.data,spgne_aavg.data,lags,correlation=True)
llcorr_pc2  = np.array(llcorr_pc2)


llcorr_pc1_flx  = np.array(ut.calc_lag_regression_1d(pc1.data,spgne_aavg_flx.data,lags,correlation=True))
llcorr_pc2_flx  = np.array(ut.calc_lag_regression_1d(pc2.data,spgne_aavg_flx.data,lags,correlation=True))

# Check the relationship between the PCs to make sure they are uncorrelated
llcorr_interpc  = ut.calc_lag_regression_1d(pc1.data,pc2.data,lags,correlation=True)
llcorr_interpc  = np.array(llcorr_interpc)

#%% PLot the timeseries

  
timeplot = pc1.time
fig,axs = plt.subplots(3,1,constrained_layout=True,figsize=(8,4.5))

ax = axs[0]
ax.plot(timeplot,pc1,color='goldenrod')

ax = axs[1]
ax.plot(timeplot,pc2,color="cornflowerblue")

ax = axs[2]
ax.plot(timeplot,spgne_aavg,c='k')

tsnames = ["ENSO PC1","ENSO PC2","SPGNE SST"]

for aa,ax in enumerate(axs):
    ax.set_xlim([timeplot[0],timeplot[-1]])
    ax.set_ylim([-3.0,3.0])
    ax.axhline([0],ls='dashed')
    ax.set_ylabel("%s [$\degree C$]" % (tsnames[aa]))


savename = "%sTimeseriesPlot.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Plot Lead Lag Relationship

xtks = np.arange(-60,66,6)
fig,ax = plt.subplots(1,1)

plot_lls = [llcorr_pc1,llcorr_pc2]
plotcols = ['goldenrod','cornflowerblue']
plotnames = ["PC1","PC2"]

for ii in range(2):
    llcorr = plot_lls[ii]
    lagmax = lags[np.argmax(np.abs(llcorr))]
    label  = "%s (Max Corr @ Lag %02i Months)" % (plotnames[ii],lagmax)
    
    ax.plot(lags,llcorr,marker="o",c=plotcols[ii],label=label,markersize=2.5)

# Uncomment to check that they are uncorrelated at lag 0
#ax.plot(lags,llcorr_interpc,label="PC1 vs PC2",color="c",ls='dashed')

ax = viz.add_axlines(ax)
#ax.axhline([0],ls='solid',lw=.55,c='k')
#ax.axvline([0],ls='solid',lw=.55,c='k')

ax.legend()

ax.set_xlim([-60,60])
ax.set_xticks(xtks)
ax.set_ylim([-.3,.3])

ax.set_xlabel("<-- SPGNE SST Leads | ENSO PC Leads --> ")
ax.set_ylabel("Correlation")

savename = "%sSPGNE_ENSO_Leadlag_Relationship.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Plot Lead Lag Relationship of the fluxes

plot_lls  = [llcorr_pc1_flx,llcorr_pc2_flx]
plotcols  = ['goldenrod','cornflowerblue']
plotnames = ["PC1","PC2"]

fig,ax = plt.subplots(1,1)
for ii in range(2):
    llcorr = plot_lls[ii]
    lagmax = lags[np.argmax(np.abs(llcorr))]
    label  = "%s (Max Corr @ Lag %02i Months)" % (plotnames[ii],lagmax)
    
    ax.plot(lags,llcorr,marker="o",c=plotcols[ii],label=label,markersize=2.5)

# Uncomment to check that they are uncorrelated at lag 0
#ax.plot(lags,llcorr_interpc,label="PC1 vs PC2",color="c",ls='dashed')

ax = viz.add_axlines(ax)
#ax.axhline([0],ls='solid',lw=.55,c='k')
#ax.axvline([0],ls='solid',lw=.55,c='k')

ax.legend()

ax.set_xlim([-60,60])
ax.set_xticks(xtks)
ax.set_ylim([-.3,.3])

ax.set_xlabel("<-- SPGNE FLX Leads | ENSO PC Leads --> ")
ax.set_ylabel("Correlation")

savename = "%sSPGNE_ENSO_Leadlag_Relationship_Flux.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Next, try removing each component

lagmaxes  = [2,14]
ilagmaxes = [list(lags).index(ll) for ll in lagmaxes]
pcs_in    = [pc1,pc2]
ntime     = len(pc1)

from scipy import stats

spgne_detrended_manual = []
sggne_model = []
pcin_fit = []
tsin_fit  = []
r2s = []
slopes = []
intercepts  = []
spgne_detrended = []
for ii in range(2):
    lagmax = lagmaxes[ii]
    
    if lagmax <0:
        lagmax = np.abs(lagmax)
        pcin   = pcs_in[ii][:(ntime-lagmax)] # Lag the pc
        tsin   = spgne_aavg[lagmax:]
    else:
        pcin   = pcs_in[ii][lagmax:] # Lead the pc
        tsin   = spgne_aavg[:(ntime-lagmax)]
    print(np.corrcoef(pcin,tsin)[0,1])
    dtout  = proc.detrend_by_regression(tsin,pcin)
    spgne_detrended.append(dtout)
    
    # Now do a manual detrend
    slope, intercept, r_value, p_value, std_err = stats.linregress(tsin,pcin)
    
    model     = pcin * slope + intercept
    spgne_detrended_manual.append(tsin - model)
    tsin_fit.append(tsin)
    pcin_fit.append(pcin)
    r2s.append(r_value)    
    slopes.append(slope)
    intercepts.append(intercept)

# Repeat removal for fxlues
spgne_detrended_flx = []
lagmaxes_flx = [29,-49]
for ii in range(2):
    lagmax     = lagmaxes_flx[ii]
    if lagmax < 0:
        lagmax = np.abs(lagmax)
        pcin   = pcs_in[ii][lagmax:] # Lag the pc
        tsin   = spgne_aavg_flx[:(ntime-lagmax)]
    else:
        pcin   = pcs_in[ii][:(ntime-lagmax)] # Lead the pc
        tsin   = spgne_aavg_flx[lagmax:]
        
    dtout  = proc.detrend_by_regression(tsin,pcin)
    spgne_detrended_flx.append(dtout)  

#%% Visualize Detrending

ii     = 0
vmax   = 3.5

fig,axs = plt.subplots(1,2,constrained_layout=True,figsize=(8,5))

for ii in range(2):
    ax = axs[ii]
    sc = ax.scatter(tsin_fit[ii].data.flatten(),pcin_fit[ii].data.flatten(),
                    c=1979+np.arange(len(tsin_fit[ii]))/12,alpha=0.75)
    
    
    #ax.scatter(spgne_aavg,)
    
    
    ax.set_xlabel("Tropical Pacific PC %i" % (ii+1))
    ax.set_ylabel("SPGNE SST, Lag %i Months" % (lagmaxes[ii]))
    ax.set_title("Correlation=%.2f\n Slope = %.2f, Intercept = %.2e" % (r2s[ii],slopes[ii],intercepts[ii]))
    
    xplot = np.linspace(-3,3,100)
    ax.plot(xplot, xplot*slopes[ii] + intercepts[ii],label="Fit",c='gray',ls='dashed')
    
    ax.set_xlim([-vmax,vmax])
    ax.set_ylim([-vmax,vmax])
    
    ax = viz.add_axlines(ax)
    #ax.axhline([0],c='k',lw=0.55)
    #ax.axvline([0],c='k',lw=0.55)
    ax.legend()
    ax.set_aspect(1)

cb = viz.hcbar(sc,ax=axs.flatten(),pad=0.04,fraction=0.045)
cb.set_label("Time")

savename = "%sENSO-SPGNE-Fit-Scatterplot.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Visualize Timeseries

fig,axs = plt.subplots(2,1,figsize=(12.5,8))

for ii in range(2):
    ax = axs[ii]
    
    plotvar = tsin_fit[ii]
    ax.plot(plotvar.time,plotvar,label="SPGNE SST (Raw)",color="k",lw=2.5)
    
    plotvar = spgne_detrended_manual[ii]
    ax.plot(plotvar.time,plotvar,label="SPGNE SST (ENSO Removed)",color="blue",lw=1.5)
    
    plotvar = pcin_fit[ii]
    ax.plot(plotvar.time,plotvar,label="Tropical Pacific PC %i" % (ii+1),color="red",lw=2)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("SST Anomaly (degree C)")
    
    ax.axhline([0],c='k',lw=0.55)
    ax.set_ylim([-3.5,3.5])
    ax.set_xlim([spgne_aavg.time[0],spgne_aavg.time[-1]])
    
    ax.legend(ncol=2)

savename = "%sENSO-SPGNE-Fit-Detrend-Visualization.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%%

    
#%% Investigate Characteristics of the Detrended Timeseries

# Set up Timeseries
ts_in    = [spgne_aavg,spgne_detrended[0].sst,spgne_detrended[1].sst,pc1,pc2]
tsnames  = ["Raw","Remove PC1, Lag 2","Remove PC2, Lag 14","PC1","PC2"]
tscolors = ["k","goldenrod","cornflowerblue",'darkred','midnightblue']

ssts     = [ts.data.squeeze() for ts in ts_in]
specout  = scm.quick_spectrum(ssts,nsmooth=2,pct=0.10,return_dict=True)    

# Also compute spectra for fluxes
flxs            = [spgne_aavg_flx,spgne_detrended_flx[0].qnet,spgne_detrended_flx[1].qnet]
flxs            = [flx.data.squeeze() for flx in flxs]
specout_flx     = scm.quick_spectrum(flxs,nsmooth=2,pct=0.10,return_dict=True) 

#%% Make Function

def init_specplot(ax,decadal_focus=False,xlab=True,ylab=True,xperlab=True,
                  fsz_axis=14,fsz_ticks=14):
    
    if decadal_focus:
        xper            = np.array([20,10,5,1,0.5])
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
        ax.set_ylabel("Power [$\degree C ^2 / cycle \, per \, mon$]",fontsize=fsz_axis)

    ax2 = ax.twiny()
    ax2.set_xlim([xper_ticks[0],0.5])
    ax2.set_xscale('log')
    ax2.set_xticks(xper_ticks,labels=xper,fontsize=fsz_ticks)
    if xperlab:
        ax2.set_xlabel("Period (Years)",fontsize=fsz_ticks)
    
    return ax,ax2
    

#%% Plot Power Spectra

fsz_ticks =14
fsz_axis = 14
fsz_legend=16

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(10,4.5))
obs_cutoff = 10 # in years
obs_cutoff = 1/(obs_cutoff*12)

decadal_focus = False
if decadal_focus:
    xper            = np.array([20,10,5,1,0.5])
else:
    xper            = np.array([40,10,7,5,3,2,1,0.5])

xper_ticks      = 1 / (xper*12)
dtmon_fix       = 60*60*24*30

for ii in range(5):
    plotspec        = specout['specs'][ii] / dtmon_fix
    plotfreq        = specout['freqs'][ii] * dtmon_fix
    CCs             = specout['CCs'][ii] / dtmon_fix
    
    color_in = tscolors[ii]
    
    
    # if ii == 3:
        
    #     iplot_hifreq = np.where(plotfreq > obs_cutoff)[0]
    #     ax.loglog(plotfreq,plotspec,label="",c=color_in,ls='dashed',lw=1.5)
    #     plotfreqhi     = plotfreq[iplot_hifreq]
    #     plotspechi     = plotspec[iplot_hifreq]
        
    #     ax.loglog(plotfreqhi,plotspechi,lw=4,label=expnames_long[ii],c=color_in)
        
    # else:
    if ii <3:
        ls='solid'
    else:
        ls='dashed'
        
    ax.loglog(plotfreq,plotspec,lw=4,label=tsnames[ii],c=color_in,ls=ls)
        
        
    
    # # Plot the 95% Confidence Interval (for stochastic model output)
    # if ii < 3:
        
    #     plotspec1 = cesm_specdicts[ii]['specs'] / dtmon_fix
    #     plotfreq1 = cesm_specdicts[ii]['freqs'][0,:] * dtmon_fix
    #     bnds      = np.quantile( plotspec1 ,[0.025,0.975],axis=0)
    #     ax.fill_between(plotfreq1,bnds[0],bnds[1],color=expcols[ii],alpha=0.15,zorder=1)
        
    # else: # plot for ERA5
    ax.loglog(plotfreq,CCs[:,0],ls='solid',lw=0.5,c=tscolors[ii])
    ax.loglog(plotfreq,CCs[:,1],ls='dashed',lw=0.9,c=tscolors[ii])

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


ax.set_xlabel("Frequency (1/month)",fontsize=fsz_axis)
ax.set_ylabel("Power [$\degree C ^2 / cycle \, per \, mon$]",fontsize=fsz_axis)

ax2 = ax.twiny()
ax2.set_xlim([xper_ticks[0],0.5])
ax2.set_xscale('log')
ax2.set_xticks(xper_ticks,labels=xper,fontsize=fsz_ticks)
ax2.set_xlabel("Period (Years)",fontsize=fsz_ticks)

# # Plot Confidence Interval (ERA5)
# alpha           = 0.05
# cloc_era        = [8e-2,1e-2]
# dof_era         = metrics_out['dofs'][-1]
# cbnds_era       = proc.calc_confspec(alpha,dof_era)
# proc.plot_conflog(cloc_era,cbnds_era,ax=ax,color=dfcol,cflabel=r"95% Confidence") #+r" (dof= %.2f)" % dof_era)

# ax.set_ylim(ylim)

ax.legend(fontsize=fsz_legend,framealpha=0.5,edgecolor='none')

# for ax in [ax,ax2]:
#     ax.tick_params(labelsize=fsz_ticks)
    
# viz.label_sp(1,alpha=0.15,ax=ax,fontsize=fsz_title,y=1.17,x=-.1,
#              fontcolor=dfcol)


savename = "%sPower_Spectra_ENSO_Removal.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Check for Fluxes


fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(10,4.5))

ax,ax2 = init_specplot(ax)

for ii in range(3):
    plotspec        = specout_flx['specs'][ii] / dtmon_fix
    plotfreq        = specout_flx['freqs'][ii] * dtmon_fix
    CCs             = specout_flx['CCs'][ii] / dtmon_fix
    
    if len(plotspec) != len(plotfreq):
        if len(plotspec) < len(plotfreq):
            plotfreq = plotfreq[:-1]
        else:
            plotspec = plotspec[:-1]
        
    
    color_in = tscolors[ii]
    
    

    # else:
    if ii <3:
        ls='solid'
    else:
        ls='dashed'
        
    ax.loglog(plotfreq,plotspec,lw=4,label="SPGNE SST " + tsnames[ii],c=color_in,ls=ls)
    
    # else: # plot for ERA5
    #ax.loglog(plotfreq,CCs[:,0],ls='solid',lw=0.5,c=tscolors[ii])
    #ax.loglog(plotfreq,CCs[:,1],ls='dashed',lw=0.9,c=tscolors[ii])
 
ax.legend()


#%% Redo for SST, but separate the plots


fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(10,8))

for a in range(2):
    ax = axs[a]
    ax1,ax2 = init_specplot(ax)
    axs[a] = ax1
    

# Part 1: Plot SST
ax = axs[0]
for ii in range(3):
    plotspec        = specout['specs'][ii] / dtmon_fix
    plotfreq        = specout['freqs'][ii] * dtmon_fix
    CCs             = specout['CCs'][ii] / dtmon_fix
    
    color_in = tscolors[ii]
    

    # else:
    if ii <3:
        ls='solid'
    else:
        ls='dashed'
        
    ax.loglog(plotfreq,plotspec,lw=4,label="SPGNE SST " + tsnames[ii],c=color_in,ls=ls)
    
    # else: # plot for ERA5
    ax.loglog(plotfreq,CCs[:,0],ls='solid',lw=0.5,c=tscolors[ii])
    ax.loglog(plotfreq,CCs[:,1],ls='dashed',lw=0.9,c=tscolors[ii])
    
    
 
ax.legend()

ax = axs[1]
# Part 2: Plot for PC1
for ii in [3,4]:
    plotspec        = specout['specs'][ii] / dtmon_fix
    plotfreq        = specout['freqs'][ii] * dtmon_fix
    CCs             = specout['CCs'][ii] / dtmon_fix
    
    color_in = tscolors[ii]
    

    # else:
    if ii <3:
        ls='solid'
    else:
        ls='dashed'
        
    ax.loglog(plotfreq,plotspec,lw=4,label="Tropical Pacific SST " + tsnames[ii],c=color_in,ls=ls)
    
    # else: # plot for ERA5
    ax.loglog(plotfreq,CCs[:,0],ls='solid',lw=0.5,c=tscolors[ii])
    ax.loglog(plotfreq,CCs[:,1],ls='dashed',lw=0.9,c=tscolors[ii])
ax.legend()

savename = "%sPower_Spectra_ENSO_Removal_separate.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% For a given PC, repeat removal across several lags
sst_bypc = []
specs_bypc = []
for ii in range(2):
    
    sst_removed = []
    for lagmax in lags:
        
        
        if lagmax <0:
            pcin   = pcs_in[ii][:(ntime-np.abs(lagmax))] # Lag the pc
            tsin   = spgne_aavg[np.abs(lagmax):]
        else:
            pcin   = pcs_in[ii][lagmax:] # Lead the pc
            tsin   = spgne_aavg[:(ntime-lagmax)]
        print(np.corrcoef(pcin,tsin)[0,1])
        dtout  = proc.detrend_by_regression(tsin,pcin)
        sst_removed.append(dtout)
    
    sst_bypc.append(sst_removed)
    sstslag        = [sst['sst'].data.squeeze() for sst in sst_removed]
    specsout_bylag = [scm.quick_spectrum([ssts,],nsmooth=2,pct=0.10,return_dict=True) for ssts in sstslag]
    specs_bypc.append(specsout_bylag)

#%% Plot Power Spectra


fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(10,8))

for a in range(2):
    ax = axs[a]
    ax1,ax2 = init_specplot(ax)
    axs[a] = ax1
    

# Part 1: PC1
ax = axs[0]
for ii in [0,1]:
    
    plotspec        = specout['specs'][ii] / dtmon_fix
    plotfreq        = specout['freqs'][ii] * dtmon_fix
    CCs             = specout['CCs'][ii] / dtmon_fix
    
    color_in = tscolors[ii]
    

    # else:
    if ii <3:
        ls='solid'
    else:
        ls='dashed'
        
    ax.loglog(plotfreq,plotspec,lw=4,label="SPGNE SST " + tsnames[ii],c=color_in,ls=ls)
    
    # else: # plot for ERA5
    ax.loglog(plotfreq,CCs[:,0],ls='solid',lw=0.5,c=tscolors[ii])
    ax.loglog(plotfreq,CCs[:,1],ls='dashed',lw=0.9,c=tscolors[ii])
ax.legend()
for ll in range(len(lags)):
    ii = 0
    plotspec        = specs_bypc[ii][ll]['specs'][0] / dtmon_fix
    plotfreq        = specs_bypc[ii][ll]['freqs'][0] * dtmon_fix
    if len(plotspec) != len(plotfreq):
        if len(plotspec) < len(plotfreq):
            plotfreq = plotfreq[:-1]
        else:
            plotspec = plotspec[:-1]
    ax.loglog(plotfreq,plotspec,lw=4,label="",c='gray',alpha=1)
    #CCs             = specout['CCs'][ii] / dtmon_fix
    


ax = axs[1]
# Part 2: Plot for PC1
for ii in [0,2]:
    plotspec        = specout['specs'][ii] / dtmon_fix
    plotfreq        = specout['freqs'][ii] * dtmon_fix
    CCs             = specout['CCs'][ii] / dtmon_fix
    
    color_in = tscolors[ii]
    

    # else:
    if ii <3:
        ls='solid'
    else:
        ls='dashed'
        
    ax.loglog(plotfreq,plotspec,lw=4,label="Tropical Pacific SST " + tsnames[ii],c=color_in,ls=ls)
    
    # else: # plot for ERA5
    ax.loglog(plotfreq,CCs[:,0],ls='solid',lw=0.5,c=tscolors[ii])
    ax.loglog(plotfreq,CCs[:,1],ls='dashed',lw=0.9,c=tscolors[ii])
ax.legend()
for ll in range(len(lags)):
    ii = 1
    plotspec        = specs_bypc[ii][ll]['specs'][0] / dtmon_fix
    plotfreq        = specs_bypc[ii][ll]['freqs'][0] * dtmon_fix
    if len(plotspec) != len(plotfreq):
        if len(plotspec) < len(plotfreq):
            plotfreq = plotfreq[:-1]
        else:
            plotspec = plotspec[:-1]
    ax.loglog(plotfreq,plotspec,lw=4,label="",c='gray',alpha=1)
    #CCs             = specout['CCs'][ii] / dtmon_fix
    
savename = "%sPower_Spectra_ENSO_Removal_separate_bylag.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')





#%% Compute Power Spectra


    
    


#%% Scrap Below...




#%% Calculate Lead Lag relationship with each point (note this was too slow...)

ntime,nlat,nlon = sst_spgne.shape
rr     = np.zeros((len(lags),nlat,nlon)) * np.nan
rr_flx = rr.copy()
pc1_detrend = sp.signal.detrend(pc1.data)
for o in tqdm(range(nlon)):
    
    for a in range(nlat):
        
        ptdata        = sst_spgne.isel(lon=o,lat=a).data
        rr[:,a,o]     = ut.calc_lag_regression_1d(pc1_detrend,ptdata,lags,correlation=False)
        
        ptdata_flx    = flx_spgne.isel(lon=o,lat=a).data
        rr_flx[:,a,o] = ut.calc_lag_regression_1d(pc1_detrend,ptdata_flx,lags,correlation=False)
        

coords     = dict(lag=lags,lat=sst_spgne.lat,lon=sst_spgne.lon)
ds_sst_pc1 = xr.DataArray(rr,coords=coords,dims=coords,name='sst_pc1')
ds_flx_pc1 = xr.DataArray(rr_flx,coords=coords,dims=coords,name='flx_pc1')

#%% Make Lag Regression Plots for SST

# def make_da_like(array_in,ds_reference,make_da=True):
#     # For a matching data array and reference dataset, return the coords
#     reference_shape = np.array(ds_reference.shape)
#     target_shape    = np.array(array_in.shape)
#     coord_match     = {}
#     for ii in target_shape:
    

bbox_plot   = [-50,-10,50,65]
proj        = ccrs.PlateCarree()
cints_rho   = np.arange(-1,1.1,0.1)

ii = 0
for lag in tqdm(lags):
    # Initialize Map
    fig,ax,_    = viz.init_orthomap(1,1,bbox_plot,centlon=-30,centlat=55,figsize=(12.5,4.5))
    ax          = viz.add_coast_grid(ax,bbox=bbox_plot,proj=proj,fill_color='k')
    
    plotvar     = ds_sst_pc1.sel(lag=lag) #[ll,:,:]
    pcm         = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                                transform=proj,vmin=-.5,vmax=.5,cmap='cmo.balance')
    
    cl          = ax.contour(plotvar.lon,plotvar.lat,plotvar,linewidths=0.75,
                                transform=proj,levels=cints_rho,colors="k")
    ax.clabel(cl,fontsize=12)
    cb          = viz.hcbar(pcm,ax=ax,fontsize=16)
    cb.set_label("SST - PC1 Correlation, Lag %02i" % lag,fontsize=18)
    
    figname     = "%sSPGNE_ENSO_SST_PC1_Correlation_iter%03i_lag%02i.png" % (figpath,ii,lag)
    plt.savefig(figname,dpi=150,bbox_inches='tight')
    
    ii += 1

#%% Look at the Value and Month of Maximum Lag
# Note: Stopped because I wanted the maximum relationship regardless of sign...

fig,axs,_=viz.init_orthomap(1,2,bbox_plot,centlon=-30,centlat=55,figsize=(12.5,4.5))

# for ii in range(2):
    
#     ax = axs[ii]
#     ax = viz.add_coast_grid(ax,bbox=bbox_plot,proj=proj,fill_color='k')
    
#     if ii == 0:
#         plotvar = np.abs(ds_sst_pc1).max('lag') #* np.argmax(np.abs(ds_sst_pc1).max('lag'))
#         vmax    = 0.25
#     else:
#         plotvar = ds_sst_pc1.idxmax('lag')
#         vmax    = 60
        
#     pcm         = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
#                                 transform=proj,cmap='cmo.balance',
#                                 vmin=-vmax,vmax=vmax,)
#     cb          = viz.hcbar(pcm,ax=ax,fontsize=16)


#%% Take Average Over SPGNE

sstspg        = proc.sel_region_xr(ds_sst['SST'],bbox_spgne)
sstspg_noenso = proc.sel_region_xr(ds_sst_noenso['sst'],bbox_spgne)

#%% Note this turned out to be too raw

fig,ax        = plt.subplots(1,1,constrained_layout=True,figsize=(12.5,4.5))

plotvar       = proc.area_avg_cosweight(sstspg)

ax.plot(plotvar.time,plotvar,label="SST With ENSO")

# plotvar = proc.area_avg_cosweight(sstspg_noenso)
# ax.plot(plotvar.valid_time,plotvar,label="SST Without ENSO")

ax.legend()



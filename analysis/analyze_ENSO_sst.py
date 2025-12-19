#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Analyze ENSO-related SST

works with output from [integrate_ENSO_forcing] and [construct_ENSO_forcing]

Created on Wed Dec 17 16:28:11 2025

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
#%% Part 1: Data Loading + Preprocessing
# ========================================================

#% Load Raw SST and preprocess

# Copied largely from area_average_variance_ERA5

# # Load SST
dpath_era   = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
ncname_era  = dpath_era + "ERA5_sst_NAtl_1979to2024.nc"
ds_era      = xr.open_dataset(ncname_era).sst
ds_era      = proc.sel_region_xr(ds_era,bbox_spgne).load()

# Deseason
dsa_era     = proc.xrdeseason(ds_era)

# Detrend by Regression to the global Mean
ds_gmsst        = xr.open_dataset(dpath_era + nc_gmsst).GMSST_MaxIce.load()
dtout           = proc.detrend_by_regression(dsa_era,ds_gmsst,regress_monthly=True)
sst_era         = dtout.sst

#%% Load ENSO Index

# Load ENSO Files
ensonc        = "ERA5_ensotest_ENSO_detrendGMSSTmon_pcs3_1979to2024.nc"
ensopath      = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/enso/"
ensoid        = xr.open_dataset(ensopath + ensonc).load()

# Flatten PCs (from regress_enso_SPGNE)
pc1     = ensoid.pcs.isel(pc=0).data.flatten()
tcoords = dict(time=sst_era.time)
pc1     = xr.DataArray(pc1,coords=tcoords,dims=tcoords,name="enso")

pc2     = ensoid.pcs.isel(pc=1).data.flatten()
pc2     = xr.DataArray(pc2,coords=tcoords,dims=tcoords,name="enso")


#%% Load ENSO Forcing from [construct_ENSO_forcing]

outpath  = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/enso/"
ensolag  = 1
flxnames = ['qnet','Fprime']

forcings = []
for ii in range(2):
    
    outname = "%sERA5_%s_ENSO_related_forcing_ensolag%i.nc" % (outpath,flxnames[ii],ensolag)
    ds = xr.open_dataset(outname)[flxnames[ii]]#.load()
    ds = proc.sel_region_xr(ds,bbox_spgne).load()
    
    if ii == 0:
        ds = ds# *-1
    
    forcings.append(ds)

#%% Load Simulated SST response to ENSO from [integrate_enso_forcing]

lag_bymonth    = True
forcing_names  = ["PC1","PC2","PCsum"]
vnames_out     = []
for ii in range(2):
    for nn in tqdm(range(3)):
        vnames_out.append("%s_%s" % (flxnames[ii],forcing_names[nn]))

enso_sst = []
for jj in range(6):
    savename = "%sERA5_SST_ENSO_Component_%s.nc" % (outpath,vnames_out[jj])
    ds = xr.open_dataset(savename).SST.load()
    if "qnet" in savename:
        ds = ds #* -1
    
    
    enso_sst.append(ds)


"""
enso_sst - Simulated ENSO-related SPGNE SSTs
forcings - Qnet and Fprime forcings related to ENSO
p1/pc2   - ENSO Indices
sst_era  - SST in ERA5
"""
# ========================================================      
#%% Part 2: Analysis
# ========================================================

# Take Area Averages
ssts_sim     = [proc.area_avg_cosweight(ds) for ds in enso_sst]
sst_spgne    = proc.area_avg_cosweight(sst_era)
qnet_spgne   = proc.area_avg_cosweight(forcings[0])
fprime_spgne = proc.area_avg_cosweight(forcings[1])


# Separate Forcings
forcing_names = ["ENSO PC","Qnet","Fprime"]
forcing_pc1   = [pc1,qnet_spgne.isel(pc=0),fprime_spgne.isel(pc=0)]
forcing_pc2   = [pc2,qnet_spgne.isel(pc=1),fprime_spgne.isel(pc=1)]




#%% 2A. Just Plot the Timeseries

#% SSTs
fsz_axis = 16



simcols = ["cornflowerblue","blue","midnightblue",
           "hotpink","red","brown",
           ]

fig,ax = plt.subplots(1,1,figsize=(12.5,4),constrained_layout=True)



ax.axhline([0],ls='solid',lw=0.55,c='k')



ax.tick_params(labelsize=14)
ax.set_ylabel(r"SST Anomaly [$\degree$C]",fontsize=fsz_axis)

# Plot each other timeseries
for jj in range(6):
    plotvar = ssts_sim[jj] * 3
    ax.plot(plotvar.time,plotvar,label=vnames_out[jj],lw=3,c=simcols[jj])

# Plot Raw SST
plotvar = sst_spgne
ax.plot(plotvar.time,plotvar,c="k",label="SST Raw",lw=3)
ax.set_xlim([plotvar.time[0],plotvar.time[-1]])

ax.legend(ncol=3,fontsize=14)


#%% Plot PC1 SST and Fluxes

fig,axs =  plt.subplots(3,1,figsize=(12.5,8),constrained_layout=True)

for ax in axs:
    ax.axhline([0],ls='solid',lw=0.55,c='k')
    ax.set_xlim([plotvar.time[0],plotvar.time[-1]])
    ax.tick_params(labelsize=14)
    
# First Axes, Plot ENSO SST and Simulated SSTs
ax = axs[0]
plotvar = pc1
ax.plot(plotvar.time,plotvar,c="red",label="PC1",lw=3)
ax.set_ylabel("PC1 [degC]",fontsize=fsz_axis)

# SST Response
ax = axs[2]
for ss in [0,3]:
    
    plotvar = ssts_sim[ss]
    
    ax.plot(plotvar.time,plotvar,c=simcols[ss],label=vnames_out[ss],lw=2)
ax.set_ylabel("SST Response [degC]",fontsize=fsz_axis)
ax.legend()

# Qnet and Fprime
ax = axs[1]
for ii in [1,2]:
    plotvar = forcing_pc1[ii]
    ax.plot(plotvar.time,plotvar,label=flxnames[ii-1],lw=3)
    
ax.set_ylabel("Forcing [W/m2]",fontsize=fsz_axis)
ax.legend()



#%% Plot PC2 SST and Fluxes

fig,axs =  plt.subplots(3,1,figsize=(12.5,8),constrained_layout=True)

for ax in axs:
    ax.axhline([0],ls='solid',lw=0.55,c='k')
    ax.set_xlim([plotvar.time[0],plotvar.time[-1]])
    ax.tick_params(labelsize=14)
    
# First Axes, Plot ENSO SST and Simulated SSTs
ax = axs[0]
plotvar = pc2
ax.plot(plotvar.time,plotvar,c="red",label="PC1",lw=3)
ax.set_ylabel("PC2 [degC]",fontsize=fsz_axis)

# SST Response
ax = axs[2]
for ss in [1,4]:
    
    plotvar = ssts_sim[ss]
    
    ax.plot(plotvar.time,plotvar,c=simcols[ss],label=vnames_out[ss],lw=2)
ax.set_ylabel("SST Response [degC]",fontsize=fsz_axis)
ax.legend()

# Qnet and Fprime
ax = axs[1]
for ii in [1,2]:
    plotvar = forcing_pc2[ii]
    ax.plot(plotvar.time,plotvar,label=flxnames[ii-1],lw=3)
    
ax.set_ylabel("Forcing [W/m2]",fontsize=fsz_axis)
ax.legend()

#%% Remove Each case from SST


#for mult in np.arange(20):
    
mult = 1

sst_noenso = []
for jj in range(6):
    if lag_bymonth:
        sst_in = sst_spgne.data
    else:
        sst_in = sst_spgne.data[12:]
    sst_rm = sst_in - ssts_sim[jj].data * mult
    sst_noenso.append(sst_rm)
    
    
#sst_noenso = [sst_spgne - ds for ds in ssts_sim]

#%% Now Make the spectr

if lag_bymonth:
    
    ssts_in  = [sst_spgne.data,] + sst_noenso
else:
    ssts_in  = [sst_spgne.data[12:],] + sst_noenso
specout  = scm.quick_spectrum(ssts_in,nsmooth=2,pct=0.10,return_dict=True)    


#%% Plot the spectra


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
    

#%%

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(10,4.5))
ax,ax2 = init_specplot(ax,decadal_focus=True)

dtmon_fix       = 60*60*24*30

for ii in range(7):
    
    plotspec        = specout['specs'][ii] / dtmon_fix
    plotfreq        = specout['freqs'][ii] * dtmon_fix
    CCs             = specout['CCs'][ii] / dtmon_fix
    
    
    if ii == 0:
        color_in = "k"
        label = "SST Raw"
        ls = 'solid'
    else:
        color_in = simcols[ii-1]
        label=vnames_out[ii-1]
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

ax.set_title("Mult = %02i" % mult,fontsize=20)
ax.set_ylim([1e-3,1e3])
    
savename = "%sENSO_Removal_Power_Spectra_SPGNE_SST_mult%02i.png" % (figpath,mult)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#color_in = tscolors[ii]


    
# ======================================================
#%% Look at the spectra of the SST response and forcing
# ======================================================

pc1ts           = [ds.data for ds in forcing_pc1]
pc2ts           = [ds.data for ds in forcing_pc2]

specout_pc1     = scm.quick_spectrum(pc1ts,nsmooth=2,pct=0.10,return_dict=True)
specout_pc2     = scm.quick_spectrum(pc2ts,nsmooth=2,pct=0.10,return_dict=True)


sstresp         = [ssts_sim[jj].data for jj in range(6)]
specout_sstresp = scm.quick_spectrum(sstresp,nsmooth=2,pct=0.10,return_dict=True)

#%% Make Spectra Plot (Forcing)

fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(10,8))

specout = [specout_pc1,specout_pc2]
speccols = ["red",'cornflowerblue','orange']
for a,ax in enumerate(axs):
    ax,ax2 = init_specplot(ax,decadal_focus=True)
    
    for ii in range(3):
        
        plotspec        = specout[a]['specs'][ii] / dtmon_fix
        plotfreq        = specout[a]['freqs'][ii] * dtmon_fix
        CCs             = specout[a]['CCs'][ii] / dtmon_fix
        

        color_in = speccols[ii]
        label    = forcing_names[ii] 
        ls = 'solid'
        
        ax.loglog(plotfreq,plotspec,lw=4,label=label,c=color_in,ls=ls)
        
        ax.loglog(plotfreq,CCs[:,0],ls='solid',lw=0.5,c=color_in)
        ax.loglog(plotfreq,CCs[:,1],ls='dashed',lw=0.9,c=color_in)
        
        
        
    ax.legend()
        

#%% Make Spectra Plot (Response)
        
fig,axs = plt.subplots(3,1,constrained_layout=True,figsize=(10,10))

#specout  = [specout_pc1,specout_pc2]

speccols = ['cornflowerblue','orange']
for a,ax in enumerate(axs):
    ax,ax2 = init_specplot(ax,decadal_focus=True)
    
    if a == 0:
        loopii = [0,3]
    elif a == 1:
        loopii = [1,4]
    else:
        loopii = [2,5]
    
    for jj in range(2):
        ii = loopii[jj]
        
        
        
        plotspec        = specout_sstresp['specs'][ii] / dtmon_fix
        plotfreq        = specout_sstresp['freqs'][ii] * dtmon_fix
        CCs             = specout_sstresp['CCs'][ii] / dtmon_fix
        

        color_in = speccols[jj]
        label    = forcing_names[jj+1] 
        ls = 'solid'
        
        ax.loglog(plotfreq,plotspec,lw=4,label=label,c=color_in,ls=ls)
        
        ax.loglog(plotfreq,CCs[:,0],ls='solid',lw=0.5,c=color_in)
        ax.loglog(plotfreq,CCs[:,1],ls='dashed',lw=0.9,c=color_in)
        
        
    # Also Plot SPGNE SST
    ii = 0
    plotspec        = specout['specs'][ii] / dtmon_fix
    plotfreq        = specout['freqs'][ii] * dtmon_fix
    CCs             = specout['CCs'][ii] / dtmon_fix
    color_in = "k"
    label    = "SST Raw"
    ls       = 'solid'
    ax.loglog(plotfreq,plotspec,lw=4,label=label,c=color_in,ls=ls)
        
        
        
    ax.legend()

# ========================================================      
#%% Try Different Approach: Pointwise Removal
# ========================================================

sstpointremove = []
for ii in range(6):
    sstin   = sst_era
    sstenso = enso_sst[ii]
    sstdiff = sstin.data-sstenso.data
    coords  = dict(time=sstenso.time,lat=sstenso.lat,lon=sstenso.lon)
    sstdiff = xr.DataArray(sstdiff,coords=coords,dims=coords,name='sst')
    sstpointremove.append(sstdiff)
    
# Take Area Average
sstpointremove_aavg = [proc.area_avg_cosweight(ds) for ds in sstpointremove]


sstresp_pa          = [ds.data for ds in sstpointremove_aavg]
specout_sstresp_pa = scm.quick_spectrum(sstresp_pa,nsmooth=2,pct=0.10,return_dict=True)

#%% Look at Difference



        
fig,axs = plt.subplots(3,1,constrained_layout=True,figsize=(10,10))

#specout  = [specout_pc1,specout_pc2]
speccols = ['cornflowerblue','orange']
for a,ax in enumerate(axs):
    ax,ax2 = init_specplot(ax,decadal_focus=True)
    
    if a == 0:
        loopii = [0,3]
    elif a == 1:
        loopii = [1,4]
    else:
        loopii = [2,5]
    
    # # Plot Area-Average Removal
    # for ii in range(2):
        
    #     plotspec        = specout_sstresp['specs'][ii] / dtmon_fix
    #     plotfreq        = specout_sstresp['freqs'][ii] * dtmon_fix
    #     CCs             = specout_sstresp['CCs'][ii] / dtmon_fix
        

    #     color_in = speccols[ii]
    #     label    = forcing_names[ii] 
    #     ls = 'solid'
        
    #     ax.loglog(plotfreq,plotspec,lw=4,label=label,c=color_in,ls=ls)
        
    #     ax.loglog(plotfreq,CCs[:,0],ls='solid',lw=0.5,c=color_in)
    #     ax.loglog(plotfreq,CCs[:,1],ls='dashed',lw=0.9,c=color_in)
    
    
    # Also Plot SPGNE SST
    ii = 0
    plotspec        = specout['specs'][ii] / dtmon_fix
    plotfreq        = specout['freqs'][ii] * dtmon_fix
    CCs             = specout['CCs'][ii] / dtmon_fix
    color_in = "k"
    label    = "SST Raw"
    ls       = 'solid'
    ax.loglog(plotfreq,plotspec,lw=4,label=label,c=color_in,ls=ls)
    
    # Plot Pointwise Removal
    for jj in range(2):
        ii = loopii[jj]
        
        plotspec        = specout_sstresp_pa['specs'][ii] / dtmon_fix
        plotfreq        = specout_sstresp_pa['freqs'][ii] * dtmon_fix
        CCs             = specout_sstresp_pa['CCs'][ii] / dtmon_fix
        

        color_in = speccols[ii]
        label    = forcing_names[ii] 
        ls = 'dashed'
        
        ax.loglog(plotfreq,plotspec,lw=4,label=label,c=color_in,ls=ls)
        
        ax.loglog(plotfreq,CCs[:,0],ls='solid',lw=0.5,c=color_in)
        ax.loglog(plotfreq,CCs[:,1],ls='dashed',lw=0.9,c=color_in)
        
        
        
        

        
        
        
    ax.legend()
    
#%%


#%% Load ENSO Forcing Patterns from [construct_ENSO_forcing]

#outpath  = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/enso/"
#ensolag  = 1
#flxnames = ['qnet','Fprime']

patterns = []
for ii in range(2):
    
    outname = "%sERA5_%s_ENSO_related_pattern_ensolag%i.nc" % (outpath,flxnames[ii],ensolag)
    ds = xr.open_dataset(outname)[flxnames[ii]]#.load()
    ds = proc.sel_region_xr(ds,bbox_spgne).load()
    
    if ii == 0:
        ds = ds# *-1
    
    patterns.append(ds)
    
#%% Plot the 

bbox_plot   = [-50,-10,50,65]
proj        = ccrs.PlateCarree()

def init_spgne():
    fig,ax,_    = viz.init_orthomap(1,1,bbox_plot,centlon=-30,centlat=55,figsize=(12.5,4.5))
    ax          = viz.add_coast_grid(ax,bbox=bbox_plot,proj=proj,fill_color='k')
    return fig,ax


#%%

nn = 0
im = 0
ii = 0

for ii in range(2):
    for nn in range(2):
        for im in tqdm(range(12)):
            cints_flx = np.arange(-30,35,5)
            
            fig,ax  = init_spgne()
            
            plotvar = patterns[ii].isel(pc=nn,month=im)
            pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                                    transform=proj,cmap='cmo.balance',vmin=-20,vmax=20)
            
            cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,colors='k',
                                    transform=proj,linewidths=0.75,levels=cints_flx)
            
            ax.clabel(cl)
            
            ax.set_title("%s PC %s, Month %02i" % (flxnames[ii],nn+1,im+1))
            cb      = viz.hcbar(pcm,ax=ax)
            
            
            savename="%sSPGNE_ENSO_Response_Lag1_%s_PC%i_month%02i.png" % (figpath,flxnames[ii],nn+1,im+1)
            plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)
        
        



    
#%%



#ssts_in = []








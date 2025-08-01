#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Examine the Structure of Large Scale Forcing over the region for Fprime

Created on Tue Jul 29 15:23:23 2025

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


#%% Load Fprime for ERA5

# Load Fprime
fpath           = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/reanalysis/proc/NATL_proc_obs/"
fnc             = "ERA5_Fprime_QNET_timeseries_QNETpilotObsAConly_nroll0_NAtl.nc"
ds_fprime       = xr.open_dataset(fpath + fnc).load()


# Load Ice Mask
dsmask_era5     = dl.load_mask(expname='ERA5').mask
dsmaskplot      = xr.where(np.isnan(dsmask_era5),0,1)


figpath  = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20250730/"
proc.makedir(figpath)


#%% Mask and prepare for calculations

fprime = ds_fprime.Fprime.squeeze() * dsmask_era5

#fprime = proc.xrdeseason(fprime)

#%% Perform EOF Analysis (copied from NHFLX_EOF_monthly)
# Basically transformed it into a function


def calc_monthly_eof(daf,bboxeof,N_mode=None,concat_ens=True,mask=None,bbox_check=None):
    
    if 'ens' not in daf.dims: # Add dummy ens variable
        daf = daf.expand_dims(dim={'ens':[0,]},axis=1)
        print("Adding ens dim")
        
    
    flxa     = daf # [Time x Ens x Lat x Lon] # Anomalize variabless
    
    # Apply area weight
    wgt    = np.sqrt(np.cos(np.radians(daf.lat.values))) # [Lat]
    flxwgt = flxa * wgt[None,None,:,None]
    
    # Apply Max if needed
    if mask is not None:
        print("Applying provided mask...")
        flxwgt = flxwgt * mask
    
    # Select Region
    flxreg = proc.sel_region_xr(flxwgt,bboxeof)
    
    flxout     = flxreg.values
    ntime,nens,nlatr,nlonr = flxout.shape
    if concat_ens:
        # IMPORTANT NOTE (implement fix later)
        # Variable must be stacked as [ens x time x otherdims]
        if flxout.shape[0] != nens:
            ens_reshape_flag = True
            print("Warning, since ensemble dimension is NOT first, temporarily permuting array to ens x time")
            flxout = flxout.transpose(1,0,2,3)
        else:
            ens_reshape_flag = False
        print("Stacking Dimensions")
        flxout = flxout.reshape(nens*ntime,1,nlatr,nlonr)
        ntime,nens,nlatr,nlonr = flxout.shape
    npts       = nlatr*nlonr
    nyr        = int(ntime/12)
    if N_mode is None: # Set EOFs to number of years
        N_mode=nyr
    
    # Repeat for full variable
    flxout_full= flxa.values
    _,_,nlat,nlon=flxout_full.shape
    if ens_reshape_flag:
        print("Permuting full variable")
        print("\tOriginal Shape %s" % str(flxout_full.shape))
        flxout_full = flxout_full.transpose(1,0,2,3)
        print("\tNew Shape %s" % str(flxout_full.shape))
    npts_full  = nlat*nlon
    if concat_ens:
        flxout_full = flxout_full.reshape(ntime,1,nlat,nlon)
    print("\tFinal Shape %s" % str(flxout_full.shape))
    
    # Check to see if N_mode exceeds nyrs
    if N_mode > nyr:
        print("Requested N_mode exists the maximum number of years, adjusting....")
        N_mode=nyr
    
    # Preallocate for EOF Analysis
    eofall    = np.zeros((N_mode,12,nens,nlat*nlon)) * np.nan
    pcall     = np.zeros((N_mode,12,nens,nyr)) * np.nan
    varexpall = np.zeros((N_mode,12,nens)) * np.nan
        
    # Loop for ensemble memmber
    for e in tqdm.tqdm(range(nens)):
        
        # Remove NaN Points
        flxens            = flxout[:,e,:,:].reshape(ntime,npts) #  Time x Space
        okdata,knan,okpts = proc.find_nan(flxens,0)
        _,npts_valid = okdata.shape
        
        # Repeat for full data
        flxens_full       = flxout_full[:,e,:,:].reshape(ntime,npts_full)
        okdataf,knanf,okptsf = proc.find_nan(flxens_full,0)
        _,npts_validf = okdataf.shape
        
        # Reshape to [yr x mon x pts]
        okdatar  = okdata.reshape(nyr,12,npts_valid)
        okdatarf = okdataf.reshape(nyr,12,npts_validf)
        
        # Calculate EOF by month
        for im in range(12):
            
            # Compute EOF
            datain          = okdatar[:,im,:].T # --> [space x time]
            eofs,pcs,varexp = proc.eof_simple(datain,N_mode,1)
            
            # Standardize PCs
            pcstd = pcs / pcs.std(0)[None,:]
            
            # Regress back to dataset
            datainf = okdatarf[:,im,:].T
            eof,b = proc.regress_2d(pcstd.T,datainf.T) # [time x pts]
            
            
            # Save the data
            eofall[:,im,e,okptsf] = eof.copy()
            pcall[:,im,e,:] = pcs.T.copy()
            varexpall[:,im,e] = varexp.copy()
    
    # Reshape the variable
    eofall = eofall.reshape(N_mode,12,nens,nlat,nlon) # (86, 12, 42, 96, 89)
    
    
    # Flip Signs
    if bbox_check is not None:
        print("Flipping boxes based on [bbox_check]")
        nmode_check = len(bbox_check)
        for N in tqdm.tqdm(range(nmode_check)):
            chkbox = bbox_check[N]
            for e in range(nens):
                for m in range(12):
                    
                    
                    sumflx = proc.sel_region(eofall[N,[m],e,:,:].transpose(2,1,0),flxa.lon.values,flxa.lat.values,chkbox,reg_avg=True)
                    #sumslp = proc.sel_region(eofslp[:,:,[m],N],lon,lat,chkbox,reg_avg=True)
                    
                    if sumflx > 0:
                        print("Flipping sign for NHFLX, mode %i month %i" % (N+1,m+1))
                        eofall[N,m,e,:,:]*=-1
                        pcall[N,m,e,:] *= -1
    else:
        print("Sign of EOF pattern will not be checked.")
    
    startyr   = daf.time.data[0]
    nyrs      = int(len(daf.time)/12)
    if concat_ens:
        tnew      = np.arange(0,int(ntime/12))
    else:
        tnew      = xr.cftime_range(start=startyr,periods=nyrs,freq="YS",calendar="noleap")

    # Make Dictionaries
    coordseof = dict(mode=np.arange(1,N_mode+1),mon=np.arange(1,13,1),ens=np.arange(1,nens+1,1),lat=flxa.lat,lon=flxa.lon)
    daeof     = xr.DataArray(eofall,coords=coordseof,dims=coordseof,name="eofs")

    coordspc  = dict(mode=np.arange(1,N_mode+1),mon=np.arange(1,13,1),ens=np.arange(1,nens+1,1),yr=tnew)
    dapcs     = xr.DataArray(pcall,coords=coordspc,dims=coordspc,name="pcs")

    coordsvar = dict(mode=np.arange(1,N_mode+1),mon=np.arange(1,13,1),ens=np.arange(1,nens+1,1))
    davarexp  = xr.DataArray(varexpall,coords=coordsvar,dims=coordsvar,name="varexp")
    
    ds_eof    = xr.merge([daeof,dapcs,davarexp])
    
    # # Return as DataArray
    # coord_eof    = dict(mode=np.arange(N_mode)+1,mon=np.arange(1,13,1),ens=np.arange(nens)+1,lat=daf.lat.data,lon=daf.lon.data)
    # da_eof       = xr.DataArray(eofall,dims=coord_eof,coords=coord_eof,name="eofs")
    
    # coord_pc     = dict(mode=np.arange(N_mode)+1,mon=np.arange(1,13,1),ens=np.arange(nens)+1,yr=np.arange(nyr)+1)
    # da_pcs       = xr.DataArray(pcall,dims=coord_pc,coords=coord_pc,name="pcs")
    
    # coord_varexp = dict(mode=np.arange(N_mode)+1,mon=np.arange(1,13,1),ens=np.arange(nens)+1)
    # da_varexpall = xr.DataArray(varexpall,dims=coord_varexp,coords=coord_varexp,name="varexp")
    
    # ds_out       = xr.merge([da_eof,da_pcs,da_varexpall])
    
    return ds_eof.squeeze()


#%% Calculate EOF Patterns

bboxeof    = [-80,20,0,65]
N_mode     = 40

spgbox     = [-60,20,40,80]
eapbox     = [-60,20,40,60] # Shift Box west for EAP
bbox_check = [spgbox,eapbox,]    

eof_out    = calc_monthly_eof(ds_fprime.Fprime.squeeze(),bboxeof,mask=dsmask_era5,bbox_check=bbox_check)


#%% Plottin Things

proj        = ccrs.PlateCarree()
mons3       = proc.get_monstr()
fsz_axis    = 14
fsz_ticks   = 12


bboxSPGNE   = [-40,-15,52,62]



#%% Sanity Check, Look at first 2 EOFs

N           = 1
imon        = 0

cints       = np.arange(-80,90,10)

for imon in range(12):
    
    fig,ax,bb   = viz.init_regplot()
    
    plotvar     = eof_out.eofs.isel(mode=N,mon=imon)
    
    
    pcm         = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                                vmin=cints[0],vmax=cints[-1],cmap='cmo.balance')
    
    
    cl          = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                             colors='k',linewidths=0.75,levels=cints)
    clbl        = ax.clabel(cl,fontsize=fsz_ticks)
    viz.add_fontborder(clbl,w=2.5)
    
    
    viz.plot_box(bboxSPGNE,ax=ax,color='purple',proj=proj,linewidth=1.5)
    
    
    
    # pcm         = ax.contourf(plotvar.lon,plotvar.lat,plotvar,transform=proj,
    #                           levels=cints,cmap='cmo.balance')
    
    cb          = viz.hcbar(pcm,ax=ax)
    cb.set_label("%s $F'$ [W/m2]\nMode %i (VarExp=%.2f" % (mons3[imon],N+1,eof_out.varexp.isel(mon=imon,mode=N)*100) + "%)",fontsize=fsz_axis)
    
    
    
    figname = "%sEOF_Fprime_Output_mon%02i_mode%02i.png" % (figpath,imon+1,N+1)
    plt.savefig(figname,dpi=150,bbox_inches='tight')


#%%


#%% Compute Monthly Stdev

fprime_monstd = ds_fprime.Fprime.groupby('time.month').std('time')

#%% Compute Monthly Amplitude of EOF

eof_monstd    = np.sqrt((eof_out.eofs**2).sum('mode'))



#%% Check Effect of Modes Inclusion

nmodes_sum      = np.array([1,2,5,10,15,20,30,40])
nsum            = len(nmodes_sum)
monvars_nmode   = []
for nn in range(nsum):
    modelim             = nmodes_sum[nn]
    eof_sel             = eof_out.eofs.sel(mode=slice(1,modelim))
    
    eof_sum_subset      = np.sqrt((eof_sel**2).sum('mode'))
    
    mvsel =  proc.area_avg_cosweight(proc.sel_region_xr(eof_sum_subset,bboxSPGNE))
    monvars_nmode.append(mvsel)

#%% Compare Monthly Average over SPGNE

invars     = [fprime_monstd,eof_monstd]

invars_avg = [proc.area_avg_cosweight(proc.sel_region_xr(iv,bboxSPGNE)) for iv in invars]
innames    = ["Fprime Std","EOF (All)"]
inls       = ["solid",'dashed']
inc        = ['k','pink']

#%%
fig,ax=viz.init_monplot(1,1)
for ii in range(2):
    ax.plot(mons3,invars_avg[ii],label=innames[ii],lw=2,ls=inls[ii],c=inc[ii])
    
    
for nn in range(nsum):
    ax.plot(mons3,monvars_nmode[nn],label="EOF (n=%i)" % nmodes_sum[nn])
    
ax.legend()



#%% Plot Variance Explained

fig,ax = plt.subplots(1,1,constrained_layout=True)
for im in range(12):
    plotvar = np.cumsum(eof_out.varexp.isel(mon=im))
    ax.plot(plotvar,label=mons3[im])
    
ax.legend()

ax.set_xlim([0,45])
ax.set_xticks(np.arange(0,46,2))
ax.set_ylim([0,1.05])
ax.set_yticks(np.arange(0,1.05,0.05))
ax.grid(True)
ax.set_ylabel("Cumulative % Variance Explained")
ax.set_xlabel("# Modes Included")


figname = "%sEOF_Fprime_ERA5_Cumulative_Varexp.png" % (figpath)
plt.savefig(figname,dpi=150,bbox_inches='tight')

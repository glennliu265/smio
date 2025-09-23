#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Formulate EOF-based forcing in ERA5

Notes
    - Currently written to run on Astraeus (local)
    - Copied [check_Fprime_EOF.py] and modified... on 2025.09.17


Created on Wed Sep 17 09:58:00 2025

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

#%% (copied from NHFLX_EOF_monthly, basically transformed it into a function)

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
    
    return ds_eof.squeeze()

#%% Load Fprime for ERA5

# Load Fprime
fpath           = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/reanalysis/proc/NATL_proc_obs/"
fnc             = "ERA5_Fprime_QNET_timeseries_QNETgmsstMON_nroll0_NAtl.nc"#"ERA5_Fprime_QNET_timeseries_QNETpilotObsAConly_nroll0_NAtl.nc"
ds_fprime       = xr.open_dataset(fpath + fnc).load()


# Load Ice Mask
dsmask_era5     = dl.load_mask(expname='ERA5').mask
dsmaskplot      = xr.where(np.isnan(dsmask_era5),0,1)

figpath  = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20250730/"
proc.makedir(figpath)

#%% Preprocessing

#% Mask and prepare for calculations
fprime = ds_fprime.Fprime.squeeze() * dsmask_era5

#%% Calculate EOF Patterns

# Indicate bounding box and number of modes
bboxeof    = [-80,20,0,65]
N_mode     = None # Set to None to automatically compute for # of years

# Indicate boxes to check sign over...
spgbox     = [-60,20,40,80]
eapbox     = [-60,20,40,60] # Shift Box west for EAP
bbox_check = [spgbox,eapbox,]    

# Perform EOF computation using function
eof_out    = calc_monthly_eof(ds_fprime.Fprime.squeeze(),bboxeof,N_mode=N_mode,mask=dsmask_era5,bbox_check=bbox_check)

#%% Perform filtering and correction
# Copied format from [reemergence/preprocessing/correct_eof_forcing_SSS]
# Wrote a function that takes output of "calc_monthly_eof" and performs filtering + correction


target_var = ds_fprime.Fprime
eof_thres  = 0.90

def compute_eof_correction(eof_out,target_var,eof_thres):
    """
    eof_out: Dataset from [calc_monthly_eof] containing:
        eofs  : EOF Patterns       [Mode x Mon x Lat x Lon] (see original script in <correct_eof_forcing_SSS> for a version that supports ensembles)
        varexp: Variance Explained [Mode x Mon]
        
        
    Note: Assumes eof_out and target_var have the same lat/lon!
    
    """
    
    # (1) Calculate Monthly Variance of the target variable
    target_monvar = target_var.groupby('time.month').std('time') # [mon x ens x lat x lon]
    target_monvar = target_monvar.rename({'month':'mon'})
    
    # (2) Transpose and prep variables for input
    eofvar_in = eof_out.eofs # [Mode x Mon x Lat x Lon]
    monvar_in = target_monvar.transpose('ens','mon','lat','lon').squeeze() # Remove Ensemble Dimension
    varexp_in = eof_out.varexp # [Mode x Month]
    
    # (3) Perform Filtering, and replace into dataset
    eofs_filtered,varexp_cumu,nmodes_needed,varexps_filt=proc.eof_filter(eofvar_in.data,varexp_in.data,eof_thres,axis=0,return_all=True)
    
    coords_eof    = dict(mode=eofvar_in.mode,mon=eofvar_in.mon,lat=eofvar_in.lat,lon=eofvar_in.lon)
    ds_eofs_filtered = xr.DataArray(eofs_filtered,
                                    coords=coords_eof,
                                    dims=coords_eof,
                                    name='eofs')#xr.full_like(eofvar_in,eofs_filtered)
    coords_varex  = dict(mode=eofvar_in.mode,mon=eofvar_in.mon)
    ds_varexp_cumu = xr.DataArray(varexp_cumu,
                                  coords=coords_varex,
                                  dims=coords_varex,
                                  name="cumulative_variance_explained"
                                  )
    ds_varexps_filt = xr.DataArray(varexps_filt,
                                  coords=coords_varex,
                                  dims=coords_varex,
                                  name="variance_explained_filtered"
                                  )
    ds_nmodes_needed = xr.DataArray(nmodes_needed,
                                    coords=dict(mon=eofvar_in.mon),
                                    dims =dict(mon=eofvar_in.mon),
                                    name = "number_modes_needed"
                                    )
    filt_out = xr.merge([ds_eofs_filtered,ds_varexp_cumu,ds_varexps_filt,ds_nmodes_needed])
    
    # (4) Compute Stdev of EOFs
    eofs_std = np.sqrt((ds_eofs_filtered**2).sum('mode'))
    
    # Compute pointwise correction
    correction_diff = monvar_in - eofs_std
    
    filt_out['correction_factor'] = correction_diff
    filt_out['original_stdev'] = monvar_in
    filt_out['threshold'] = eof_thres
    return filt_out

filt_out = compute_eof_correction(eof_out,target_var,eof_thres)
filt_out = filt_out.rename(dict(eofs="Fprime"))


#%% Save the full forcing file


fpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/forcing/"
fname = "%s%s_EOFFilt%03i_corrected.nc" % (fpath,fnc[:-3],eof_thres*100)

edict = proc.make_encoding_dict(filt_out)

filt_out.to_netcdf(fname,encoding=edict)


#outname  = proc.addstrtoext()
    
    
    
    
    
    










# #%% Plottin Things

# proj        = ccrs.PlateCarree()
# mons3       = proc.get_monstr()
# fsz_axis    = 14
# fsz_ticks   = 12


# bboxSPGNE   = [-40,-15,52,62]



# #%% Sanity Check, Look at first 2 EOFs

# N           = 1
# imon        = 0

# cints       = np.arange(-80,90,10)

# for imon in range(12):
    
#     fig,ax,bb   = viz.init_regplot()
#     plotvar     = eof_out.eofs.isel(mode=N,mon=imon)
    
#     pcm         = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
#                                 vmin=cints[0],vmax=cints[-1],cmap='cmo.balance')
    
    
#     cl          = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
#                              colors='k',linewidths=0.75,levels=cints)
#     clbl        = ax.clabel(cl,fontsize=fsz_ticks)
#     viz.add_fontborder(clbl,w=2.5)
    
    
#     viz.plot_box(bboxSPGNE,ax=ax,color='purple',proj=proj,linewidth=1.5)
    
    
    
#     # pcm         = ax.contourf(plotvar.lon,plotvar.lat,plotvar,transform=proj,
#     #                           levels=cints,cmap='cmo.balance')
    
#     cb          = viz.hcbar(pcm,ax=ax)
#     cb.set_label("%s $F'$ [W/m2]\nMode %i (VarExp=%.2f" % (mons3[imon],N+1,eof_out.varexp.isel(mon=imon,mode=N)*100) + "%)",fontsize=fsz_axis)
    
    
    
#     figname = "%sEOF_Fprime_Output_mon%02i_mode%02i.png" % (figpath,imon+1,N+1)
#     plt.savefig(figname,dpi=150,bbox_inches='tight')

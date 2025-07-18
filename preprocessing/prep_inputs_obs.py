#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Preparing Inputs for stochastic model for observations, SST

Currently written to run on Astraeus

Created on Mon Mar 17 11:56:39 2025

@author: gliu

"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import time
import sys
import cartopy.crs as ccrs
import glob
import matplotlib as mpl
import pandas as pd
import scipy as sp

#%% Import modules
stormtrack = 0
if stormtrack:
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
    
    # Path to the processed dataset (qnet and ts fields, full, time x lat x lon)
    #datpath =  "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/hfdamping_RCP85/01_PREPROC/"
    datpath =  "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/output/anom/"
    figpath =  "/home/glliu/02_Figures/01_WeeklyMeetings/20240621/"
    
else:
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")

    # Path to the processed dataset (qnet and ts fields, full, time x lat x lon)
    datpath =  "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/reanalysis/proc/"
    figpath =  "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/02_Figures/20220511/"
from amv import proc,viz
import scm


import amv.proc as hf # Update hf with actual hfutils script, most relevant functions
import amv.loaders as dl
#%% User Edits

# Plot Settings
mpl.rcParams['font.family'] = 'Avenir'
proj = ccrs.PlateCarree()
bbplot = [-80, 0, 35, 75]
mons3 = proc.get_monstr()

figpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20250501/"
proc.makedir(figpath)

#%% Load Ice Edge

icemask = dl.load_mask("ERA5").mask
plotmask = xr.where(np.isnan(icemask),0,icemask)

# =========================
#%% (1) Load ERA5 HFF
# =========================

flxname         = "qnet"
dpath           = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/"
if flxname == "qnet":
    dof   = (2024-1979 + 1 - 2 - 1) * 3
    vname           = "qnet_damping"
    ncname_era5     = "ERA5_qnet_hfdamping_NAtl_1979to2024_ensorem1_detrend1.nc"
else:
    dof   = (2021-1979 + 1 - 2 - 1) * 3
    vname           = "thflx_damping"
    ncname_era5     = "ERA5_thflx_hfdamping_NAtl_1979to2021_ensorem1_detrend1.nc"
ds_era5         = xr.open_dataset(dpath + ncname_era5)

dt              = 3600*24  # *30 #Effective Processing Period of 1 day

lone            = ds_era5.lon
late            = ds_era5.lat

if flxname == "thflx": # Old version needed to be converted to months
    damping_era5 = ds_era5.thflx_damping / dt * -1
else:
    damping_era5 = ds_era5[vname] * -1

# Copy ERA5
hff_era5 = damping_era5.copy()


#%% Perform significance testing

# Set Significance calculation settings
"""
Some Options for Signifiance Testing

pilot      : Same as was used for the SSS Paper
noPositive : Just set positive HFF to zero
p10        : Use p = 0.10
p20        : Use p = 0.20
AConly     : Test Autocorrelation Only

"""
signame = "p10"#"noPositive"#"pilot" #"noPositive" # 
print("Significance Testing Option is: %s" % (signame))

hff   = damping_era5.copy()
rsst  = ds_era5.sst_autocorr.copy()
rflx  = ds_era5.sst_flx_crosscorr.copy()
setdict = {  # Taken from hfcalc_params
    'ensorem': 1,      # 1=enso removed, 0=not removed
    'ensolag': 1,      # Lag Applied toENSO and Variable before removal
    'monwin': 3,      # Size of month window for HFF calculations
    'detrend': 1,      # Whether or not variable was detrended
    'tails': 2,      # tails for t-test

    'p': 0.05,   # p-value for significance testing
    'sellags': [0,],   # Lags included (indices, so 0=lag1)
    'lagstr': "lag1",  # Name of lag based on sellags
    # Significance test option: 1 (No Mask); 2 (SST autocorr); 3 (SST-FLX crosscorr); 4 (Both), 5 (Replace with SLAB values),6, zero out negative values
    'method': 4
}


if signame == "pilot":
    setdict['method'] = 4 # Apply Significance Testing to Both
elif signame == "noPositive":
    setdict['method'] = 6 # Apply Significance Testing to Both
elif signame == "p10":
    setdict['p'] = 0.10
elif signame == "p20":
    setdict['p'] = 0.20
elif signame == "AConly":
    setdict['method'] = 2
    
# dof was set above

# Compute and Apply Mask
st = time.time()
dampingmasked, freq_success, sigmask = scm.prep_HF(hff, rsst, rflx,
                                                   setdict['p'], setdict['tails'], dof, setdict['method'],
                                                   returnall=True)  # expects, [month x lag x lat x lon], should generalized with ensemble dimension?
print("Completed significance testing in %.2fs" % (time.time()-st))



dampingout = dampingmasked[:,setdict['sellags'],:,:].squeeze()
dampingout = xr.where(np.isnan(dampingout),0.,dampingout)
#dampingout[np.isnan(dampingout)] = 0

#%% Sanity Check for the damping...

imon    = 1
ilag    = 0

for imon in range(12):
    plotvar = hff.isel(lag=ilag,month=imon) #dampingout.isel(month=imon)#
    #isneg    = xr.where(plotvar<0,1,0)
    plotmask = sigmask[imon,ilag]
    
    if np.any(np.isnan(plotmask)):
        plotmask = xr.where(np.isnan(plotmask),0,1)
    else:
        plotmask = plotmask#xr.where(plotmask==0,0,1)
    
    bbsel   = [-80,0,50,65]
    
    fsz_title = 24
    
    
    # Get Count of points in neative region
    bbsim   = [-40,-15,52,62]
    isneg_reg,_,_ = proc.sel_region(plotmask.T,plotvar.lon.data,plotvar.lat.data,bbsim)
    isneg_reg     = np.logical_not(isneg_reg,dtype=bool)
    total_neg = isneg_reg.sum((0,1))
    total_pts = np.prod(np.array(np.shape(isneg_reg)))
    ptcount   = "%s/%s pts (%.2f" % (total_neg,total_pts,total_neg/total_pts*100) + "%) are set to 0"
    
    
    fig,axs,_ = viz.init_orthomap(2,1,bboxplot=bbsel,figsize=(12,8))
    
    # Plot Before with significance dots
    ax = axs[0]
    ax = viz.add_coast_grid(ax,bbox=bbsel,proj=proj)
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                        transform=proj,
                        cmap='cmo.balance',vmin=-35,vmax=35)
    viz.plot_mask(plotvar.lon,plotvar.lat,plotmask.T,geoaxes=True,proj=proj,ax=ax,color='gray',markersize=0.2)
    ax.set_title('Before Mask (Lag %i, Month %s)\n%s' % (ilag+1,mons3[imon],ptcount),fontsize=fsz_title)
    
    # PLot After Masking
    ax = axs[1]
    plotvar = dampingout.isel(month=imon)
    ax = viz.add_coast_grid(ax,bbox=bbsel,proj=proj)
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                        transform=proj,
                        cmap='cmo.balance',vmin=-35,vmax=35)
    viz.plot_mask(plotvar.lon,plotvar.lat,plotmask.T,geoaxes=True,proj=proj,ax=ax,color='gray',markersize=0.2)
    
    
    # Set Title and Colorbar
    ax.set_title('After Mask',fontsize=fsz_title)
    cb = viz.hcbar(pcm,ax=axs.flatten())
    
    
    for ax in axs:
        
        # Plot Bounding Box for analysis
        viz.plot_box(bbsim,ax=ax,color='limegreen',linewidth=2.5,proj=proj)
    
    figname = "%sHFF_Check_ERA5_%s_%s_mon%02i_lag%i.png" % (figpath,flxname,signame,imon+1,ilag+1)
    plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Check Points in SPGNE Region

ilag      = 0
plotvar   = hff.isel(lag=ilag).min('month') #dampingout.isel(month=imon)#

isneg     = xr.where(plotvar<0,1,0)

bbplot2   = [-50,0,50,65]

fig,ax,_  = viz.init_orthomap(1,1,bboxplot=bbplot2,figsize=(12,8),centlon=-25)

# Plot the Heat Flux Damping
pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                    transform=proj,
                    cmap='cmo.balance',vmin=-35,vmax=35,zorder=-2)

# Plot Negative Points
viz.plot_mask(plotvar.lon,plotvar.lat,isneg.T,reverse=True,
              geoaxes=True,proj=proj,ax=ax,color='k',markersize=0.2)


plotvar   = plotmask
ax.contour(plotvar.lon,plotvar.lat,plotvar,levels=[0,1],transform=proj,colors='cyan',zorder=-1)

ax        = viz.add_coast_grid(ax,bbox=bbplot2,proj=proj,fill_color="k")
viz.plot_box(bbsim,ax=ax,color='limegreen',linewidth=2.5,proj=proj)




# Print some information
isneg_reg = proc.sel_region_xr(isneg,bbsim)
total_neg = isneg_reg.data.sum((0,1))
total_pts = np.prod(np.array(np.shape(isneg_reg)))
ptcount   = "%s/%s pts (%.2f" % (total_neg,total_pts,total_neg/total_pts*100) + "%) are negative"
ax.set_title("ERA5 Minimum Heat Flux Feedback, Lag %02i\n%s" % (ilag+1,ptcount),fontsize=fsz_title)

figname = "%sHFF_Check_SPGNE_ERA5_%s_%s_monMIN_lag%i.png" % (figpath,flxname,signame,ilag+1)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Put into DataArray

# Get Dimensions
mons        = np.arange(1,13,1)
lon         = ds_era5.lon.values
lat         = ds_era5.lat.values

dims        = dict(mon=mons,lat=lat,lon=lon,)
da          = xr.DataArray(dampingout,name='damping',
                    dims=dims,coords=dims)

edict = proc.make_encoding_dict(da)
outpath_damping = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/damping/"
outname         = outpath_damping + "ERA5_%s_damping_%s.nc" % (flxname,signame)
print("Saving to %s" % outname)
da.to_netcdf(outname,encoding=edict)


#%% Plot the HFF sigtest results

imon        = 9
ilag        = 0
fig, ax, _  = viz.init_orthomap(1, 1, bbplot, figsize=(14, 6))
ax          = viz.add_coast_grid(ax, bbplot, fill_color='lightgray',
                        proj=proj, line_color="dimgray")

pcm = ax.pcolormesh(lon,lat,dampingmasked[imon,ilag,:,:],
                   vmin=-35, vmax= 35,transform=proj,cmap='cmo.balance')

viz.hcbar(pcm)


# ================================
#%% (2) Load the MLD and get kprev
# ================================



dpath_proc = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/reanalysis/proc/NATL_proc_obs/"
ncname     = dpath_proc + "MIMOC_RegridERA5_mld_NAtl_Climatology.nc"
ds_mld     = xr.open_dataset(ncname).load()

ds_mld     = ds_mld.rename(dict(mld='h'))
ds_mld     = ds_mld.rename(dict(month='mon'))

outpath_h  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/"
outname    = outpath_h + "MIMOC_regridERA5_h_pilot.nc"
edict      = proc.make_encoding_dict(ds_mld)
ds_mld.to_netcdf(outname,encoding=edict)

#%% Compute Kprev

# Compute kprev for ens-mean mixed layer depth cycle
infunc     = lambda x: scm.find_kprev(x,debug=False,returnh=False)
st         = time.time()

kprevall = xr.apply_ufunc(
    infunc, # Pass the function
    ds_mld, # The inputs in order that is expected
    input_core_dims =[['mon'],], # Which dimensions to operate over for each argument... 
    output_core_dims=[['mon'],], # Output Dimension
    vectorize=True, # True to loop over non-core dims
    )
print("Completed kprev calc in %.2fs" % (time.time()-st))

kprevall   = kprevall.transpose('mon','lat','lon').rename(dict(h='kprev'))
edict      = proc.make_encoding_dict(kprevall)
outname    = outpath_h + "MIMOC_regridERA5_kprev_pilot.nc"
kprevall.to_netcdf(outname,encoding=edict)

#%%

nlat = len(lat)
nlon = len(lon)

for a in range(nlat):
    for o in range(nlon):
        dspt = ds_mld.isel(lat=a,lon=o)
        out = scm.find_kprev(dspt.h.data,debug=False,returnh=False)
        
# ================================
#%% (3) Work on Land/Ice mask
# ================================ 

# Load Sea Ice Masks
dpath_ice = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/reanalysis/proc/NATL_proc_obs/"
nc_masks = dpath_ice + "OISST_ice_masks_1981_2020.nc"
ds_masks = xr.open_dataset(nc_masks).load()

# Make Land Mask
land_nc     = dpath_ice + "ERA5_land_mask_1980_and_2020_NATL.nc"
ds_land     = xr.open_dataset(land_nc).load()

ds_land   = xr.where(np.isnan(ds_land.land_mask),1,np.nan,)


#ds_land  = xr.where(np.isnan(ds_mld.h.mean('mon')),np.nan,1)

# Adjust ice mask
ds_ice = ds_masks.mask_mon
ds_ice = xr.where(ds_ice,np.nan,1)

# Make the mask
#mask   = ds_land

#%% ok, I guess we have to regrid the oisst mask
# let's just do this conservatively
import tqdm

#ds_ice = ds_ice.data
ds_ice_arr = ds_ice.data

nlat = len(ds_ice.lat)
nlon = len(ds_ice.lon)

for a in tqdm.tqdm(range(nlat)):
    for o in range(nlon):
        
        dspt = ds_ice.isel(lon=o,lat=a)
        
        lonf = dspt.lon.data
        latf = dspt.lat.data
        
        ds_land_pt = proc.selpt_ds(ds_land,lonf,latf)
        
        if np.isnan(ds_land_pt.data):
            ds_ice[a,o] = 1
            
#%% Save the plot

# Reverse the mask
ds_ice_out = xr.where(ds_ice == 1,np.nan,1)
 # Ok I'm not sure whats going on but I will work twith this later
 
 
 
#%% (4) Do calculations for the forcing


vname   = "QNET"
dampstr = "QNETpilotObsAConly"
ncpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/reanalysis/proc/NATL_proc_obs/"
ncname = "ERA5_Fprime_%s_timeseries_%s_nroll0_NAtl.nc"  % (vname,dampstr) #"ERA5_Fprime_%s_timeseries_%spilotObs_nroll0_NAtl.nc" % (vname,vname)
ds     = xr.open_dataset(ncpath+ncname).load()

# COmpute the monthly standard deviation
dsmon = ds.groupby('time.month').std('time')
dsmon = dsmon.rename(dict(month='mon'))
 
#outname = ""
#ds_ice_arr.plot()
outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/forcing/"
edict   = proc.make_encoding_dict(dsmon)
outname = outpath + "ERA5_Fprime_%s_std_pilot.nc" % dampstr #vname
dsmon.to_netcdf(outname,encoding=edict)

#%% Repeat above but using qnet damping


#%% (5) ORAS5 Mixed-Layer Depths
# Works with regridded climatology from regrid_subsurface_damping_era5

ncname  = "ORAS5_CDS_mld_NAtl_1979_2024_scycle_regridERA5.nc"
ncpath  = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
ds      = xr.open_dataset(ncpath+ncname).load()

mldpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/"
ncref   = mldpath + "MIMOC_regridERA5_h_pilot.nc"
dsref   = xr.open_dataset(ncref).load()


ds      = ds.rename(dict(mld='h',month='mon'))
diff    = ds.h - dsref.h

outname = mldpath + "ORAS5_CDS_regridERA5_h.nc"
edict   = proc.make_encoding_dict(ds)
ds.to_netcdf(outname,encoding=edict)

#%%



#%%


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate Subsurface Damping from EN4 Profiles
    - Use regridded mimoc from [regrid_mimoc_EN4.py]
    - Based on [preproc_detrainment_data] (ACF calculation)
    - Then copied from [calc_detrainment_correlation_pointwise] for corr-based lbd_d

Update [2025.07.01], support ORAS5 Calculations
    - Used cropped region from [crop_oras5_natl]
    - Used regridded MLD to ERA5 resolution (MIMOC)

Update [2025.07.09], Add ORAS5 2019-2025 with toggle

Created on Tue Jun 24 13:10:54 2025

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


#%% Helper Functions

# Functions ---
def calc_acf_ens(ts_monyr,lags):
    # tsens is the anomalized values [yr x mon x z]
    acfs_mon = []
    for im in range(12):
        basemonth   = im+1
        varin       = ts_monyr[:,:,:]  # Month x Year x Npts
        out         = proc.calc_lagcovar_nd(varin, varin, lags, basemonth, 1)
        
        acfs_mon.append(out)
        # <End Month Loop>
    return np.array(acfs_mon) # [Mon Lag Depth]

def fit_exp_ens(acfs_mon,lagmax):
    # acfs_mon [month x lag x depth] : Monthly lagged ACFs
    
    _,nlags,nz = acfs_mon.shape
    tau_est = np.zeros((12, nz))
    acf_est = np.zeros((12, nlags, nz))
    
    for im in range(12):
        for zz in range(nz):
            acf_in = acfs_mon[im, :, zz] # Select Depth and Month
            
            outdict             = proc.expfit(acf_in, lags, lagmax=lagmax)
            tau_est[im, zz]     = outdict['tau_inv'].copy()
            acf_est[im, :, zz]  = outdict['acf_fit'].copy()
    return tau_est,acf_est

#%% Load Datasets (depending on dataset)

"""
Possible Dataset Names

EN4
- MIMOC MLDs
- EN4...

ORAS5
- MIMOC MLD
- ORAS opa0 1979-2018

ORAS5_avg
- MIMOC MLDs
- ORAS5 average of opa0 to opa4, 1979-2024 (operational and consolidated)

ORAS5_avg_mld003
- ORAS5 MLD 0.03 threshold
- Same votemper as ORAS5_avg


"""


dataset_name = "ORAS5_avg_mld003" # EN4 # Indicate which Dataset


mldname = "MIMOC"
st = time.time()
if dataset_name == "EN4":
    # Load EN4 Profiles
    en4path     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/EN4/proc/"
    en4nc       = "EN4_3D_TEMP_NAtl_1900_2021.nc"
    ds_temp     = xr.open_dataset(en4path + en4nc).temperature.load()
    
    
    # Load and query MIMOC (regridded by regrid_mimoc_EN4.py)
    method          = "bilinear"
    mimocpath       = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
    mimocnc         = "%sMIMOC_RegridEN4_mld_%s_Global_Climatology.nc" % (mimocpath,method)
    ds_mld = xr.open_dataset(mimocnc).load()
    
    # Set Cropping Time Period/Region
    tstart = 1900
    tend   = 2021
    bbcalc = [-80,0,0,65]
    
elif dataset_name == "ORAS5":
    
    # Load ORAS5 Profiles
    oraspath = "/Users/gliu/Globus_File_Transfer/Reanalysis/ORAS5/proc/"
    orasnc   = "ORAS5_opa0_TEMP_NAtl_1979_2018.nc"
    ds_temp  = xr.open_dataset(oraspath + orasnc).TEMP.load()
    
    # Also load 2019-2024
    orasnc2  = "ORAS5_CDS_TEMP_NAtl_2019_2024.nc"
    ds_temp2 = xr.open_dataset(oraspath + orasnc2).TEMP.load()
    ds_temp  = xr.concat([ds_temp,ds_temp2],dim='time')
    
    
    # Load MIMOC (ERA5 Resolution)
    mimocpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
    mimocnc   = "MIMOC_RegridERA5_mld_NAtl_Climatology.nc"
    ds_mld    = xr.open_dataset(mimocpath + mimocnc).load()
    
    # Set Cropping Time Period
    tstart = 1979
    tend   = 2024
    bbcalc = [-40,-15,52,62]
    
    ds_temp = ds_temp.rename(dict(z_t='depth'))
    
elif dataset_name == "ORAS5_avg" or "ORAS5_avg_mld03":
    
    # Load ORAS5 Profiles
    oraspath = "/Users/gliu/Globus_File_Transfer/Reanalysis/ORAS5/proc/"
    orasnc   = "ORAS5_opaAVG_TEMP_NAtl_1979_2024.nc"
    ds_temp  = xr.open_dataset(oraspath + orasnc).TEMP.load()
    
    if "mld003" in dataset_name:
        print("Loading MLDs from ORAS (CDS, Ens 01")
        mldpath  = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
        mldnc    = "ORAS5_CDS_mld_NAtl_1979_2024_scycle.nc"
        ds_mld   = xr.open_dataset(mldpath + mldnc).load()
        mldname  = "ORAS5mld003"
        
    else:
        print("Loading MLDs from MIMOC")
        # Load MIMOC (ERA5 Resolution)
        mimocpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
        mimocnc   = "MIMOC_RegridERA5_mld_NAtl_Climatology.nc"
        ds_mld    = xr.open_dataset(mimocpath + mimocnc).load()
    
    # Set Cropping Time Period
    tstart = 1979
    tend   = 2024
    bbcalc = [-40,-15,52,62]
    
    ds_temp = ds_temp.rename(dict(z_t='depth'))
    
else:
    
    print("Currently only supports the following datasets:\n")
    print("\tEN4")
    print("\tORAS5")
    print("\tORAS5_avg")
    
    
proc.printtime(st,print_str="Loaded")

#%% Load ERA5 GMSST for detrending

# Load GMSST For ERA5 Detrending
detrend_obs_regression = True
dpath_gmsst     = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
nc_gmsst        = "ERA5_GMSST_1979_2024.nc"
ds_gmsst        = xr.open_dataset(dpath_gmsst + nc_gmsst).GMSST_MeanIce.load()


#%% Detrainment calculation Options



# Set Time Period Strings
tstr    = "%04ito%04i" % (tstart,tend)

lags    = np.arange(0,37,1)
nlags   = len(lags)

lagmax  = 3

# detrend = 'ensmean'
detrend = 'gmsst' # Set to None to do a linear detrend


#%% First, restrict MLD and calculate kprev

# Slice to region (MLD)
mld2d = False
try:
    ds_mld = proc.sel_region_xr(ds_mld.mld,bbcalc) # Mon x Lat x Lon
except:
    mld2d = True
    print("Lon not found, trying curvilinear region selection...")
    ds_mld = proc.sel_region_xr_cv(ds_mld,bbcalc)
    

# Slice to time (TEMP)
ds_temp = ds_temp.sel(time=slice("%04i-01-01" % tstart,"%04i-12-31" % tend)) # [time x depth x lat x lon]

# Slice to region (TEMP)
if "TLONG" in ds_temp.coords:
    ds_temp = proc.sel_region_xr_cv(ds_temp,bbcalc,debug=False)
    
    

# Slice to max MLD Depth (take first level that exceeds deepest)
if mld2d:
    max_mld             = np.ceil(np.nanmax(ds_mld.mld.data.flatten()))
else:
    max_mld             = np.ceil(np.nanmax(ds_mld.data.flatten()))
en4depths           = ds_temp.depth
depthdiff           = en4depths.data - max_mld
idx_deeper          = np.where(depthdiff>0)[0][0] # Find first index deeper than max climatological MLD
print("Maximum Mixed-Layer Depth is %.2f meters in MIMOC" % (max_mld))
print("This is between %i (%.2f) and %i (%.2f) in %s" % (idx_deeper-1,en4depths.isel(depth=idx_deeper-1),
                                                          idx_deeper,en4depths.isel(depth=idx_deeper),
                                                          dataset_name,
                                                          ))
ds_temp = ds_temp.isel(depth=slice(0,idx_deeper+1))



#%% Reprocess and detrend (took 82 sec for NATL ORAS5, 2.36 sec for SPGNE)

dtvar_anom = proc.xrdeseason(ds_temp)
if detrend == "gmsst":
    dtout      = proc.detrend_by_regression(dtvar_anom,ds_gmsst)
    dtvar_anom = dtout['TEMP']
else:
    dtvar_anom = proc.xrdetrend(dtvar_anom)


#%% Retrieve Dimensions and reshape to year x mon



if "TLONG" in ds_temp.coords: # For Processed ORAS5 data, look for TLONG

    lon         = dtvar_anom.TLONG.data
    lat         = dtvar_anom.TLAT.data
    lon360_flag = True
    coords2D    = True
    
else: # For EN4, 1-D lat/lon
    
    lon         = dtvar_anom.lon.data
    lat         = dtvar_anom.lat.data
    lon360_flag = False
    coords2D   = False
    
    
    
z                  = dtvar_anom.depth.data
ntime,nz,nlat,nlon = dtvar_anom.shape
nyr                = int(ntime/12)
dtvar_yrmon        = dtvar_anom.data.reshape((nyr,12,nz,nlat,nlon))

# Looping will be done w.r.t. MLD coordinates
if mld2d:
    lon_mld = ds_mld.TLONG
    lat_mld = ds_mld.TLAT
else:
    lon_mld = ds_mld.lon
    lat_mld = ds_mld.lat


#%% Calculate the ACF and detrainment (timescale-based)

lbd_d_all   = np.zeros((12,nlat,nlon)) * np.nan          # Estimated Detrainment Damping
tau_est_all = np.zeros((12,nz,nlat,nlon))  * np.nan      # Fitted Timescales
acf_est_all = np.zeros((12,nlags,nz,nlat,nlon)) * np.nan # Fitted ACF
acf_mon_all = np.zeros((12,nlags,nz,nlat,nlon)) * np.nan # Actual ACF


for o in tqdm.tqdm(range(nlon)):
    for a in range(nlat):
        
        # Retrieve variable at point
        varpt = dtvar_yrmon[:,:,:,a,o].copy() # Yr x Mon x Depth
        if np.all(np.isnan(varpt)):
            continue # Skip the Point because it is on land
        
        # Debugging ====
        #plt.pcolormesh(np.arange(nyr),dtvar_anom.depth,varpt[:,0,:].T,vmin=-1.5,vmax=1.5,cmap='RdBu_r'),plt.colorbar()
        
        depthsum = np.sum(varpt,(0,1))
        idnan_z  = np.where(np.isnan(depthsum))[0]
        varpt[:,:,idnan_z] = 0 # Set to Zeros
        
        
        # Retrieve Mixed layer depth cycle at a point
        if coords2D:
            lonf = lon[a,o]
            latf = lat[a,o]
        else:
            lonf = lon[o]
            latf = lat[a]
        if lon360_flag and lonf > 180:
            print("Converting Longitude to degrees West")
            lonf = lonf - 360
        
        if mld2d:
            hpt  = proc.find_tlatlon(ds_mld,lonf,latf,verbose=False).mld
        else:
            hpt  = ds_mld.sel(lon=lonf,lat=latf,method='nearest').values#[month]
        if np.any(np.isnan(hpt)):
            continue
        
        # Section here is taken from calc_detrainemtn_damping_pt -------------
        # Input Data
        ts_monyr     = varpt.transpose(1,0,2)        # Anomalies [mon x yr x otherpts (z)]
        hclim        = hpt                           # MLD Cycle [mon]
        
        # (1) Estimate ACF
        acfs_mon = calc_acf_ens(ts_monyr,lags) # [mon x lag x depth]
        acfs_mon[:,:,idnan_z] = 0 # To Avoid throwing an error
        
        # (2) Fit Exponential Func
        tau_est,acf_est = fit_exp_ens(acfs_mon,lagmax) # [mon x depth], [mon x lags x depth]
        
        # (3) Compute Detraiment dmaping
        kprev,_ = scm.find_kprev(hclim)
        lbd_d   = scm.calc_tau_detrain(hclim,kprev,z,tau_est,debug=False)
        
        # Correct zeros back to nans
        acfs_mon[:,:,idnan_z] = np.nan
        tau_est[:,idnan_z] = np.nan
        acf_est[:,:,idnan_z] = np.nan
        
        # Save Output
        lbd_d_all[:,a,o]       = lbd_d.copy()
        tau_est_all[:,:,a,o]   = tau_est.copy()
        acf_est_all[:,:,:,a,o] = acf_est.copy()
        acf_mon_all[:,:,:,a,o] = acfs_mon.copy()
        
#%% Check the output


mons       = np.arange(1,13,1)
nlat       = np.arange(0,nlat)
nlon       = np.arange(0,nlon)

if coords2D:
    coordsxy = dict(nlat=nlat,nlon=nlon)
    
    da_TLONG = xr.DataArray(ds_temp.TLONG,coords=coordsxy,dims=coordsxy,name="TLONG")
    da_TLAT  = xr.DataArray(ds_temp.TLAT,coords=coordsxy,dims=coordsxy,name="TLAT")
    
    # Make data arrays
    lcoords    = dict(mon=mons,lat=nlat,lon=nlon)
    da_lbdd    = xr.DataArray(lbd_d_all,coords=lcoords,dims=lcoords,name="lbd_d")
    
    taucoords  = dict(mon=mons,z_t=z,lat=nlat,lon=nlon)
    da_tau     = xr.DataArray(tau_est_all,coords=taucoords,dims=taucoords,name="tau")
    
    acfcoords  = dict(mon=mons,lag=lags,z_t=z,lat=nlat,lon=nlon)
    da_acf_est = xr.DataArray(acf_est_all,coords=acfcoords,dims=acfcoords,name="acf_est")
    da_acf_mon = xr.DataArray(acf_mon_all,coords=acfcoords,dims=acfcoords,name="acf_mon")
    
    ds_out     = xr.merge([da_lbdd,da_tau,da_acf_est,da_acf_mon,da_TLONG,da_TLAT])
    
else: # 1-D coorindates Case (EN4)
    
    # Make data arrays
    lcoords    = dict(mon=mons,lat=lat,lon=lon)
    da_lbdd    = xr.DataArray(lbd_d_all,coords=lcoords,dims=lcoords,name="lbd_d")

    taucoords  = dict(mon=mons,z_t=z,lat=lat,lon=lon)
    da_tau     = xr.DataArray(tau_est_all,coords=taucoords,dims=taucoords,name="tau")

    acfcoords  = dict(mon=mons,lag=lags,z_t=z,lat=lat,lon=lon)
    da_acf_est = xr.DataArray(acf_est_all,coords=acfcoords,dims=acfcoords,name="acf_est")
    da_acf_mon = xr.DataArray(acf_mon_all,coords=acfcoords,dims=acfcoords,name="acf_mon")
    
    ds_out     = xr.merge([da_lbdd,da_tau,da_acf_est,da_acf_mon,])


edict      = proc.make_encoding_dict(ds_out)
savename   = "%s%s_%s_lbd_d_params_%s_detrend%s_lagmax%i_%s.nc" % (mimocpath,dataset_name,mldname,
                                                                       'TEMP','linear',lagmax,tstr)
ds_out.to_netcdf(savename,encoding=edict)

#%% Use the ACFs to compute the detrainment, correlation based
# Copied from [calc_detrainment_correlation_pointwise]

# Correlation Options ---
detrainceil = False # True to use ceil rather than floor of the detrainment month
interpcorr  = True  # True to interpolate between ceil and floor of detrianment month
dtdepth     = True  # Set to true to retrieve temperatures at the detrainment depth
imshift     = 1     # Stop [imshift] months before the entraining month



#%% Part 1) Compute Detrainment Depths

# Compute kprev for ens-mean mixed layer depth cycle
infunc = lambda x: scm.find_kprev(x,debug=False,returnh=False)
st     = time.time()
kprevall = xr.apply_ufunc(
    infunc, # Pass the function
    ds_mld, # The inputs in order that is expected
    input_core_dims =[['month'],], # Which dimensions to operate over for each argument... 
    output_core_dims=[['month'],], # Output Dimension
    vectorize=True, # True to loop over non-core dims
    )
print("Completed kprev calc in %.2fs" % (time.time()-st))

# Compute detrainment depths for ens-mean mld
st = time.time()
hdetrainall = xr.apply_ufunc(
    scm.get_detrain_depth, # Pass the function
    kprevall, # The inputs in order that is expected
    ds_mld,
    input_core_dims=[['month'],['month']], # 
    output_core_dims=[['month'],],#['ens'],['lat'],['lon']],
    vectorize=True,
    )
print("Completed hdetrain calc in %.2fs" % (time.time()-st))
        
#%% Part 2) Compute Pointwise Correlation

lbd_in      = ds_out.lbd_d
corr_out    = np.zeros(lbd_in.shape)*np.nan
_,nlat,nlon = corr_out.shape

z_t         = ds_temp.depth

acf_ens     = ds_out.acf_mon

# Note, edited so that dtdepth = True 
debug       = True

# Loop for each point
for a in tqdm.tqdm(range(nlat)):
    for o in range(nlon):
        
        # Select MLD at point
        # Retrieve Mixed layer depth cycle at a point
        if coords2D:
            lonf = lon[a,o]
            latf = lat[a,o]
        else:
            lonf = lon[o]
            latf = lat[a]
        if lon360_flag and lonf > 180:
            print("Converting Longitude to degrees West")
            lonf = lonf - 360
        
        if mld2d:
            hpt  = proc.find_tlatlon(ds_mld,lonf,latf,verbose=False).mld
        else:
            hpt  = ds_mld.sel(lon=lonf,lat=latf,method='nearest').values#[month]
        if np.all(np.isnan(hpt)): # Skip for land point
            continue
        
        immax = hpt.argmax().values.item()
        immin = hpt.argmin().values.item()
        
        # Compute kprev for the ensemble member
        if mld2d:
            kprev   = proc.find_tlatlon(kprevall,lonf,latf).mld.data
        else:
            kprev   = kprevall.sel(lon=lonf,lat=latf,method='nearest').values
        
        for im in range(12):
            detrain_mon = kprev[im]
            if detrain_mon == 0.:
                continue # Skip when there is no entrainment
                
            # Get indices for autocorrelation
            dtid_floor = int(np.floor(detrain_mon)) - 1 
            dtid_ceil  = int(np.ceil(detrain_mon)) - 1
            entrid     = (im - imshift)%12
            
            # Check for cases (on first detrain month) where dt_ceil > entr_id
            if (dtid_ceil) >= entrid:
                
                if (dtid_floor) >= entrid: # Entrain Month # lower than detraining months, so it is one year later. Add 12
                    entrid = entrid + 12
                    #if debug:
                        # print("dtfloor: %i, dtceil: %i,  entrid %i" % (dtid_floor,dtid_ceil,entrid))
                        # print("WARNING: Floor of detrain month >= the entraining month. Check o=%i,a=%i, mon %02i, ens %02i for %s" % (o,a,im+1,e+1,vname))
                        
                    
                else: # first Entraining month, set to same value...
                    # if debug:
                    #     print("dtfloor: %i, dtceil: %i,  entrid %i" % (dtid_floor,dtid_ceil,entrid))
                    #     print("Settin ceil to floor")
                    dtid_ceil = dtid_floor # Just double count the same value
                
                #break
                #quit()
            if debug:
                print("Detaining Months [%i,%f,%i], Entrain Month [%i]" % (dtid_floor+1,detrain_mon,dtid_ceil+1,entrid+1))
            
            # First, get the depths
            if dtdepth: # Just retrieve at the detrainment depth
                if mld2d:
                    h_detrain  = proc.find_tlatlon(kprevall,lonf,latf).mld.isel(month=im).values.item(0)
                else:
                    h_detrain  = hdetrainall.sel(lon=lonf,lat=latf,method='nearest').isel(month=im).values.item(0)
                zz_floor   = proc.get_nearest(h_detrain,z_t.values)
                zz_ceil    = zz_floor # same depth for detrain
                
            else: # Retrieve ACF at each corresponding depth before/after the detrainment time
                
                h_floor    = hpt.isel(mon=dtid_floor).values.item()
                h_ceil     = hpt.isel(mon=dtid_ceil).values.item()
                zz_floor   = proc.get_nearest(h_floor,z_t.values)
                zz_ceil    = proc.get_nearest(h_ceil,z_t.values)
                
            # Retrieve the ACF, with lag 0 at the detrain month
            acf_floor = acf_ens.isel(lat=a,lon=o,mon=dtid_floor,z_t=zz_floor)     # [Lag]
            acf_ceil  = acf_ens.isel(lat=a,lon=o,mon=dtid_ceil,z_t=zz_ceil)       # [Lag]
            
            # Calculate dlag
            dlag_floor = entrid - dtid_floor
            if dlag_floor < 1:
                dlag_floor = dlag_floor + 12
            dlag_ceil  = entrid - dtid_ceil
            if dlag_ceil < 1:
                dlag_ceil  = dlag_ceil + 12
            
            # Retrieve Correlation
            corr_floor = acf_floor.isel(lag=dlag_floor).values.item()
            corr_ceil  = acf_ceil.isel(lag=dlag_ceil).values.item()
            
            # Interp if option is chosen
            if interpcorr:
                dm       = detrain_mon - (dtid_floor+1)
                corr_mon = np.interp(dm,[0,1],[corr_floor,corr_ceil])
                #corr_mon = np.interp(detrain_mon,[dtid_floor+1,dtid_ceil+1],[corr_floor,corr_ceil])
            elif detrainceil:
                corr_mon = corr_ceil
            else:
                
                corr_mon = corr_floor
            corr_out[im,a,o] = corr_mon.copy()
#%%
# Save the output (for a variable and ensemble member)
if coords2D:
    nlat       = np.arange(0,nlat)
    nlon       = np.arange(0,nlon)
    coords = dict(mon=np.arange(1,13,1),lat=nlat,lon=nlon)
    da_corr = xr.DataArray(corr_out,coords=coords,dims=coords,name="lbd_d")
    
    da_out = xr.merge([da_corr,da_TLONG,da_TLAT,])
else:
    coords = dict(mon=np.arange(1,13,1),lat=lat,lon=lon)
    
    da_out = xr.DataArray(corr_out,coords=coords,dims=coords,name="lbd_d")
edict  = {'lbd_d':{'zlib':True}}
savename = "%s%s_%s_corr_d_%s_detrend%s_lagmax%i_interp%i_ceil%i_imshift%i_dtdepth%i_%s.nc" % (mimocpath,dataset_name,mldname,"TEMP","RAW",lagmax,
                                                                                                  interpcorr,detrainceil,imshift,dtdepth,tstr)
da_out.to_netcdf(savename,encoding=edict)                

#%% Check Results

if coords2D:
    
    for im in range(12):
        plt.scatter(da_out.TLONG,da_out.TLAT,c=da_out.lbd_d.isel(mon=im),),plt.colorbar(),plt.show()
else:
    for im in range(12):
        da_out.isel(mon=im).plot(),plt.colorbar(),plt.show()



        
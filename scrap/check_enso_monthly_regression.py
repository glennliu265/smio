#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Using ERA5, check how different types of detrending will impact the ENSO index

- Copied hfcalc: compute_enso_index.py


Created on Wed Sep 17 16:28:51 2025

@author: gliu
"""


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import time
import sys
import cartopy.crs as ccrs
import glob
import pandas as pd

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

#%% ENSO Calculation and Cropping Options

# Select time crop (prior to preprocessing)
# croptime          = True # Cut the time prior to detrending, EOF, etc
# tstart            = '1979-01-01' #'1920-01-01'#'0001-01-01' # "2006-01-01" # 
# tend              = '2024-12-31'#'2021-12-31' #'2005-12-31'#'2000-02-01' # "2101-01-01" # 
# timestr           = "%sto%s" % (tstart[:4],tend[:4])

# ENSO Parameters
pcrem               = 3                   # PCs to calculate
bbox                = [120, 290, -20, 20] # ENSO Bounding Box
#bbox               =  [-40,-15,52,62]

# Toggles and Options
overwrite           = True # Set to True to overwrite existing output...
save_netcdf         = True # Set true to save netcdf version, false to save npz
debug               = True # Debug toggle

#%% Dataset option (load full TS variable in [time x lat x lon360])
# Example provided below here is for CESM1


# Output Path (Checks for an "enso" folder)
outpath             = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/enso/ensotest/"#"/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/output/"

# ERA5 with Monthly GMSST Removal
dataset_name        = "ERA5_ensotest"
datpath             = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
ncname              = "sst_1979_2024.nc"

# NetCDF Information
vname               = "sst"
lonname             = "longitude"
latname             = "latitude"
timename            = "valid_time"
concat_dim          = "time"
keepvars            = [timename,latname,lonname,vname]
ensnum              = 1 # Irrelevant for now, need to add ensemble support...
detrend             = 1 # 1 to remove linear trend 

croptime            = False # Cut the time prior to detrending, EOF, etc
tstart              = '1979-01-01' 
tend                = '2024-12-31'
timestr             = "%sto%s" % (tstart[:4],tend[:4])
yr_range            = timestr

# Mask Information (first run a maskmaker script/section such as that in preproc_CESM2_PiControl.py)
maskpath            = None
maskname            = None

#%% Load GMSST Information (for detrending)


# Load GMSST
dpath_gmsst = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
nc_gmsst    = "ERA5_GMSST_1979_2024.nc"
ds_gmsst    = xr.open_dataset(
    dpath_gmsst + nc_gmsst).load()  # .GMSST_MeanIce.load()



#%% Load Variable

# Load Variable (7.09sec)
st=time.time()
ds_all = xr.open_dataset(datpath+ncname).load()
proc.printtime(st)

#%% Preprocess

# Rename Dimensions
if np.any(np.array(bbox[:2]) < 0):
    lon180=True
else:
    lon180=False
ds_all = proc.format_ds(ds_all,latname=latname,lonname=lonname,timename=timename,lon180=lon180,verbose=True)#ds_all.rename(dict(valid_time='time'))


ds_reg = proc.sel_region_xr(ds_all,bbox)

# Deseason
dsa = proc.xrdeseason(ds_reg.sst)

#%% Now Try Different Detrending Approaches


# (1): Simple Linear Detrend (9.68s)
dt1     = proc.xrdetrend(dsa)
dt1mon  = proc.xrdetrend_nd(dsa,1,return_fit=False,regress_monthly=True)

# (2): Quadratic Detrend (separately and separately by month)
dt2     = proc.xrdetrend_nd(dsa,2,return_fit=False)
dt2mon  = proc.xrdetrend_nd(dsa,2,return_fit=False,regress_monthly=True)

# (3): Removing GMSST
gmout       = proc.detrend_by_regression(dsa,ds_gmsst.GMSST_MeanIce)
gmoutmon    = proc.detrend_by_regression(dsa,ds_gmsst.GMSST_MeanIce,regress_monthly=True)

dt3         = gmout.sst
dt3mon      = gmoutmon.sst


#%% Visualize timeseries over Nino3.4 box

bboxnino34 = [360-170,360-120,-5,5,]
#bboxnino34 = bbox# For SPGNE Region
dtall      = [dsa,dt1,dt1mon,dt2,dt2mon,dt3,dt3mon]
expnames   = ["Raw","Linear","Linear (Monthly)","Quadratic","Quadratic (Monthly)","GMSST Removal","GMSST Removal (Monthly)"]


expcols    = ['gray',"hotpink","red","cornflowerblue","midnightblue","goldenrod","brown"]
els        = ["solid","solid","dashed","solid",'dashed',"solid",'dashed']

ndt = len(dtall)
dtall34 = [proc.sel_region_xr(ds,bboxnino34) for ds in dtall]
aavgs   = [proc.area_avg_cosweight(ds) for ds in dtall34]


#%% Plot the timeseries

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,3.5))

for n in range(ndt):
    plotvar = aavgs[n]
    ax.plot(plotvar.time,plotvar,label=expnames[n],c=expcols[n],ls=els[n],lw=2)
    
    
ax.legend(ncol=3,fontsize=8)
ax.set_xlabel("Year")
ax.set_ylabel("SST Anomaly (degC)")

ax.set_title("Nino3.4 Index (ERA5, 1979-2024)")
#ax.set_title("SPGNE (ERA5, 1979-2024)")
#
ax.set_ylim([-2,2])
ax.set_ylim([-3.5,3.5])
ax.set_xlim([plotvar.time[-60],plotvar.time[-1]])


#%% Plot what happens to monthly variance

monvar = [ds.groupby('time.month').std('time') for ds in aavgs]
mons3 = proc.get_monstr()

fig,ax = viz.init_monplot(1,1,figsize=(8,4))
for n in range(ndt):
    plotvar = monvar[n]
    ax.plot(mons3,plotvar,label=expnames[n],c=expcols[n],ls=els[n],lw=2)
ax.legend()
ax.set_ylabel("Monthly Standard Deviation (degC)")


#%% Lets quickly calculate some other metrics

lags    = np.arange(61)
metrics = scm.compute_sm_metrics([ds.data for ds in aavgs],lags=lags,detrend_acf=False,nsmooth=2)

#%% Plot the ACF

kmonth = 2
fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,3.5))

ax,_ = viz.init_acplot(kmonth,lags[::3],lags,ax=ax)

for n in range(ndt):
    plotvar = metrics['acfs'][kmonth][n]
    ax.plot(lags,plotvar,label=expnames[n],c=expcols[n],ls=els[n],lw=2)
ax.legend(ncol=3)
ax.set_ylabel("Correlation with %s Anomalies" % (mons3[kmonth]))

#%% Plot the power spectra  

decadal_focus = False

dtmon_fix       = 60*60*24*30

if decadal_focus:
    xper            = np.array([10,7,5,3,1,0.5])
else:
    xper            = np.array([40,10,7,5,3,1,0.5])
xper_ticks      = 1 / (xper*12)

fig,ax      = plt.subplots(1,1,figsize=(8,4.5),constrained_layout=True)

for ii in range(ndt):

    plotspec        = metrics['specs'][ii] / dtmon_fix
    plotfreq        = metrics['freqs'][ii] * dtmon_fix
    CCs             = metrics['CCs'][ii] / dtmon_fix

    ax.loglog(plotfreq,plotspec,lw=2.5,label=expnames[ii],c=expcols[ii],ls=els[ii])
    
    #ax.loglog(plotfreq,CCs[:,0],ls='dotted',lw=0.5,c=expcols[ii])
    #ax.loglog(plotfreq,CCs[:,1],ls='dashed',lw=0.9,c=expcols[ii])

ax.set_xlim([xper_ticks[0],0.5])
ax.axvline([1/(6)],label="",ls='dotted',c='gray')
ax.axvline([1/(12)],label="",ls='dotted',c='gray')
ax.axvline([1/(3*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(5*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(7*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(10*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(40*12)],label="",ls='dotted',c='gray')

ax.set_xlabel("Frequency (1/Month)",fontsize=14)
ax.set_ylabel("Power [$\degree C ^2 cycle \, per \, mon$]")

ax2 = ax.twiny()
ax2.set_xlim([xper_ticks[0],0.5])
ax2.set_xscale('log')
ax2.set_xticks(xper_ticks,labels=xper)
ax2.set_xlabel("Period (Years)",fontsize=14)


# # Plot Confidence Interval (ERA5)
# alpha           = 0.05
# cloc_era        = [plotfreq[0],1e-1]
# dof_era         = metrics_out['dofs'][-1]
# cbnds_era       = proc.calc_confspec(alpha,dof_era)
# proc.plot_conflog(cloc_era,cbnds_era,ax=ax,color=dfcol,cflabel=r"95% Confidence") #+r" (dof= %.2f)" % dof_era)

ax.legend()


#%% Function Workshop (write and ND xarray detrender)


def xrdetrend_nd(invar,order,regress_monthly=False,return_fit=False):
    # Given an DataArray [invar] and order
    # Fit the timeseries and detrend
    
    # Change to [lon x lat x time]
    reshape_flag = False
    try:
        invar       = invar.transpose('lon','lat','time')
        invar_arr   = invar.data # [lon x lat x time]
        
    except:
        print("Warning, input is not 3d or doesn't have ('lon','lat','time')")
        reshape_output = proc.make_2d_ds(invar,keepdim='time') #[1 x otherdims x time]
        invar_arr      = reshape_output[0].data
        reshape_flag = True
    
    # Filter out NaN points
    nlon,nlat,ntime = invar_arr.shape
    invar_rs = invar_arr.reshape(nlon*nlat,ntime) # Space x Time
    nandict  = proc.find_nan(invar_rs,1,return_dict=True)
    
    if regress_monthly: # Do regression separately for each month
    
        # Reshape to space x yr x mon
        cleaned_data = nandict['cleaned_data']
        nok,_        = cleaned_data.shape
        nyr          = int(ntime/12)
        cd_yrmon     = cleaned_data.reshape((nok,nyr,12)) # Check 
        
        # Preallocate and detrend separately for each month
        detrended_bymon = np.zeros(cd_yrmon.shape)*np.nan # Detrended Variable
        fit_bymon       = detrended_bymon.copy() # n-order polynomial fit
        for im in range(12):
            # Get data for month
            cdmon                   = cd_yrmon[:,:,im]
            xdim                    = np.arange(nyr)
            detrended_mon,fit_mon   = proc.detrend_poly(xdim,cdmon,order)
            detrended_bymon[:,:,im] = detrended_mon.T
            fit_bymon[:,:,im]       = fit_mon
            
            # Debug Plot (by month)
            # ii = 22
            # fig,ax = plt.subplots(1,1)
            # ax.plot(xdim,detrended_mon[:,ii],label="Detrended",color='blue')
            # ax.plot(xdim,fit_mon[ii,:],label="Fit",color="red")
            # ax.plot(xdim,cdmon[ii,:],label="Raw",color='gray',ls='dashed')
            # ax.legend()
        
        # Reshape the variables
        detrended_bymon = detrended_bymon.reshape(nok,ntime) # [Space x Time]
        fit_bymon = fit_bymon.reshape(nok,ntime) # [Space x Time]
        
        # # Debug Plot (full timeseries)
        # ii = 77
        # fig,ax = plt.subplots(1,1)
        # xdim = np.arange(ntime)
        # ax.plot(xdim,detrended_bymon.T[:,ii],label="Detrended",color='blue')
        # ax.plot(xdim,data_fit[ii,:],label="Fit",color="red")
        # ax.plot(xdim,nandict['cleaned_data'][ii,:],label="Raw",color='gray',ls='dashed')
        # ax.legend()
        
        # Replace Detrended data in original array
        arrout = np.zeros((nlon*nlat,ntime)) * np.nan
        arrout[nandict['ok_indices'],:] = detrended_bymon
        arrout = arrout.reshape(nlon,nlat,ntime).transpose(2,1,0) # Flip to time x lat x lon
        
        if return_fit:
            fitout = np.zeros((nlon*nlat,ntime)) * np.nan
            fitout[nandict['ok_indices'],:] = fit_bymon
            fitout = fitout.reshape(nlon,nlat,ntime).transpose(2,1,0)
        
    else: # Do regression for all months together...
        
        # Apply a fit using proc.detrend_poly
        xdim = np.arange(invar.shape[-1]) # Length of time dimension
        data_detrended,data_fit = proc.detrend_poly(xdim,nandict['cleaned_data'],order)
        #data_detrended         # [time x space]
        #data_fit               # [space x Time]
        #nandict['cleaned_data' # [space x time]
        
        # # Debug Plot
        ii = 77
        fig,ax = plt.subplots(1,1)
        ax.plot(xdim,data_detrended[:,ii],label="Detrended",color='blue')
        ax.plot(xdim,data_fit[ii,:],label="Fit",color="red")
        ax.plot(xdim,nandict['cleaned_data'][ii,:],label="Raw",color='gray',ls='dashed')
        ax.legend()
        
        # Replace Detrended data in original array
        arrout = np.zeros((nlon*nlat,ntime)) * np.nan
        arrout[nandict['ok_indices'],:] = data_detrended.T
        arrout = arrout.reshape(nlon,nlat,ntime).transpose(2,1,0) # Flip to time x lat x lon
        
        if return_fit:
            fitout = np.zeros((nlon*nlat,ntime)) * np.nan
            fitout[nandict['ok_indices'],:] = data_fit
            fitout = fitout.reshape(nlon,nlat,ntime).transpose(2,1,0)
    
    # Replace into data array
    
    
    # Prepare Output as DataArrays # [(time) x lat x lon]
    if reshape_flag is False: # Directly transpose and assign coords [time x lat x lon]
        coords_full     = dict(time=invar.time,lat=invar.lat,lon=invar.lon)
        if regress_monthly: # Add "mon" coordinate for monthly regression
            coords          = dict(mon=np.arange(1,13,1),lat=invar.lat,lon=invar.lon)
        else:
            coords          = dict(lat=invar.lat,lon=invar.lon)
        
        da_detrend      = xr.DataArray(arrout,coords=coords_full,dims=coords_full,name=invar.name)
        if return_fit:
            da_fit          = xr.DataArray(fitout,coords=coords_full,dims=coords_full,name='fit')
    
    else: # Need to undo reshaping and reassign old coords...
        da_detrend      = reshape_2d_ds(arrout,invar,reshape_output[2],reshape_output[1])
        da_fit          = reshape_2d_ds(fitout,invar,reshape_output[2],reshape_output[1])
        
    if return_fit:
        dsout = xr.merge([da_detrend,da_fit])
    else:
        dsout = da_detrend
    return dsout























#%%



#%%

#%% Find File and Load Variable

# 1A. Load Variable ----------

# Create filename/list and load 
if dataset_name == "cesm2_pic":
    searchstr = "%s%s/*%s*.nc" % (datpath,vname,vname) # Searches for datpath + *LANDFRAC*.nc
elif dataset_name == "OISST": # Just grab tropical pacific
    searchstr = "%s%s*%s*TropicalPacific.nc" % (datpath,dataset_name,vname) # Searches for datpath + *LANDFRAC*.nc
elif dataset_name == "CESM2_FOM":
    searchstr = datpath + "b.e21.B1850.f09_g17.CMIP6-piControl.001.cam.h0.TS.*.nc" 
elif dataset_name == "CESM2_SOM":
    searchstr = datpath + "e.e21.E1850.f09_g17.CMIP6-piControl.001_branch2.cam.h0.TS.*.nc"
    
elif dataset_name == "CESM2_MCOM":
    searchstr = datpath + "b.e21.B1850.f09_g17.1dpop2-gterm.005.*.TS.nc"
elif dataset_name == "ERA5": # Tropical Pacific Box
    if yr_range is None:
        yr_range = "1979to2021"
    searchstr = "%s%s*%s*TropicalPacific_%s.nc" % (datpath,dataset_name,vname,yr_range) # Searches for datpath + *LANDFRAC*.nc
else:
    searchstr = "%s%s*%s*.nc" % (datpath,dataset_name,vname) # Searches for datpath + dataset_name*LANDFRAC*.nc"
nclist    = glob.glob(searchstr)
nclist.sort()
print("Found %i files for %s" % (len(nclist),vname))

# Drop Unnecessary variables
if concat_dim is None or len(nclist) == 1: # Assume no concantenation is needed
    ds_all    = xr.open_dataset(nclist[0])
else:
    print("Concatenating files by dim <%s>" % (concat_dim))
    ds_all    = xr.open_mfdataset(nclist,concat_dim=concat_dim,combine='nested')
ds_all    = hf.ds_dropvars(ds_all,keepvars)
try:
    ds_all    = hf.fix_febstart(ds_all)
except:
    print("Warning, Time is not in datetime... converting")
    print("First timestep (pre conversion) is %s" % (ds_all[timename][0]))
    timeconv = pd.to_datetime(ds_all.time.data)
    ds_all[timename] = timeconv
    print("First timestep (post conversion) is %s" % (ds_all[timename][0]))
    
    #ds_all['time2'] = pd.to_datetime(ds_all.time.data)
    print("Warning: February Start Fix was not implemented")
    #print("First timestep is %s" % (ds_all[timename][0]))

# Load it
st        = time.time()
ds_all    = ds_all[vname]#.load()
print("Loaded in %.2fs" % (time.time()-st))

# 1B. Load and Apply Mask
if (maskpath is None) or (maskname is None):
    print("No mask will be applied")
else:
    st = time.time()
    mask = xr.open_dataset(maskpath+maskname).mask.load()
    ds_all = ds_all * mask
    print("Mask applied in %.2fs" % (time.time()-st))

# Set ensemble flag
lensflag = False
if 'ens' in list(ds_all.dims):
    print("Ensemble dimension detected!")
    lensflag = True

#%% 2. Preprocessing

"""

From this point on you should have:
    
    TS [time x lat x lon360]... Land-Ice Mask Applied

"""

# Select the Box and time period  load
st    = time.time()
dsreg = ds_all.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
dsreg = dsreg.sel(time=slice(tstart,tend))
dsreg = dsreg.load()
print("Output Loaded in %.2fs" % (time.time()-st))

# Remove seasonal cycle
#ds_anom  = hf.xrdeseason(dsreg) # hf.xrdeseason is not working?
ds_anom  = proc.xrdeseason(dsreg,check_mon=False)

# Remove Trend if option is set
if detrend:
    
    if 'ens' in list(ds_anom.dims):
        print("Detrending by removing ensemble mean")
        # Detrend by removing ensemble average
        da = ds_anom - ds_anom.mean('ens')
        
        da = da.transpose('ens','time','lat','lon')
        
    else:
        print("Detrending by removing linear fit")
        ds_anom   = ds_anom.transpose('time','lat','lon')
        
        # Simple Linear Detrend
        dt_dict   = hf.detrend_dim(ds_anom.values,0,return_dict=True)# ASSUME TIME in first axis
        
        # Put back into DataArray
        da = xr.DataArray(dt_dict['detrended_var'],dims=ds_anom.dims,coords=ds_anom.coords,name=vname)

else:
    da = ds_anom.copy()

print("Data preprocessed in %.2fs" % (time.time()-st))



#%% Part 3, Compute ENSO Indices (copied from calc_ENSO_general)

# ------------------- -------- General Portion --------------------------------

"""

IN : ncfile, <dataset_name>_<vname>_manom_detrend#.nc
    Anomalized, detrended ts with landice masked applied
    
OUT : npz file <dataset_name>_ENSO_detrend#_pcs#.npz
    PC File containing:
        eofall (ENSO EOF Patterns)          [lon x lat x month x pc]
        pcall  (ENSO principle components)  [(ens) x time x month x pc]
        varexpall (ENSO variance explained) [month x pc]]
        lon,lat,time,ensobbox variables

ex: ncep_ncar_ts_manom_detrend1.nc --> ncep_ncar_ENSO_detrend1_pcs3.npz

"""

st = time.time()

# # Open the dataset
# savename = "%s%s_%s_manom_detrend%i_%s.nc" % (datpath,dataset_name,vnames_in[0],detrend,timestr)
# if lensflag:
#     savename = proc.addstrtoext(savename,"_ens%02i"%(ensnum),adjust=-1)
# da = xr.open_dataset(savename)

# Slice to region
#da = da.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))

# Check if ENSO has already been calculated and skip if so
if outpath is None:
    proc.makedir("%senso/"% datpath) 
    savename = "%senso/%s_ENSO_detrend%i_pcs%i_%s.nc" % (outpath,dataset_name,detrend,pcrem,timestr)
else:
    savename = "%s%s_ENSO_detrend%i_pcs%i_%s.nc" % (outpath,dataset_name,detrend,pcrem,timestr)

# if lensflag:
#     savename = proc.addstrtoext(savename,"_ens%02i"%(ensnum),adjust=-1)
query = glob.glob(savename)
if (len(query) < 1) or (overwrite == True):
    
    # Read out the variables # [(ens) x time x lat x lon]
    st        = time.time()
    invar     = da.values
    lon       = da[lonname].values
    lat       = da[latname].values
    times     = da[timename].values
    print("Data loaded in %.2fs"%(time.time()-st))
    
    if lensflag:
        nens = len(da.ens)
        
        eofall_ens      = []
        pcall_ens       = []
        varexpall_ens   = []
        for e in range(nens):
            invar_ens              = invar[e,...]
            # Portion Below is taken from calc_ENSO_PIC.py VV ***********
            eofall,pcall,varexpall = scm.calc_enso(invar_ens,lon,lat,pcrem,bbox=bbox)
            
            eofall_ens.append(eofall.copy())
            pcall_ens.append(pcall.copy())
            varexpall_ens.append(varexpall.copy())
        
        eofall    = np.array(eofall_ens)
        pcall     = np.array(pcall_ens)
        varexpall = np.array(varexpall_ens)
        
    else:
        # Portion Below is taken from calc_ENSO_PIC.py VV ***********
        eofall,pcall,varexpall = scm.calc_enso(invar,lon,lat,pcrem,bbox=bbox)
    
    # Sanity Check ------------------------------------------------------------
    if debug:
        im = 0
        ip = 0
        proj = ccrs.PlateCarree(central_longitude=180)
        fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
        ax = viz.add_coast_grid(ax,bbox=bbox)
        if lensflag:
            plotvar = eofall[0,:,:,im,ip]
        else:
            plotvar = eofall[:,:,im,ip]
            
        pcm = ax.pcolormesh(lon,lat,plotvar,vmin=-1,vmax=1,
                            cmap='cmo.balance',transform=ccrs.PlateCarree())
        cb = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.055,pad=0.1)
        cb.set_label("SST Anomaly ($\degree C \sigma_{ENSO}^{-1}$)")
        ax.set_title("EOF %i, Month %i\n Variance Explained: %.2f" % (ip+1,im+1,varexpall[im,ip]*100)+"%")
    
    # Saving ------------------------------------------------------------------
    if save_netcdf:
        
        mons    = np.arange(1,13,1)
        
        years   = np.arange(int(len(times)/12))
        pcnums  = np.arange(1,pcrem+1)
        
        # Make Dictionary
        coords_eofs   = dict(lat=lat,lon=lon,month=mons,pc=pcnums) # 
        coords_pcs    = dict(year=years,month=mons,pc=pcnums)
        coords_varexp = dict(month=mons,pc=pcnums)
        if lensflag:
            ens     = np.arange(1,nens+1,1)
            # Unpack and repack dict to append item to start # https://www.geeksforgeeks.org/python-append-items-at-beginning-of-dictionary/
            coords_eofs,coords_pcs,coords_varexp = [{**{'ens':ens},**dd} for dd in [coords_eofs,coords_pcs,coords_varexp]]
        
        
        da_eofs       = xr.DataArray(eofall,coords=coords_eofs,dims=coords_eofs,name='eofs')
        da_pcs        = xr.DataArray(pcall,coords=coords_pcs,dims=coords_pcs,name='pcs')
        da_varexp     = xr.DataArray(varexpall,coords=coords_varexp,dims=coords_varexp,name='varexp')
        
        # Merge everything
        da_out        = xr.merge([da_eofs,da_pcs,da_varexp])
        
        # Add Additional Variables
        da_out['time']      = times
        da_out['enso_bbox'] = bbox
        
        edict = proc.make_encoding_dict(da_out)
        da_out.to_netcdf(savename,encoding=edict)
        
    else:
        
        # Save Output
        np.savez(savename,**{
                 'eofs': eofall, # [(ens) x lon x lat x month x pc]
                 'pcs': pcall,   # [Year, Month, PC]
                 'varexp': varexpall,
                 'lon': lon,
                 'lat':lat,
                 'times':times,
                 'enso_bbox':bbox}
                )
    print("Data saved to %s in %.2fs"%(savename,time.time()-st))
else:
    print("Skipping. Found existing file: %s" % (str(query)))
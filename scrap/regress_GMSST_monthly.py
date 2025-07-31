#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test script to regress GMSST, but using monthly values

Created on Wed Jul 30 17:59:32 2025

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

#%% Load GMSST for Detrending (from common_load)

# Load GMSST 
dpath_gmsst     = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
nc_gmsst        = "ERA5_GMSST_1979_2024.nc"
ds_gmsst        = xr.open_dataset(dpath_gmsst + nc_gmsst).load()#.GMSST_MeanIce.load()
detrend_gm      = lambda ds_in: proc.detrend_by_regression(ds_in,ds_gmsst.Mean_Ice)

# Load GMSST (Older)
nc_gmsst_pre    = "ERA5_GMSST_1940_1978.nc"
ds_gmsst_pre    = xr.open_dataset(dpath_gmsst + nc_gmsst_pre).load()#.GMSST_MeanIce.load()

ds_gmsst_merge = xr.concat([ds_gmsst_pre,ds_gmsst],dim='time')

# Set Figure Path
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20250801/"
proc.makedir(figpath)


#%% Load ERA5 Datasets

dp      = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
nc2     = "ERA5_sst_NAtl_1940to1978.nc"
nc1     = "ERA5_sst_NAtl_1979to2024.nc"
ncs     = [dp+nc2,dp+nc1,]


dsall   = xr.open_mfdataset(ncs,combine='nested',concat_dim='time').load()





#%% Restrict to Region/Time

ystart      = '1979'
yend        = '2024'
bbox        = [-80,0,0,65]#[-40,-15,52,62]
dsreg       = proc.sel_region_xr(dsall,bbox)
dsreg       = dsreg.sel(time=slice("%s-01-01" % ystart,"%s-12-31"%yend))

ds_gmsst_sel =  ds_gmsst_merge.GMSST_MaxIce.sel(time=slice("%s-01-01" % ystart,"%s-12-31"%yend))

#%% Initial Preprocessing (Deseasonalize)

dsanom = proc.xrdeseason(dsreg).sst


#%% Now Perform Monthly regression (based on function)


invar = dsanom
in_ts = ds_gmsst_sel
regress_monthly = True

mon1 = proc.detrend_by_regression(invar,in_ts,regress_monthly=True)
mon0 = proc.detrend_by_regression(invar,in_ts,regress_monthly=False)


#%% Lets Explore Some Differences
lags = np.arange(61)


sst_in = [mon0.sst,mon1.sst]
aavgs  = [proc.area_avg_cosweight(sst).data for sst in sst_in]

tsm    = scm.compute_sm_metrics(aavgs,lags=lags,nsmooth=2,detrend_acf=False)



#%% Check Impacts on ACF


kmonth = 2
xtks   = lags[::3]


fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,4.5))
ax,_ = viz.init_acplot(kmonth,xtks,lags,ax=ax)

mlabs=["All Months","Separately by Month"]
for ii in range(2):
    plotvar = tsm['acfs'][kmonth][ii]
    ax.plot(lags,plotvar,label=mlabs[ii],lw=2.5)

ax.legend()

#%% Check Regression Coefficients over the region

mons3   = proc.get_monstr()

fig,ax  = viz.init_monplot(1,1)
regpat  = mon1['regression_pattern']


# Plot Monthly Values
pat_avg = proc.area_avg_cosweight(regpat)
_,nlat,nlon = regpat.shape
for a in tqdm.tqdm(range(nlat)):
    for o in range(nlon):
        plotvar = regpat.isel(lat=a,lon=o)
        ax.plot(mons3,plotvar,alpha=0.1)
ax.plot(mons3,pat_avg,color="k",label="Region Avg., Separate Month")


# Now Get the Constants
regpat0 = mon0['regression_pattern']
avg0    = proc.area_avg_cosweight(regpat0)

std0    = regpat0.data.reshape(nlat*nlon).std()
ax.axhline([avg0],label="Regression Coeff (Region avg., All Months)",color='midnightblue',ls='dashed')

ax.axhline([avg0+std0],label="",color='midnightblue',ls='dotted')

ax.axhline([avg0-std0],label="Regression Coeff (Region std, All Months)",color='midnightblue',ls='dotted')


ax.legend()

#%% Look At the Regression patterns
proj      = ccrs.PlateCarree()

fsz_title = 18
fsz_ticks = 18

# regpat  = mon1['intercept']
# regpat0 = mon0['intercept']

vmax    = 6
regpat  = mon1['regression_pattern']

for im in range(12):
    
    #fig,ax,_ = viz.init_regplot(regname="SPGE")
    fig,ax,_ = viz.init_regplot()
    plotvar = regpat.isel(mon=im)
    
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                        vmin=-vmax,vmax=vmax,cmap='cmo.balance')


    plotvar = mon1.sigmask.isel(mon=im)
    xx      = viz.plot_mask(plotvar.lon,plotvar.lat,plotvar.T,markersize=.1,
                            proj=proj,geoaxes=True,ax=ax)
    
    cb = viz.hcbar(pcm,ax=ax)
    cb.ax.tick_params(labelsize=fsz_ticks)
    
    cb.set_label("SST (deg C per deg C GMSST",fontsize=fsz_ticks)
    ax.set_title("%s Regression" % mons3[im],fontsize=fsz_title)
    
    figname = "%sGMSST_Regression_Year_%sto%s_mon%02i.png" % (figpath,ystart,yend,im+1)
    plt.savefig(figname,dpi=150,bbox_inches='tight')


#%% Plot Pattern for All Months

fig,ax,_ = viz.init_regplot()
regpat0 = mon0['regression_pattern']
plotvar = regpat0#.isel(mon=im)

pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                    vmin=-vmax,vmax=vmax,cmap='cmo.balance')

plotvar = mon0.sigmask
xx      = viz.plot_mask(plotvar.lon,plotvar.lat,plotvar.T,markersize=.1,
                        proj=proj,geoaxes=True,ax=ax)

cb = viz.hcbar(pcm,ax=ax)
cb.ax.tick_params(labelsize=fsz_ticks)
cb.set_label("SST (deg C per deg C GMSST",fontsize=fsz_ticks)
ax.set_title("All Months Regression",fontsize=fsz_title)

figname = "%sGMSST_Regression_Year_%sto%s_ALL_Month.png" % (figpath,ystart,yend)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Check Timeseries Removal At a Point

lonf            = -37
latf            = 37
locfn,loctitle  = proc.make_locstring(lonf,latf)

loopds = [mon0,mon1]
cols = ['cornflowerblue','hotpink']
loopnames = ["All Month","Seperate Month"]

loopname_short = ["AllMon","SepMon"]






# PLot GMSST removal
for ii in range(2):
    
    fig,ax = plt.subplots(1,1,figsize=(12.5,4.5))
    
    # PLot Raw SST
    sst_raw = proc.selpt_ds(dsanom,lonf,latf)
    ax.plot(sst_raw.time,sst_raw,label="Raw SST Anomaly",c='midnightblue')
    
    loopin = loopds[ii]
    
    sstpt  = proc.selpt_ds(loopin.sst,lonf,latf)
    fitpt  = proc.selpt_ds(loopin.fit,lonf,latf)
    
    #xrange = np.arange(len(fitpt))
    lab  = loopnames[ii]
    ax.plot(sstpt.time.data,sstpt,color=cols[ii],ls='solid',label="Detrended",lw=1)
    ax.plot(fitpt.time.data,fitpt,color="k",ls='dashed',label="Fit")
    
    #ax.plot(xrange,fitpt,color=cols[ii],ls='dashed')
    ax.legend(fontsize=14,ncol=2)
    ax.axhline([0],color='k',lw=0.44)
    ax.set_ylim([-2.5,2.5])
    ax.set_ylabel("SST (degC)",fontsize=14)
    ax.set_xlim([sstpt.time.isel(time=0),sstpt.time.isel(time=-1)])
    ax.tick_params(labelsize=14)
    ax.set_title("GMSST Regression using: " + loopnames[ii] + "s\n %s" % loctitle,fontsize=16)
    
    figname = "%sGMSST_Regression_Timeseries_%s_%s.png" % (figpath,locfn,loopname_short[ii])
    plt.savefig(figname,dpi=150,bbox_inches='tight')
    
#%% Examine Impact on SST

sstin = [proc.selpt_ds(mon0.sst,lonf,latf).data,proc.selpt_ds(mon1.sst,lonf,latf).data]




#%%

#%% Scrap below
def detrend_by_regression(invar,in_ts,regress_monthly=False):
    # Given an DataArray [invar] and Timeseries [in_ts]
    # Detrend the timeseries by regression
    
    # Change to [lon x lat x time]
    
    reshape_flag = False
    try:
        invar       = invar.transpose('lon','lat','time')
        invar_arr   = invar.data # [lon x lat x time]
        
    except:
        print("Warning, input is not 3d or doesn't have ('lon','lat','time')")
        reshape_output = make_2d_ds(invar,keepdim='time') #[1 x otherdims x time]
        invar_arr      = reshape_output[0].data
        reshape_flag = True
    ints_arr         = in_ts.data # [time]
    
    if regress_monthly: # Do regression separately for each month
        
        nlon,nlat,ntime = invar_arr.shape
        nyr             = int(ntime/12)
        ints_monyr      = ints_arr.reshape(nyr,12)
        invar_monyr     = invar_arr.reshape(nlon,nlat,nyr,12) # [lat x lon x yr x mon]
        
        betas      = []
        intercepts = []
        ymodels    = []
        ydetrends  = []
        sigmasks   = []
        for im in range(12):
            
            outdict     = regress_ttest(invar_monyr[:,:,:,im],ints_monyr[:,im])
            beta        = outdict['regression_coeff'] # Lon x Lat
            intercept   = outdict['intercept'] 
            
            
            # Remove the Trend
            ymodel      = beta[:,:,None] * ints_monyr[None,None,:,im] + intercept[:,:,None]
            ydetrend    = invar_monyr[:,:,:,im] - ymodel
            
            betas.append(beta)
            intercepts.append(intercept)
            ymodels.append(ymodel)
            ydetrends.append(ydetrend)
            sigmasks.append(outdict['sigmask'])
        
        beta        = np.array(betas)       # [Month x lon x lat]
        intercept   = np.array(intercepts)  # [Month x lon x lat]
        ymodel      = np.array(ymodels)     # [Month x lon x lat x yr]
        ydetrend    = np.array(ydetrends)   # [Month x lon x lat x yr]
        sigmasks    = np.array(sigmasks)
        
        ymodel      = ymodel.transpose(1,2,3,0).reshape(nlon,nlat,ntime)
        ydetrend    = ydetrend.transpose(1,2,3,0).reshape(nlon,nlat,ntime)
        
        # Flip to [time x lat x lon]
        sigmask_out     = sigmasks.transpose(0,2,1) 
        beta            = beta.transpose(0,2,1)
        intercept       = intercept.transpose(0,2,1)
        
    else:
        
        # Perform the regression
        outdict     = regress_ttest(invar_arr,ints_arr)
        beta        = outdict['regression_coeff'] # Lon x Lat
        intercept   = outdict['intercept'] 
        
        # Remove the Trend
        ymodel      = beta[:,:,None] * ints_arr[None,None,:] + intercept[:,:,None]
        ydetrend    = invar_arr - ymodel
        
        # Prepare for input [lat x lon]
        sigmask_out     = outdict['sigmask'].T
        beta            = beta.T
        intercept       = intercept.T
        
    # Prepare Output as DataArrays # [(time) x lat x lon]
    
    if reshape_flag is False: # Directly transpose and assign coords [time x lat x lon]
        coords_full     = dict(time=invar.time,lat=invar.lat,lon=invar.lon)
        if regress_monthly: # Add "mon" coordinate for monthly regression
            coords          = dict(mon=np.arange(1,13,1),lat=invar.lat,lon=invar.lon)
        else:
            coords          = dict(lat=invar.lat,lon=invar.lon)
        
        da_detrend      = xr.DataArray(ydetrend.transpose(2,1,0),coords=coords_full,dims=coords_full,name=invar.name)
        da_fit          = xr.DataArray(ymodel.transpose(2,1,0),coords=coords_full,dims=coords_full,name='fit')
        
        da_pattern      = xr.DataArray(beta,coords=coords,dims=coords,name='regression_pattern')
        da_intercept    = xr.DataArray(intercept,coords=coords,dims=coords,name='intercept')
        da_sig          = xr.DataArray(sigmask_out,coords=coords,dims=coords,name='sigmask')
        
    else: # Need to undo reshaping and reassign old coords...
        
        da_detrend      = reshape_2d_ds(ydetrend,invar,reshape_output[2],reshape_output[1])
        da_fit          = reshape_2d_ds(ymodel,invar,reshape_output[2],reshape_output[1])
        
        if regress_monthly: # Add additional "Month" variable at the end
        
            ref_da        = invar.isel(time=0).squeeze().expand_dims(dim={'mon':np.arange(1,13,1)},axis=-1)
            newshape      = list(reshape_output[2][:-1]) + [12,] # [Lon x Lat x Time]
            newshape_dims = reshape_output[1][:-1] + ['mon',]
        else:
            ref_da        = invar.isel(time=0).squeeze() #
            newshape      = reshape_output[2][:1] # Just Drop Time Dimension # [Lat x Lon]
            newshape_dims = reshape_output[1][:-1]
            
        da_pattern      = reshape_2d_ds(beta, ref_da, reshape_output[2][:-1],reshape_output[1][:-1]) # Drop time dim
        da_intercept    = reshape_2d_ds(intercept, ref_da, reshape_output[2][:-1],reshape_output[1][:-1]) # Drop time dim
        da_sig          = reshape_2d_ds(sigmask_out, ref_da,reshape_output[2][:-1],reshape_output[1][:-1]) # Drop time dim

    dsout = xr.merge([da_detrend,da_fit,da_pattern,da_intercept,da_sig],compat='override',join='override')
    
    return dsout


#%% 






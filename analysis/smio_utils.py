#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for analysis for the smio paper



quick_spectrum

Functions Included

add_coast_grid
calc_autocorr
calc_clim
calc_conflag
calc_confspec
calc_dof
calc_stds_sample*
calc_lag_corr_1d*
calc_lag_covar
calc_monvar
calc_pearsonconf
calc_T2
compute_sm_metrics

detrend_by_regression
detrend_dim
find_nan
fix_febstart
get_box_coords
hcbar
init_acplot
init_orthomap
label_sp
lp_butter
lon360to180_xr
lon180to360_xr
get_monstr

make_2d_ds
mcsampler
mcsample_stdev_metrics*

plot_box
printtime
quick_spectrum 
regress_2d
regress_ttest
reshape_2d_ds
sel_region_xr
tilebylag
xrdeseason
xrdetrend
year2mon
yo_taper
yo_spec
yo_speccl


Created on Fri Apr 10 15:21:12 2026

@author: gliu
"""

import numpy as np
import xarray as xr
import calendar as cal
import numpy.ma as ma
from scipy import signal,stats
from scipy import fft
from scipy.signal import butter, lfilter, freqz, filtfilt, detrend
import os
import time
import scipy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import scipy as sp
import pandas as pd
import datetime
import tqdm
import string

import cmocean as cmo

import cartopy.feature as cfeature

from tqdm import tqdm
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import matplotlib.path as mpath
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms

#%%


def add_coast_grid(ax,bbox=[-180,180,-90,90],proj=None,blabels=[1,0,0,1],ignore_error=False,
                   fill_color=None,line_color='k',grid_color='gray',c_zorder=1,
                   fix_lon=False,fix_lat=False,fontsize=12):
    """
    Add Coastlines, grid, and set extent for geoaxes
    
    Parameters
    ----------
    ax : matplotlib geoaxes
        Axes to plot on 
    bbox : [LonW,LonE,LatS,LatN], optional
        Bounding box for plotting. The default is [-180,180,-90,90].
    proj : cartopy.crs, optional
        Projection. The default is None.
    blabels : ARRAY of BOOL [Left, Right, Upper, Lower] or dict
        Lat/Lon Labels. Default is [1,0,0,1]
    ignore_error : BOOL
        Set to True to ignore error associated with gridlabeling
    fill_color : matplotlib color string
        Add continents with a given fill
    c_zorder : layering order of the continents
    
    Returns
    -------
    ax : matplotlib geoaxes
        Axes with setup
    """
    
    if type(blabels) == dict: # Convert dict to array
        blnew = [0,0,0,0]
        if blabels['left'] == 1:
            blnew[0] = 1
        if blabels['right'] == 1:
            blnew[1] = 1
        if blabels['upper'] == 1:
            blnew[2] = 1
        if blabels['lower'] == 1:
            blnew[3] = 1
        blabels=blnew
    
    if proj is None:
        proj = ccrs.PlateCarree()
        
    if fill_color is not None: # Shade the land
        ax.add_feature(cfeature.LAND,facecolor=fill_color,zorder=c_zorder)
    #ax.add_feature(cfeature.COASTLINE,color=line_color,lw=0.75,zorder=0)
    ax.coastlines(color=line_color,lw=0.75)
    ax.set_extent(bbox,proj)
    
    gl = ax.gridlines(crs=proj, draw_labels=True,
                  linewidth=0.75, color=grid_color, alpha=0.5, linestyle="dotted",
                  )
    
    # Remove the degree symbol
    if ignore_error:
        #print("Removing Degree Symbol")
        gl.xformatter = LongitudeFormatter(zero_direction_label=False,degree_symbol='')
        gl.yformatter = LatitudeFormatter(degree_symbol='')
        #gl.yformatter = LatitudeFormatter(degree_symbol='')
        gl.rotate_labels = False
    
    if fix_lon is not False:
        gl.xlocator = mticker.FixedLocator(fix_lon)
    if fix_lat is not False:
        gl.ylocator = mticker.FixedLocator(fix_lat)
    
    gl.left_labels      = blabels[0]
    gl.right_labels     = blabels[1]
    gl.top_labels       = blabels[2]
    gl.bottom_labels    = blabels[3]
    
    # Set Fontsize
    gl.xlabel_style = {'size':fontsize}
    gl.ylabel_style = {'size':fontsize}
    return ax

def calc_autocorr(sst,lags,basemonth,calc_conf=False,conf=0.95,tails=2,verbose=False,detrend=True):
    """
    Calculate autocorrelation for output of stochastic models
    
    Parameters
    ----------
    sst : DICT
        SST timeseries for each experiment
    lags : ARRAY
        Lags to calculate autocorrelation for
    basemonth : INT
        Month corresponding to lag 0 (ex. Jan=1) (NOT THE INDEX)
    calc_conf : BOOL
        Set to true to calculate confidence intervals
    conf : NUMERIC
        Confidence Level (default = 0.95)
    tails : INT
        Number of tails (1 or 2)
       
    Returns
    -------
    autocorr : DICT
        Autocorrelation stored in same order as sst
    """
    n = len(sst)
    autocorr = {}
    confs = {}
    for model in range(n):
        
        # Get the data
        tsmodel = sst[model]
        tsmodel = year2mon(tsmodel) # mon x year
        
        # Deseason (No Seasonal Cycle to Remove)
        tsmodel2 = tsmodel - np.mean(tsmodel,1)[:,None]
        
        # Detrend (Linear)
        if detrend:
            tsmodel2 = signal.detrend(tsmodel2,axis=1,type='linear')
        
        # Calculate the autocorrelation (set detrendopt to zero)
        autocorr[model] = calc_lagcovar(tsmodel2,tsmodel2,lags,basemonth,0,debug=verbose)
        
        confs[model] = calc_conflag(autocorr[model],conf,tails,tsmodel.shape[1])
    if calc_conf:
        return autocorr,confs
    return autocorr

def calc_clim(ts,dim,returnts=0,keepdims=False):
    """
    Given monthly timeseries with time in axis [dim], calculate the climatology...
    
    Returns: climavg,tsyrmon (if returnts=1)
    
    """
    tsshape = ts.shape
    ntime   = ts.shape[dim] 
    newshape =    tsshape[:dim:] +(int(ntime/12),12) + tsshape[dim+1::]
    
    tsyrmon = np.reshape(ts,newshape)
    climavg = np.nanmean(tsyrmon,axis=dim,keepdims=keepdims)
    
    if returnts==1:
        return climavg,tsyrmon
    else:
        return climavg

def calc_conflag(ac,conf,tails,n):
    """
    Calculate Confidence Intervals for autocorrelation function

    Parameters
    ----------
    ac : ARRAY [nlags,npts]
        Autocorrelation values by lag
    conf : NUMERIC
        Confidence level (ex. 0.95)
    tails : INT
        # of tails to consider
    n : INT
        Degrees of Freedom

    Returns
    -------
    cflags : ARRAY [nlags x 2 (upper/lower) x npts]
        Confidence interval for each lag

    """
    ND = False
    if len(ac.shape) > 1:
        ND = True
    
    if ND:
        nlags,npts = ac.shape
        cflags = np.zeros((nlags,2,npts)) # [Lag x Conf x Npts]
        
    else:
        nlags = len(ac)
        cflags = np.zeros((nlags,2)) # [Lag x Conf]
    
    for l in range(nlags):
        rhoin = ac[l,...]
        cfout = calc_pearsonconf(rhoin,conf,tails,n) # [conf x npts]
        cflags[l,...] = cfout
    return cflags

def calc_confspec(alpha,nu):
    """
    Based on code written by Tom Farrar for 12.805 @ MIT (see the func. confid).
    Copied from the 12.805 tbx.py on 2025.04.22
    
    Compute the upper and lower confidence limits for a chi-square variate.
    
    
    Parameters
    ----------
    alpha : numeric
        Significance value (For example, alpha=0.05 gives 95% confidence interval.)
    nu : numeric
        Number of degrees of freedom

    Returns
    -------
    lower: lower bound of confidence interval
    upper: upper bound of confidence interval
    
    """
    # requires:
    # from scipy import stats
    upperv=stats.chi2.isf(1-alpha/2,nu)
    lowerv=stats.chi2.isf(alpha/2,nu)
    lower=nu / lowerv
    upper=nu / upperv
    
    return (lower,upper)

def calc_dof(ts,ts1=None,calc_r1=True,ntotal=None,verbose=True,r1_in=None,r1_in_2=None):
    """
    Calculate effective degrees of freedom for autocorrelated timeseries.
    Assumes time is first dim, but can specify otherwise. Based on Eq. 31
    from Bretherton et al. 1998 (originally Bartlett 1935):
        
        N_eff = N * (1-r1*r2) / (1+r1*r2) 
        
    Inputs:
        ts          :: ARRAY [time] 1-D or 2-D Array, or lag 1 autocorrelation (r1) if calc_r1=False
        ts1         :: ARRAY [time] Another timeseries to correlate, or r1 if calc_r1=False
        calc_r1     :: BOOL    - Set to False if ts and ts1 are precalculated r1s
        ntotal      :: NUMERIC - Number of samples, must be given if calc_r1 is false (full DOF)
    Output:
        dof         :: Int Effective Degrees of Freedom
        
    """
    if calc_r1:
        if ntotal is None:
            n_tot = len(ts)
        else:
            n_tot = ntotal
    else:
        n_tot = len(ts)
    if verbose:
        print("Setting base DOF to %s" % str(n_tot))
    
    # Compute R1 for first timeseries
    if calc_r1:
        ts_base         = ts[:-1]
        ts_lag          = ts[1:]
        r1              = np.corrcoef(ts_base,ts_lag)[0,1]
    else:
        r1 = ts
    
    if r1_in is not None:
        print("Using provided r1 for timeseries 1")
        r1 = r1_in
    
    if np.any(r1<0):
        print("Warning, r1 is less than zero. Taking abs value!")
        r1 = np.abs(r1)
    
    if ts1 is None: # Square R1
        rho_in = r1**2
        
    else: # Compute R2 and compute product
        
        if calc_r1:
            ts1_base    = ts1[:-1]
            ts1_lag     = ts1[1:]
            r2          = np.corrcoef(ts1_base,ts1_lag)[0,1]
        else:
            r2          = ts1
            
        if r1_in_2 is not None:
            print("Using provided r1 for timeseries 2")
            r2 = r1_in_2
            
        if np.any(r2<0):
            print("Warning, r2 is less than zero. Taking abs value!")
            r2 = np.abs(r2)
    
        rho_in      = r1*r2
    
    # Compute DOF
    dof   = n_tot * (1-rho_in) / (1+rho_in)
    
    return dof

def calc_lag_corr_1d(var1, var2, lags):  # Can make 2d by mirroring calc_lag_covar_annn
    # Calculate the regression where
    # (+) lags indicate var1 lags  var2 (var 2 leads)
    # (-) lags indicate var1 leads var2 (var 1 leads)

    ntime = len(var1)
    betalag = []
    poslags = lags[lags >= 0]
    for l, lag in enumerate(poslags):
        varlag = var1[lag:]
        varbase = var2[:(ntime-lag)]

        # Calculate correlation
        # np.polyfit(varbase,varlag,1)[0]
        beta = sp.stats.linregress(varbase, varlag)[2]
        betalag.append(beta.item())

    neglags = lags[lags < 0]
    # Sort from least to greatest #.sort
    neglags_sort = np.sort(np.abs(neglags))
    betalead = []

    for l, lag in enumerate(neglags_sort):
        varlag = var2[lag:]  # Now Varlag is the base...
        varbase = var1[:(ntime-lag)]
        # Calculate correlation
        # beta = np.polyfit(varlag,varbase,1)[0]
        beta = sp.stats.linregress(varlag, varbase)[2]
        betalead.append(beta.item())

    # Append Together
    return np.concatenate([np.flip(np.array(betalead)), np.array(betalag)])

def calc_lagcovar(var1,var2,lags,basemonth,detrendopt,yr_mask=None,debug=True,
                  return_values=False,spearman=False):
    """
    Calculate lag-lead relationship between two monthly time series with the
    form [mon x yr]. Lag 0 is set by basemonth
    
    Correlation will be calculated for each lag in lags (lead indicate by
    negative lags)
    
    Set detrendopt to 1 for a linear detrend of each time series.
    
    
    Inputs:
        1) var1: Monthly timeseries for variable 1 [mon x year]
        2) var2: Monthly timeseries for variable 2 [mon x year]
        3) lags: lags and leads to include
        4) basemonth: lag 0 month
        5) detrendopt: 1 for linear detrend of both variables
        6) yr_mask : ARRAY of indices for selected years
        7) debug : Print check messages
        8) return_values [BOOL] : Return the lagged values and base values
    
    Outputs:
        1) corr_ts: lag-lead correlation values of size [lags]
        2) yr_count : print the count of years
        3) varbase : [yrs] Values of monthly anomalies for reference month
        4) varlags : [lag][yrs] Monthly anomalies for each lag month
    
    Dependencies:
        numpy as np
        scipy signal,stats
    
    """
    
    # Get total number of lags
    lagdim = len(lags)
    
    # Get timeseries length
    totyr = var1.shape[1]
    
    # Get total number of year crossings from lag
    endmonth = basemonth + lagdim-1
    nlagyr   = int(np.ceil(endmonth/12)) #  Ignore zero lag (-1)
    
    if debug:
        print("Lags spans %i mon (%i yrs) starting from mon %i" % (endmonth,nlagyr,basemonth))
        
    # Get Indices for each year
    if yr_mask is not None:
        # Drop any indices that are larger than the limit
        # nlagyr-1 accounts for the base year...
        # totyr-1 accounts for indexing
        yr_mask_clean = np.array([yr for yr in yr_mask if (yr+nlagyr-1) < totyr])
        
        if debug:
            n_drop = np.setdiff1d(yr_mask,yr_mask_clean)
            print("Dropped the following years: %s" % str(n_drop))
        
        yr_ids  = [] # Indices to 
        for yr in range(nlagyr):
            
            # Apply year-lag to index
            yr_ids.append(yr_mask_clean + yr)
    
    
    # Get lag and lead sizes (in years)
    leadsize = int(np.ceil(len(np.where(lags < 0)[0])/12))
    lagsize = int(np.ceil(len(np.where(lags > 0)[0])/12))
    
    # Detrend variables if option is set
    if detrendopt == 1:
        var1 = signal.detrend(var1,1,type='linear')
        var2 = signal.detrend(var2,1,type='linear')
    
    # Get base timeseries to perform the autocorrelation on
    if yr_mask is not None:
        varbase = var1[basemonth-1,yr_ids[0]] # Anomalies from starting year
    else: # Use old indexing approach
        base_ts = np.arange(0+leadsize,totyr-lagsize)
        varbase = var1[basemonth-1,base_ts]
        
    # Preallocate Variable to store correlations
    corr_ts = np.zeros(lagdim)
    
    # Set some counters
    nxtyr = 0
    addyr = 0
    modswitch = 0
    
    varlags = [] # Save for returning later
    for i in lags:

        lagm = (basemonth + i)%12
        
        if lagm == 0:
            lagm = 12
            addyr = 1         # Flag to add to nxtyr
            modswitch = i+1   # Add year on lag = modswitch
            
        if addyr == 1 and i == modswitch:
            if debug:
                print('adding year on '+ str(i))
            addyr = 0         # Reset counter
            nxtyr = nxtyr + 1 # Shift window forward
            
        # Index the other variable
        if yr_mask is not None:
            varlag = var2[lagm-1,yr_ids[nxtyr]]
            if debug:
                print("For lag %i (m=%i), first (last) indexed year is %i (%i) " % (i,lagm,yr_ids[nxtyr][0],yr_ids[nxtyr][-1]))
        else:
            lag_ts = np.arange(0+nxtyr,len(varbase)+nxtyr)
            varlag = var2[lagm-1,lag_ts]
            if debug:
                print("For lag %i (m=%i), lag_ts is between %i and %i" % (i,lagm,lag_ts[0],lag_ts[-1]))
            
        #varbase = varbase - varbase.mean()
        #varlag  = varlag - varlag.mean()
        #print("Lag %i Mean is %i ")
        
        # Calculate correlation
        if spearman == 1:
            corr_ts[i] = stats.spearmanr(varbase,varlag)[0]
            #corr_ts[i] = stats.kendalltau(varbase,varlag)[0]
        elif spearman == 2:
            corr_ts[i] = stats.kendalltau(varbase,varlag)[0]
        else:
            corr_ts[i] = stats.pearsonr(varbase,varlag)[0]
        varlags.append(varlag)
        
    if return_values:
        return corr_ts,varbase,varlags
    if yr_mask is not None:
        return corr_ts,len(yr_ids[-1]) # Return count of years as well
    return corr_ts

def calc_monvar(ts,dim=0):
    # NOTE/WARNING: Currently just works if time is in the first dimension
    # Copied from viz_synth_stochmod_combine
    # Compute Monthly Variance for a timeseries, ignoring all NaNs
    _,tsmyr = calc_clim(ts,dim,returnts=1)
    monvar  = np.nanvar(tsmyr,axis=0)
    if monvar.shape[0] != 12:
        print("Warning, this function only supports the case where time is in the first dim.")
        return monvar
    return monvar

def calc_stds_sample(aavgs):
    # Apply 10-year LP Filter to List of Timeseries and compute the st. dev.
    aavgs_lp = [lp_butter(aavg,120,6) for aavg in aavgs] # Calculate Low Pass Filter
    stds     = np.array([np.nanstd(ss) for ss in aavgs])      # Compute Standard Deviation
    stds_lp  = np.array([np.nanstd(ss) for ss in aavgs_lp])   # Compute LP-Filtered Standard Deviation
    vratio   = stds_lp/stds * 100 # Compute Ratio of Stdev.
    return aavgs_lp,stds,stds_lp,vratio


def calc_pearsonconf(rho,conf,tails,n):
    """
    rho   : pearson r [npts]
    conf  : confidence level
    tails : 1 or 2 tailed
    n     : Sample size
    """
    
    # Get z-critical
    alpha = (1-conf)/tails
    zcrit = stats.norm.ppf(1 - alpha)
    
    # Transform to z-space
    zprime = 0.5*np.log((1+rho)/(1-rho))
    
    # Calculate standard error
    SE     = 1/ np.sqrt(n-3)
    
    # Get Confidence
    z_lower = zprime-zcrit*SE
    z_upper = zprime+zcrit*SE
    
    # Convert back to r
    c_lower = np.tanh(z_lower)
    c_upper = np.tanh(z_upper)
    return c_lower,c_upper

def calc_T2(rho,axis=0,ds=False,verbose=False):
    """
    Calculate Decorrelation Timescale (DelSole 2001)
    Inputs:
    rho  : [ARRAY] Autocorrelation Function [lags x otherdims]
    axis : [INT], optional, Axis to sum along (default = 0)
    """
    # if ds:
    #     return (1+2*(rho**2).sum(axis))
    # if np.take(rho,0,axis=axis).squeeze() == 1: # (Take first lag)
    if np.any(rho == 1.):
        if verbose:
            print("Replacing %i values of Corr=1.0 with 0." % (np.sum(rho==1)))
        rho_in = np.where(rho == 1.,0,rho)
    else:
        rho_in = rho
        
    return (1+2*np.nansum(rho_in**2,axis=axis))

def compute_sm_metrics(ssts,
                    nsmooth=20,
                    pct=0.10,
                    opt=1,
                    dt=3600*24*30,
                    detrend_acf=True,
                    lags=np.arange(37)):
    """
    Given a list of 1-D timeseries for SST, loops through each one and computes the ACF for
    each basemonth, the Spectra, and the Monthly variance.

    Parameters
    ----------
    ssts : TYPE
        DESCRIPTION.
    nsmooth : INT or list of INT, optional
        Number of adjacent bands to smooth over. The default is 100.
    pct : NUMERIC, optional
        Percentage of spectra to taper. The default is 0.10.
    opt : INT, optional
        Detrending option for yo_spec (0=demean, 1=demean+detrend). The default is 1.
    dt : NUMERIC, optional
        Time step interval in sectonds. The default is 3600*24*30.
    lags : LIST of Int, optional
        Lags over which to compute the ACF. The default is np.arange(37).

    Returns
    -------
    outdict : Output dictionary with ACFs, Spectra Output, and Monthly Variance
    """
    
    nexps   = len(ssts)
    
    # 1. Compute the autocorrelation for each basemonth -----------------------
    acs_all = [] # [basemonth][experiment]
    for im in range(12):
        acout      = calc_autocorr(ssts,lags,im+1,detrend=detrend_acf) # scm function
        acout_proc = [acout[ii] for ii in range(nexps) ] # Convert from dict to list
        acs_all.append(acout_proc)
    
    # 2. Compute the spectra --------------------------------------------------
    spec_output = quick_spectrum(ssts,nsmooth,pct,opt=opt,dt=dt,return_dict=True) # scm function
    
    # 3. Compute the monthly variance -----------------------------------------
    monvars = [calc_monvar(s) for s in ssts]
    
    # 4. Save the output
    outdict = {
        "acfs"    : acs_all,
        "specs"   : spec_output['specs'],
        "freqs"   : spec_output['freqs'],
        "monvars" : monvars,
        "CCs"     : spec_output['CCs'],
        "dofs"    : spec_output['dofs'],
        "r1s"     : spec_output['r1s'],
        }
    return outdict


def get_monstr(nletters=3):
    """
    Get Array containing strings of first 3 letters of reach month
    """
    if nletters is None:
        return [cal.month_name[i][:] for i in np.arange(1,13,1)]
    else:
        return [cal.month_name[i][:nletters] for i in np.arange(1,13,1)]

def lon360to180_xr(ds,lonname='lon'):
    # Based on https://stackoverflow.com/questions/53345442/about-changing-longitude-array-from-0-360-to-180-to-180-with-python-xarray
    dsnew = ds.copy()
    dsnew.coords[lonname] = (ds.coords[lonname] + 180) % 360 - 180
    dsnew = dsnew.sortby(dsnew[lonname])
    return ds

def lon180to360_xr(ds,lonname='lon'):
    # Warning: This modifies things in place!
    dsnew = ds.copy()
    dsnew.coords[lonname] = (ds.coords[lonname] + 360) % 360
    # lon180 = ds.coords[lonname]
    # lon360 = xr.where(lon180 <0,lon180+360,lon180)
    # ds.coords[lonname] = lon360
    dsnew = dsnew.sortby(dsnew[lonname])
    return dsnew

def detrend_by_regression(invar,in_ts,regress_monthly=False,return_pattern_only=False):
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
        # Perform the regression (all months)
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
            newshape      = list(reshape_output[2][:-1]) + [12,] # [Lon x Lat x Mon]
            newshape_dims = reshape_output[1][:-1] + ['mon',]
        else:
            ref_da        = invar.isel(time=0).squeeze() #
            newshape      = reshape_output[2][:-1] # Just Drop Time Dimension # [Lat x Lon]
            newshape_dims = reshape_output[1][:-1]
            
        da_pattern      = reshape_2d_ds(beta, ref_da, newshape, newshape_dims) # Drop time dim
        da_intercept    = reshape_2d_ds(intercept, ref_da, newshape, newshape_dims) # Drop time dim
        da_sig          = reshape_2d_ds(sigmask_out, ref_da, newshape, newshape_dims) # Drop time dim
    
    if return_pattern_only: # Do not return detrended variable
        dsout = xr.merge([da_fit,da_pattern,da_intercept,da_sig],compat='override',join='override')
    else:
        dsout = xr.merge([da_detrend,da_fit,da_pattern,da_intercept,da_sig],compat='override',join='override')
    
    return dsout

def detrend_dim(invar,dim,return_dict=False,debug=False):
    
    """
    Detrends n-dimensional variable [invar] at each point along axis [dim].
    Performs appropriate reshaping and NaN removal, and returns
    variable in the same shape+order. Assumes equal spacing along [dim] for 
    detrending
    
    Also outputs linear model and coefficients.
    
    Dependencies: 
        numpy as np
        find_nan (function)
        regress_2d (function)
    
    Inputs:
        1) invar: variable to detrend
        2) dim: dimension of axis to detrend along
        
    Outputs:
        1) dtvar: detrended variable
        2) linmod: computed trend at each point
        3) beta: regression coefficient (slope) at each point
        4) interept: y intercept at each point
    """
    
    # Reshape variable
    varshape = invar.shape
    
    # Reshape to move time to first dim
    newshape = np.hstack([dim,np.arange(0,dim,1),np.arange(dim+1,len(varshape),1)])
    newvar = np.transpose(invar,newshape)
    
    # Combine all other dims and reshape to [time x otherdims]
    tdim        = newvar.shape[0]
    if len(varshape) <= 1:
        print("Warning, function will not work for 1-D arrays...")
    otherdims   = newvar.shape[1::]
    proddims    = np.prod(otherdims)
    newvar      = np.reshape(newvar,(tdim,proddims))

    
    # Find non nan points
    varok,knan,okpts = find_nan(newvar,0)
    
    # Ordinary Least Squares Regression
    tper    = np.arange(0,tdim)
    m,b     = regress_2d(tper,varok)
    
    # Squeeze things? (temp fix, need to check)
    m       = m.squeeze()
    b       = b.squeeze()
    
    # Detrend [Space,1][1,Time]
    ymod = (m[:,None] * tper + b[:,None]).T
    dtvarok = varok - ymod
    
    # Replace into variable of original size
    dtvar  = np.zeros(newvar.shape) * np.nan
    linmod = np.copy(dtvar)
    beta   = np.zeros(okpts.shape) * np.nan
    intercept = np.copy(beta)
    
    dtvar[:,okpts]      = dtvarok
    linmod[:,okpts]     = ymod
    beta[okpts]         = m
    intercept[okpts]    = b
    
    # Reshape to original size
    dtvar  = np.reshape(dtvar,((tdim,)+otherdims))
    linmod = np.reshape(linmod,((tdim,)+otherdims))
    beta = np.reshape(beta,(otherdims))
    intercept = np.reshape(beta,(otherdims))
    
    # Tranpose to original order
    oldshape = [dtvar.shape.index(x) for x in varshape]
    dtvar = np.transpose(dtvar,oldshape)
    linmod = np.transpose(linmod,oldshape)
    
    # Debug Plot
    if debug:
        klon = 33
        klat = 33
        
        mvmean = lambda x,w:  np.convolve(x, np.ones(w), 'same') / w
        w     = 120
        raw_y = invar[:,klat,klon]
        dt_y  = dtvar[:,klat,klon]
        x     = np.arange(0,len(dt_y))
        
        fig,ax = plt.subplots(1,1)
        ax.scatter(x,mvmean(raw_y,w),s=1.5,label="raw")
        ax.scatter(x,mvmean(dt_y,w),s=1.5,label='detrend')
        ax.plot(x,linmod[:,klat,klon],label="fit")
        ax.legend()
        plt.show()
    
    if return_dict:
        outdict = dict(detrended_var=dtvar,linearmodel=linmod,beta=beta,intercept=intercept)
        return outdict
    return dtvar,linmod,beta,intercept


def find_nan(data,dim,val=None,return_dict=False,verbose=True):
    """
    For a 2D array, remove any point if there is a nan in dimension [dim].
    
    Inputs:
        1) data        : 2d array, which will be summed along last dimension
        2) dim         : dimension to sum along. 0 or 1.
        3) val         : value to search for (default is NaN)
        4) return_dict : Set to True to return dictionary with clearer arguments...
    Outputs:
        1) okdata : data with nan points removed
        2) knan   : boolean array with indices of nan points
        3) okpts  : indices for non-nan points
    """
    
    # Sum along select dimension
    if len(data.shape) > 1:
        datasum = np.sum(data,axis=dim)
    else:
        datasum = data.copy()
    
    # Find non nan pts
    if val is None:
        knan  = np.isnan(datasum)
    else:
        knan  = (datasum == val)
    okpts = np.invert(knan)
    
    if len(data.shape) > 1:
        if dim == 0:
            okdata = data[:,okpts]
            clean_dim = 1
        elif dim == 1:    
            okdata = data[okpts,:]
            clean_dim = 0
    else:
        okdata = data[okpts]
    if verbose:
        print("Found %i NaN Points along axis %i." % (data.shape[clean_dim] - okdata.shape[clean_dim],clean_dim))
    if return_dict: # Return dictionary with clearer arguments
        nandict = {"cleaned_data" : okdata,
                   "nan_indices"  : knan,
                   "ok_indices"   : okpts,
                   }
        return nandict
    return okdata,knan,okpts


def get_box_coords(bbox,dx=None,dy=None):
    # Get coordinates of box for plotting an ortho map/polygon
    # Given [Westbound EastBound Southbound Northbound]
    # Returns xcoords and ycoords of path drawn from:
    # Lower Left, counterclockwise around and back.
    
    if dx is None:
        dx = np.linspace(bbox[0],bbox[1],5)
        dx = dx[1] - dx[0]
    if dy is None:
        dy = np.linspace(bbox[2],bbox[3],5)
        dy = dy[1] - dy[0]
    
    # Lower Edge (Bot. Left --> Bot. Right)
    lower_x = np.arange(bbox[0],bbox[1]+dx,dx) # x-coord
    nx = len(lower_x) 
    lower_y = [bbox[2],]*nx # y-coord
    
    # Right Edge (Bot. Right ^^^ Top Right)
    right_y = np.arange(bbox[2],bbox[3]+dy,dy)
    ny = len(right_y)
    right_x = [bbox[1],]*ny
    
    # Upper Edge (Top Left <-- Top Right)
    upper_x = np.flip(lower_x)
    upper_y = [bbox[3],]*nx
    
    # Left Edge (Bot. Left vvv Top Left)
    left_y  = np.flip(right_y)
    left_x  = [bbox[0],]*ny
    
    x_coords = np.hstack([lower_x,right_x,upper_x,left_x])
    y_coords = np.hstack([lower_y,right_y,upper_y,left_y])
    
    return x_coords,y_coords

def fix_febstart(ds):
    # Copied from preproc_CESM.py on 2022.11.15
    if ds.time.values[0].month != 1:
        print("Warning, first month is %s. Fixing."% ds.time.values[0])
        # Get starting year, must be "YYYY"
        startyr = str(ds.time.values[0].year)
        while len(startyr) < 4:
            startyr = '0' + startyr
        nmon = ds.time.shape[0] # Get number of months
        # Corrected Time
        correctedtime = xr.cftime_range(start=startyr,periods=nmon,freq="MS",calendar="noleap")
        ds = ds.assign_coords(time=correctedtime) 
    return ds

def hcbar(mpl_obj,ax=None,fig=None,fraction=0.035,pad=.01,fontsize=12,rotation=0):
    """
    Make quick horizontal colorbar. Arguments are same as ax.colorbar()
    
    """
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()
    cb = plt.colorbar(mpl_obj,ax=ax,fraction=fraction,
                      pad=pad,orientation='horizontal',)
    cb.ax.tick_params(labelsize=fontsize,rotation=rotation)
    return cb


def init_acplot(kmonth,xticks,lags,ax=None,title=None,loopvar=None,
                usegrid=True,tickfreq=None,fsz_axis=14,fsz_ticks=12,fsz_title=18,vlines=None):
    """
    Function to initialize autocorrelation plot with months on top,
    lat on the bottom
    
    Parameters
    ----------
    kmonth : INT
        Index of Month corresponding to lag=0.
    xticks : ARRAY
        Lags that will be shown
    lags : ARRAY
        Lags to visulize
    ax : matplotlib object, optional
        Axis to plot on
    title : STR, optional
        Title of plot. The default is "SST Autocorrelation, Lag 0 = Month.
    loopvar: ARRAY [12,], optional
        Monthly variable to tile and plot in the background
    vlines: indices of months to put vertical values at
    
    Returns
    -------
    ax,ax2, and ax3 if loopvar is not None : matplotlib object
        Axis with plot

    """
    if ax is None:
        
        ax  = plt.gca()
    
    # Tile Months for plotting
    mons3     = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
    mons3tile = np.tile(np.array(mons3),int(np.floor(len(lags)/12))) 
    mons3tile = np.concatenate([np.roll(mons3tile,-kmonth),[mons3[kmonth]]])
    
    # Set up second axis
    ax2 = ax.twiny()
    ax2.set_xticks(xticks)#,size=fsz_ticks)
    ax2.set_xticklabels(mons3tile[xticks], rotation = 45,fontsize=fsz_ticks)
    ax2.set_axisbelow(True)
    ax2.grid(zorder=0,alpha=0)
    ax2.set_xlim(xticks[[0,-1]])
    
    # Plot second variable if option is set
    if loopvar is not None:
        ax3 = ax.twinx()
        loopvar = tilebylag(kmonth,loopvar,lags)
        ax3.plot(lags,loopvar,color='gray',linestyle='dashed')
        ax3.tick_params(axis='y',labelcolor='gray',fontsize=fsz_ticks)
        ax3.grid(False)
    
    ax.set_xticks(xticks)
    ax.tick_params(labelsize=fsz_ticks)
    ax.set_xlim([xticks[0],xticks[-1]])
    if title is None:
        ax.set_title("SST Autocorrelation, Lag 0 = %s" % (mons3[kmonth]),fontsize=fsz_title)
    else:
        ax.set_title(title)
    ax.set_xlabel("Lags (Months)",fontsize=fsz_axis)
    ax.set_ylabel("Correlation",fontsize=fsz_axis)
    if usegrid:
        ax.grid(True,linestyle='dotted')
    #plt.tight_layout()
    
    # Adjust ticks if option is set
    if tickfreq is not None:
        lbl_new_mon = []
        lbl_new     = []
        for i in range(len(xticks)):
            
            ilag = xticks[i]
            
            if i%tickfreq == 0:
                lbl_new_mon.append(mons3tile[ilag])
                lbl_new.append(lags[ilag])
            else:
                lbl_new_mon.append("")
                lbl_new.append("")
        ax.set_xticklabels(lbl_new,fontsize=fsz_ticks)
        ax2.set_xticklabels(lbl_new_mon,fontsize=fsz_ticks)
    
    # Add some vertical lines
    if vlines is not None:
        vline_mons = [mons3[mm] for mm in vlines]
        for l,lag in enumerate(lags):
            lbl = mons3tile[l]
            if lbl in vline_mons:
                ax.axvline([l],lw=0.75,c="k",label="")
    
    if loopvar is not None:
        return ax,ax2,ax3
    return ax,ax2

def init_orthomap(nrow,ncol,bboxplot,centlon=-40,centlat=35,precision=40,
                  dx=10,dy=5,
                  frame_lw=2,frame_col="k",
                  figsize=(8,4.5),constrained_layout=True,ax=None):
    
    # Intiailize Ortograpphic map over North Atlantic.
    # Based on : https://stackoverflow.com/questions/74124975/cartopy-fancy-box
    # The default lat/lon projection
    noProj = ccrs.PlateCarree(central_longitude=0)
    
    # Set Orthographic Projection
    myProj = ccrs.Orthographic(central_longitude=centlon, central_latitude=centlat)
    myProj._threshold = myProj._threshold/precision  #for higher precision plot
    
    # Initialize Figure
    fig,axs = plt.subplots(nrow,ncol,figsize=figsize,subplot_kw={'projection': myProj},
                          constrained_layout=constrained_layout)
    
    # Get Line Coordinates
    xp,yp  = get_box_coords(bboxplot,dx=dx,dy=dy)
    
    # Draw the line
    if nrow ==1 and ncol ==1:
        #print("Nd Axis")
        axs = [axs,]
        ndaxis=False
    else:
        orishape = axs.shape
        axs      = axs.flatten()
        ndaxis   = True
    for ax in axs:
        [ax_hdl] = ax.plot(xp,yp,
            color=frame_col, linewidth=frame_lw,
            transform=noProj)
        
        # Make a polygon and crop
        tx_path                = ax_hdl._get_transformed_path()
        path_in_data_coords, _ = tx_path.get_transformed_path_and_affine()
        polygon1s              = mpath.Path( path_in_data_coords.vertices)
        ax.set_boundary(polygon1s) # masks-out unwanted part of the plot
        
    if ndaxis is False:
        axs = axs[0] # Return just the axis
    else:
        axs = axs.reshape(orishape)
    mapdict={
        'noProj'     : noProj,
        'myProj'     : myProj,
        'line_coords': (xp,yp),
        'polygon'    : polygon1s,
        }
    return fig,axs,mapdict


def label_sp(sp_id,case='upper',inside=True,ax=None,fig=None,x=0.0,y=1.0,
             fontsize=12,fontfamily='sans-serif',alpha=1,labelstyle=None,
             usenumber=False,fontcolor='k',weight='normal'):
    """
    Add alphabetical labels to subplots
    from: https://matplotlib.org/stable/gallery/text_labels_and_annotations/label_subplots.html
    
    Inputs:
        sp_id [int]                - Subplot Index for alphabet (0=A, 1=B, ...)
        case  ['upper' or 'lower'] - Case of subplot label
        inside [BOOL]              - True to plot inside, False to plot outside
        ax    [mpl.axes]           - axes to plot on. default=current axes
        fig   [mpl.fig]            - figure to scale
        x     [numeric]            - x position relative to upper left
        y     [numeric]            - y position relative to upper left
        fontsize [int]             - font size
        fontfamily [str]           - font family
        alpha [numeric]            - transparency of textbox for inside label
        labelstyle [str]           - labeling style, use %s to indicate string location "%s)"
        usenumber [bool]           - Set to true to use numeric labels (using sp_id)
    """
    
    if usenumber:
        label = str(sp_id)
    else:
        if case == 'upper':
            label = list(string.ascii_uppercase)[sp_id]
        elif case == 'lower':
            label = list(string.ascii_lowercase)[sp_id]
        else:
            print("case must be 'upper' or 'lower'!" )
    
    if labelstyle is None:
        labelstyle="%s)"
    label= labelstyle % (label)
    
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()
    
    if inside:
        trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
        ax.text(x, y, label, transform=ax.transAxes + trans,
                fontsize=fontsize, verticalalignment='top',
                bbox=dict(facecolor='1', edgecolor='none', pad=3.0,alpha=alpha),
                color=fontcolor,weight=weight)
    else:
        trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
        ax.text(x, y, label, transform=ax.transAxes + trans,
                fontsize=fontsize, va='bottom',
                color=fontcolor,weight=weight)
    return ax

def lp_butter(varmon,cutofftime,order,btype='lowpass'):
    """
    Design and apply a low-pass filter (butterworth)

    Parameters
    ----------
    varmon : 
        Input variable to filter (monthly resolution)
    cutofftime : INT
        Cutoff value in months
    order : INT
        Order of the butterworth filter
    btype : STR
        Type of filter {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}

    Returns
    -------
    varfilt : ARRAY [time,lat,lon]
        Filtered variable
    
    """
    # Input variable is assumed to be monthy with the following dimensions:
    flag1d=False
    if len(varmon.shape) > 1:
        nmon,nlat,nlon = varmon.shape
    else:
        flag1d = True
        nmon = varmon.shape[0]
    
    # Design Butterworth Lowpass Filter
    filtfreq = nmon/cutofftime
    nyquist  = nmon/2
    cutoff   = filtfreq/nyquist
    b,a      = butter(order,cutoff,btype=btype)
    
    # Reshape input
    if flag1d is False: # For 3d inputs, loop thru each point
        varmon = varmon.reshape(nmon,nlat*nlon)
        # Loop
        varfilt = np.zeros((nmon,nlat*nlon)) * np.nan
        for i in range(nlon*nlat):
            varfilt[:,i] = filtfilt(b,a,varmon[:,i])
        varfilt=varfilt.reshape(nmon,nlat,nlon)
    else: # 1d input
        varfilt = filtfilt(b,a,varmon)
    return varfilt

def make_2d_ds(ds,keepdim='time'):
    
    
    # Get List of Dims, move time to front
    oldshape      = ds.shape
    dimnames      = ds.dims
    otherdims     = list(dimnames)#.remove(keepdim)
    otherdims.remove(keepdim)
    newdims        = otherdims  + [keepdim,] 
    dstranspose    = ds.transpose(*newdims)
    
    # Convert to 3D intput where time is last [1 x otherdims x time]
    dsarr          = dstranspose.data
    oldshape_trans = dsarr.shape
    ntime          = oldshape_trans[-1]
    notherdims     = int(np.array(oldshape_trans[:-1]).prod())
    dsarr          = dsarr.reshape(1,notherdims,ntime)
    coords_rs      = dict(lon=[1,],lat=np.arange(notherdims),time=ds.time)
    dsreshape      = xr.DataArray(dsarr,dims=coords_rs,coords=coords_rs,name=ds.name)
    return dsreshape,newdims,oldshape_trans

def mcsampler(ts_full,sample_len,mciter,preserve_month=True,scramble_year=False,target_timeseries=None):
    # Taken from ensobase.utils on 2025.10.30
    # Given a monthly timeseries [time] and sample length (int), take [mciter] # of random samples.
    # if preserve_month = True, preserve the 12-month sequence as a chunk
    # if scramble_year = True, randomize years that you are selecting from (do not preserve year order)
    # if target_timeseries is not None: also select random samples from list of timeseries (must be same length as ts_full)
    
    # Function Start
    ntime_full        = len(ts_full)
    
    # 1 -- month agnostic (subsample sample length, who cares when)
    if not preserve_month:
        
        print("Month with not be preserved.")
        istarts    = np.arange(ntime_full-sample_len)
        
        sample_ids = []
        samples    = []
        for mc in range(mciter):
            # ts_full[istarts[-1]:(istarts[-1]+sample_len)] Test last possible 
            iistart = np.random.choice(istarts)
            idsel   = np.arange(iistart,iistart+sample_len) 
            msample = ts_full[idsel]
            
            
            sample_ids.append(idsel)
            samples.append(msample) # [iter][sample]
            
        samples = np.array(samples) # [iter x sample]
        # Returns 
            
    elif preserve_month:
        # 2 -- month aware (must select starting points of January + maintain the chunk, preserving the month + year to year autocorrelation)
        if not scramble_year:
            
            # Only start on the year  (to preserve month sequence)
            istarts    = np.arange(0,ntime_full-sample_len,12)
            
            # -------------------- Same as Above
            sample_ids = []
            samples    = []
            for mc in range(mciter):
                # ts_full[istarts[-1]:(istarts[-1]+sample_len)] Test last possible 
                iistart = np.random.choice(istarts)
                idsel   = np.arange(iistart,iistart+sample_len) 
                msample = ts_full[idsel]
                
                sample_ids.append(idsel)
                samples.append(msample) # [var][iter][sample]
            samples = np.array(samples) # [var x iter x sample]
            # -------------------- 
            
        # 3 -- month aware, year scramble (randomly select the year of each month, but preserve each month)
        elif scramble_year: # Scrample Year and Month
            
            # Reshape to the year and month
            nyr_full        = int(ntime_full/12)
            ts_yrmon        = ts_full.reshape(nyr_full,12)
            ids_ori         = np.arange(ntime_full)
            ids_ori_yrmon   = ids_ori.reshape(ts_yrmon.shape)
            
            nyr_sample      = int(sample_len/12)
            sample_ids      = []
            samples         = []
            for mc in range(mciter): # For each loop
                
                # Get start years
                startyears = np.random.choice(np.arange(nyr_full),nyr_sample)
                # Select random years equal to the sample length and combine
                idsel      = ids_ori_yrmon[startyears,:].flatten() 
                # ------
                msample    = ts_full[idsel]
                sample_ids.append(idsel)
                samples.append(msample) # [var][iter][sample]
            samples = np.array(samples) # [var x iter x sample]
            # -----
    
    outdict = dict(sample_ids = sample_ids, samples=samples)    
    if target_timeseries is not None:
        
        sampled_timeseries = []
        for ts in target_timeseries:
            if len(ts) != len(ts_full):
                print("Warning... timeseries do not have the same length")
            randsamp = [ts[sample_ids[mc]] for mc in range(mciter)]
            randsamp = np.array(randsamp)
            sampled_timeseries.append(randsamp) # [var][iter x time]
        outdict['other_sampled_timeseries'] = sampled_timeseries
    
    return outdict

def mcsample_stdev_metrics(target_timeseries,sample_length,mciter):
    # Given a list of target timeseries
    # Sample [mciter] random samples of length [sample_length]
    # and compute the standard deviation (std), low-pass std, and monthly std 
    """
    Output
        1) mc_stds     : LIST [ex][mciter]     standard deviations for all samples
        2) mc_stds_lp  : LIST [ex][mciter]     standard deviations for 10-year low-passed samples
        3) mc_stds_mon : LIST [ex][mciter][12] monthly standard deviations for all samples
    
    """
    
    nexp = len(target_timeseries)
    
    mc_stds         = []
    mc_stds_lp      = []
    mc_stds_mon     = []
    
    for ex in tqdm(range(nexp)):
        
        # Take Samples and compute low-pass filter
        timeseries_in   = target_timeseries[ex]
        mcdict          = mcsampler(timeseries_in,sample_length,mciter)
        samples         = [mcdict['samples'][ii,:] for ii in range(mciter)]
        samples_lp      = [lp_butter(ts,120,6) for ts in samples]
        
        # Compute Monthly Stdev
        monstds_mc      = mcdict['samples'].reshape(mciter,int(sample_length/12),12).std(1)
        mc_stds_mon.append(monstds_mc)
        
        # Compute Standard Deviation
        mc_stds.append( np.nanstd(np.array(samples),1) )
        
        # Compute Low-pass Standard Deviation
        mc_stds_lp.append( np.nanstd(np.array(samples_lp),1) )
    
    return mc_stds,mc_stds_lp,mc_stds_mon

def plot_box(bbox,ax=None,return_line=False,leglab="Bounding Box",
             color='k',linestyle='solid',linewidth=1,proj=ccrs.PlateCarree()):
    
    """
    Plot bounding box
    Inputs:
        1) bbox [1D-ARRAY] [lonW,lonE,latS,latN]
        Optional Arguments...
        2) ax           [axis] axis to plot onto
        3) return_line  [Bool] return line object for legend labeling
        4) leglabel     [str]  Label for legend
        5) color        [str]  Line Color, default = black
        6) linestyle    [str]  Line style, default = solid
        7) linewidth    [#]    Line width, default = 1  
    
    
    """
    #if lon360 is False:
    for i in [0,1]:
        if bbox[i] > 180:
            bbox[i] -= 360
            
    
    if ax is None:
        ax = plt.gca()
    
    if bbox[0] > bbox[1]:
        bbox[1] += 360
        
    
    # Plot North Boundary
    # if bbox[0] > bbox[1]: # Crossing Date Line, use 360
    #     ax.plot([bbox[0],bbox[1]+360],[bbox[3],bbox[3]],color=color,ls=linestyle,lw=linewidth,label='_nolegend_',transform=proj)
    # else:  
    ax.plot([bbox[0],bbox[1]],[bbox[3],bbox[3]],color=color,ls=linestyle,lw=linewidth,label='_nolegend_',transform=proj)
    # Plot East Boundary
    ax.plot([bbox[1],bbox[1]],[bbox[3],bbox[2]],color=color,ls=linestyle,lw=linewidth,label='_nolegend_',transform=proj)
    # Plot South Boundary
    # if bbox[0] > bbox[1]: # Crossing Date Line, use 360 
    #     ax.plot([bbox[1]+360,bbox[0]],[bbox[2],bbox[2]],color=color,ls=linestyle,lw=linewidth,label='_nolegend_',transform=proj)
    # else:
    ax.plot([bbox[1],bbox[0]],[bbox[2],bbox[2]],color=color,ls=linestyle,lw=linewidth,label='_nolegend_',transform=proj)
    # Plot West Boundary
    ax.plot([bbox[0],bbox[0]],[bbox[2],bbox[3]],color=color,ls=linestyle,lw=linewidth,label='_nolegend_',transform=proj)
    
    if return_line == True:
        linesample =  ax.plot([bbox[0],bbox[0]],[bbox[2],bbox[3]],color=color,ls=linestyle,lw=linewidth,label=leglab,transform=proj)
        return ax,linesample
    return ax


def printtime(st,print_str="Completed"):
    # Given start time, print the elapsed time in seconds
    print("%s in %.2fs" % (print_str,time.time()-st))
    
def quick_spectrum(sst,nsmooth,pct,
                   opt=1,dt=None,clvl=[.95],verbose=False,return_dict=False,dim=None,make_arr=False):
    """
    Quick spectral estimate of an array of timeseries

    Parameters
    ----------
    sst : ARRAY
        Array containing timeseries to look thru [[ts1],[ts2],...]
    nsmooth : INT
        Number of bands to smooth over 
    pct : Numeric
        Percent to taper
    opt : TYPE, optional
        Smoothing option
    dt : INT, optional
        Time Interval. The default is 3600*24*30.
    clvl : ARRAY , optional
        Array of Confidence levels. The default is [.95].
        
    dim : INT, optional
        Specify dimension to loop computation over

    Returns
    -------
    specs : ARRAY
        Array containing spectrum for each input series
    freqs : ARRAY
        Corresponding frequencies for each input
    CCs : ARRAY
        Confidence intervals for each input
    dofs : ARRAY
        Degrees of freedom for each input
    r1s : ARRAY
        AR1 parameter used to estimate CC

    """
    
    # -----------------------------------------------------------------
    # Set interval of time series (assumes monthly by default)
    # -----------------------------------------------------------------
    if dt is None:
        dt = np.ones(len(sst)) * 3600*24*30
    else:
        dt = np.ones(len(sst)) * dt
    
    # -----------------------------------------------------------------
    # Calculate and make individual plots for stochastic model output
    # -----------------------------------------------------------------
    #specparams  = []
    specs = []
    freqs = []
    CCs = []
    dofs = []
    r1s = []
    if dim is None:
        n_ts = len(sst)
    else:
        n_ts = sst.shape[dim]
    
    for i in range(n_ts):
        if dim is None:
            sstin = sst[i]
        else:
            #sstin = np.take_along_axis(sst,i,dim)
            sstin = np.take(sst,i,dim)
        dt_in = dt[i]
        
        # Calculate Spectrum
        if isinstance(nsmooth,int):
            sps = yo_spec(sstin,opt,nsmooth,pct,debug=False,verbose=verbose)
        else:
            sps = yo_spec(sstin,opt,nsmooth[i],pct,debug=False,verbose=verbose)
        
        # Save spectrum and frequency, convert to 1/sec
        P,freq,dof,r1=sps
        specs.append(P*dt_in)
        freqs.append(freq/dt_in)
        dofs.append(dof)
        r1s.append(r1)
        
        # Calculate Confidence Interval
        CC = yo_speccl(freq,P,dof,r1,clvl)
        CCs.append(CC*dt_in)
    
    if make_arr:
        arrsout = [specs,freqs,CCs,dofs,r1s]
        arrsout = [np.array(a) for a in arrsout]
        specs,freqs,CCs,dofs,r1s = arrsout
    
    if return_dict:
        output_dict = {
            "specs":specs,
            "freqs":freqs,
            "CCs"  : CCs,
            "dofs" :dofs,
            "r1s"  :r1s,
            }
        return output_dict
    return specs,freqs,CCs,dofs,r1s
    
def regress_2d(A,B,nanwarn=1,verbose=True):
    """
    Regresses A (independent variable) onto B (dependent variable), where
    either A or B can be a timeseries [N-dimensions] or a space x time matrix 
    [N x M]. Script automatically detects this and permutes to allow for matrix
    multiplication.
    Note that if A and B are of the same size, assumes axis 1 of A will be regressed to axis 0 of B
    
    Returns the slope (beta) for each point, array of size [M]
    
    
    """
    
    # Determine if A or B is 2D and find anomalies
    bothND = False # By default, assume both A and B are not 2-D.
    # Note: need to rewrite function such that this wont be a concern...
    
    # Accounting for the fact that I dont check for equal dimensions below..
    #B = B.squeeze()
    #A = A.squeeze() Commented out below because I still need to fix some things
    # Compute using nan functions (slower)
    if np.any(np.isnan(A)) or np.any(np.isnan(B)):
        if nanwarn == 1:
            print("NaN Values Detected...")
        
        # 2D Matrix is in A [MxN]
        if len(A.shape) > len(B.shape):
            
            # Tranpose A so that A = [MxN]
            if A.shape[1] != B.shape[0]:
                A = A.T
            
            # Set axis for summing/averaging
            a_axis = 1
            b_axis = 0
            
            # Compute anomalies along appropriate axis
            Aanom = A - np.nanmean(A,axis=a_axis)[:,None]
            Banom = B - np.nanmean(B,axis=b_axis)
            
        # 2D matrix is B [N x M]
        elif len(A.shape) < len(B.shape):
            
            # Tranpose B so that it is [N x M]
            if B.shape[0] != A.shape[0]:
                B = B.T
            
            # Set axis for summing/averaging
            a_axis = 0
            b_axis = 0
            
            # Compute anomalies along appropriate axis        
            Aanom = A - np.nanmean(A,axis=a_axis)
            Banom = B - np.nanmean(B,axis=b_axis)[None,:]
        

        # A is [P x N], B is [N x M]
        elif len(A.shape) == len(B.shape):
            if verbose:
                print("Note, both A and B are 2-D...")
            bothND = True
            if A.shape[1] != B.shape[0]:
                print("WARNING, Dimensions not matching...")
                print("A is %s, B is %s" % (str(A.shape),str(B.shape)))
                print("Detecting common dimension")
                # Get intersecting indices 
                intersect, ind_a, ind_b = np.intersect1d(A.shape,B.shape, return_indices=True)
                if ind_a[0] == 0: # A is [N x P]
                    A = A.T # Transpose to [P x N]
                if ind_b[0] == 1: # B is [M x N]
                    B = B.T # Transpose to [N x M]
                print("New dims: A is %s, B is %s" % (str(A.shape),str(B.shape)))
                
            # Set axis for summing/averaging
            a_axis = 1 # Assumes dim 1 of A will be regressed to dim 0 of b
            b_axis = 0
            
            # Compute anomalies along appropriate axis        
            Aanom = A - np.nanmean(A,axis=a_axis,keepdims=True)#[:,None] # Anomalize w.r.t. dim 1 of A
            Banom = B - np.nanmean(B,axis=b_axis,keepdims=True)# # Anonalize w.r.t. dim 0 of B
            
        # Calculate denominator, summing over N
        Aanom2 = np.power(Aanom,2)
        denom  = np.nansum(Aanom2,axis=a_axis,keepdims=True)     # Sum along dim 1 of A (lets say this is time)
        
        # Calculate Beta
        #if 
        if len(denom.shape)==1 or not bothND: # same as both not ND
            print("Adding singleton dimension to denom")
            denom = denom[:,None]
        beta = Aanom @ Banom / denom#[:,None] # Denom is [A[mode,time]@ B[time x space]], output is [mode x pts]
        
        b = (np.nansum(B,axis=b_axis,keepdims=True) - beta * np.nansum(A,axis=a_axis,keepdims=True))/A.shape[a_axis]
        # b is [mode x pts] [or P x M]
            
    else:
        # 2D Matrix is in A [MxN]
        if len(A.shape) > len(B.shape):
            
            # Tranpose A so that A = [MxN]
            if A.shape[1] != B.shape[0]:
                A = A.T
                
            a_axis = 1
            b_axis = 0
            
            # Compute anomalies along appropriate axis
            Aanom = A - np.mean(A,axis=a_axis)[:,None]
            Banom = B - np.mean(B,axis=b_axis)
            
        # 2D matrix is B [N x M]
        elif len(A.shape) < len(B.shape):
            
            # Tranpose B so that it is [N x M]
            if B.shape[0] != A.shape[0]:
                B = B.T
            
            # Set axis for summing/averaging
            a_axis = 0
            b_axis = 0
            
            # Compute anomalies along appropriate axis        
            Aanom = A - np.mean(A,axis=a_axis)
            Banom = B - np.mean(B,axis=b_axis)[None,:]
            
        # A is [P x N], B is [N x M]
        elif len(A.shape) == len(B.shape):
            if verbose:
                print("Note, both A and B are 2-D...")
            bothND = True
            if A.shape[1] != B.shape[0]:
                print("WARNING, Dimensions not matching...")
                print("A is %s, B is %s" % (str(A.shape),str(B.shape)))
                print("Detecting common dimension")
                # Get intersecting indices 
                intersect, ind_a, ind_b = np.intersect1d(A.shape,B.shape, return_indices=True)
                if ind_a[0] == 0: # A is [N x P]
                    A = A.T # Transpose to [P x N]
                if ind_b[0] == 1: # B is [M x N]
                    B = B.T # Transpose to [N x M]
                print("New dims: A is %s, B is %s" % (str(A.shape),str(B.shape)))
            
            # Set axis for summing/averaging
            a_axis = 1
            b_axis = 0
            
            # Compute anomalies along appropriate axis        
            Aanom = A - np.mean(A,axis=a_axis)[:,None]
            Banom = B - np.mean(B,axis=b_axis)[None,:]

        # Calculate denominator, summing over N
        Aanom2 = np.power(Aanom,2)
        denom  = np.sum(Aanom2,axis=a_axis,keepdims=True)
        if not bothND:
            
            denom = denom[:,None] # Broadcast
            
        # Calculate Beta
        beta = Aanom @ Banom / denom
            
        if bothND:
            b = (np.sum(B,axis=b_axis)[None,:] - beta * np.sum(A,axis=a_axis)[:,None])/A.shape[a_axis]
        else:
            b = (np.sum(B,axis=b_axis) - beta * np.sum(A,axis=a_axis))/A.shape[a_axis]
    
    return beta,b


def regress_ttest(in_var,in_ts,dof=None,p=0.05,tails=2,verbose=True):
    """
    Given a timeseries (in_ts) and variable (in_var), compute regression
    coefficients and perform t-test to get significance
    Note: only tested for single value DOF, need to check for map of dofs...
    h0: regression coeffs are significantly different from zero
    
    Inputs:
    -------
    invar (ARRAY: [Lon x Lat x Time])   : Input pattern to regress
    in_ts (ARRAY: [time])               : Timeseries to regress to
    dof   (NUMERIC)                     : Degrees of Freedom to use. Defaults to nt-2
    p     (NUMERIC)                     : p-value for significance testing; Default: 0.05
    tails (INT)                         : # of Tails for t-test (1 or 2); Default: 2
    
    Outputs: (all (ARRAY: [Lon x Lat] ezcept t_critval)
    --------
    regression_coeff : Map of Regression Coefficients
    intercept        : Map of Intercepts
    SSE              : Squared Sum of Errors
    se               : Residual Standard Error
    t_statistic      : T-statistic at each point
    t_critval        : Critical T-value
    sigmask          : Mask where t_statistic > t_critval
    
    """
    
    # Step (1), get needed dimensions
    nt          = in_ts.shape[0]
    nlon,nlat,_ = in_var.shape # Assume [lon x lat x time]
    invar_rs    = in_var.reshape(nlon*nlat,nt)
    
    # Step (2), Remove NaNs
    nandict     = find_nan(invar_rs,1,return_dict=True,verbose=verbose) # Sum along time in 1
    invar_rs    = nandict['cleaned_data']
    
    # Define function to replace NaN
    def replace(x):
        outvar = np.zeros((nlon*nlat))
        outvar[nandict['ok_indices']] = x
        return outvar.reshape(nlon,nlat)
    
    # A1. Compute the Slopes
    m,b = regress_2d(in_ts,invar_rs) # [1 x pts]
    
    # A2. Calculate SSE and residual standard error
    # https://www.geo.fu-berlin.de/en/v/soga-r/Basics-of-statistics/Hypothesis-Tests/Inferential-Methods-in-Regression-and-Correlation/Inferences-About-the-Slope/index.html
    yhat    = in_ts[None,:] * m.T  + b.T # Re-make the model
    epsilon = invar_rs - yhat # Residual
    SSE     = (epsilon**2).sum(1) # Errors are generally large along NAC
    if dof is None:
        if verbose:
            print("Using DOF len(time) - 2...")
        dof     = nt-2 # Note you can set DOF to be different here. I think 2 is just 2 parameters for linear regr
    se      = np.sqrt(SSE/ (dof)) # Residual Standard Error. 
    
    # A3. Compute the t-statistic
    rss_x = np.sqrt( np.sum( (in_ts - in_ts.mean()) **2))# Root Sum Square of x
    denom = se / rss_x
    tstat = m.squeeze() / denom
    
    # A4. Get Critical T
    ptilde  = p/tails
    critval = stats.t.ppf(1-ptilde,dof)
    if tails == 2:
        critval_lower = stats.t.ppf(ptilde,dof)
    
    # Make significance Mask
    if tails == 2:
        sigmask = (tstat > critval) | (tstat < critval_lower)
    else:
        sigmask = tstat > critval
    
    
    sigmask = replace(sigmask)
    
    # Return all values
    outdict = {}
    outdict["regression_coeff"] = replace(m.squeeze())
    outdict["intercept"] = replace(b.squeeze())
    outdict["SSE"] = replace(SSE)
    outdict["se"] = replace(se)
    outdict["t_statistic"] = replace(tstat)
    outdict["t_critval"] = critval
    outdict["sigmask"] = sigmask
    if tails == 2:
        outdict['t_critval_lower'] = critval_lower
    
    return outdict

def reshape_2d_ds(inarr,ds_ori,oldshape_trans,newdims):
    inarr_rs    = inarr.reshape(oldshape_trans)
    coords_new  = {}
    for dname in newdims:
        coords_new[dname] = ds_ori[dname]
    #coords_new = ds_ori.transpose(*newdims).coords 
    da_inarr_rs = xr.DataArray(inarr_rs,coords=ds_ori.coords,dims=coords_new,name=ds_ori.name,)
    
    da_inarr_rs = da_inarr_rs.transpose(*ds_ori.dims)
    return da_inarr_rs


def sel_region_xr(ds,bbox,verbose=False):
    """
    Selects region given bbox = [West Bnd, East Bnd, South Bnd, North Bnd]
    Defaults to coordinates of bbox (degrees East or West) and swaps ds accordingly
    
    
    Parameters
    ----------
    ds : xr.DataArray or Dataset
        Assumes "lat" and "lon" variables are [present]
    bbox : LIST
        Boundaries[West Bnd, East Bnd, South Bnd, North Bnd]
        
    Returns
    -------
        Subsetted datasetor dataarray
        
    """
    #bbox = np.array(bbox,dtype=object)
    check_latitude_ascending = (ds.lat[1] - ds.lat[0] > 0)
    check_lon360 = np.any(ds.lon > 180).data.item()
    if bbox[2] > bbox[3] and check_latitude_ascending:
        if verbose:
            print("Warning! Southern Latitude Bound > Northern Latitude Bound for increasing latitudes...")
            print("\tSelecting latitudes in reverse order")
        latN,latS = bbox[2:]
        bbox[2] = latS
        bbox[3] = latN
        
    # Ensure that bbox and ds have the same longitude
    bbox_lon360 = np.all(np.array(bbox)>0)
    if (bbox_lon360 == 0) and (check_lon360 == 1):
        if verbose:
            print("Converting BBOX lon to degrees East")
        for ii in [0,1]:
            if bbox[ii] < 0:
                bbox[ii] = bbox[ii] + 360
    elif (bbox_lon360 == 1) and (check_lon360 == 0):
        if verbose:
            print("Converting BBOX lon degrees Wast")
        for ii in [0,1]:
            if bbox[ii] > 180:
                bbox[ii] = bbox[ii] - 360
        
        
    
    if bbox[0] > bbox[1]: # Checks Longitude Issues 
        if verbose:
            print("Warning! Eastern Longitude Bound > Western Longitude Bound...")
        if np.any(np.array(bbox[:2]) > 180): # Degrees East
            if verbose:
                print("\tDegrees East Detected. Crossing Prime Meridian.")
            if check_lon360 == 0 :
                if verbose:
                    print("\tAutomatically converting ds to degrees East")
                ds = lon180to360_xr(ds)
            dswest = ds.sel(lon=slice(bbox[0],360),lat=slice(bbox[2],bbox[3]))
            dseast = ds.sel(lon=slice(0,bbox[1]),lat=slice(bbox[2],bbox[3]))
            return xr.concat([dswest,dseast],dim='lon')
        
        elif np.any(np.array(bbox[:2])<0): # Degrees West
            if verbose:
                print("\tDegrees West Detected. Crossing Date Line.") 
            if check_lon360 == 1:
                print("\tAutomatically converting ds to degrees West")
                ds = lon360to180_xr(ds)
            dswest = ds.sel(lon=slice(bbox[0],180),lat=slice(bbox[2],bbox[3]))
            dseast = ds.sel(lon=slice(-180,bbox[1]),lat=slice(bbox[2],bbox[3]))
            return xr.concat([dswest,dseast],dim='lon')
        
        else:
            if verbose:
                print("\t Case not found, please check function...")
            return None
        
    return ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))

def xrdeseason(ds,check_mon=True,verbose=True):
    """ Remove seasonal cycle, given an Dataarray with dimension 'time'"""
    if check_mon:
        try: 
            if ds.time[0].values.item().month != 1:
                print("Warning, first month is not Jan...")
        except:
            if verbose:
                print("Warning, not checking for feb start")
    
    return ds.groupby('time.month') - ds.groupby('time.month').mean('time')

def tilebylag(kmonth,var,lags): 
    """
    Tile a monthly variable along a lag sequence,
    shifting to recenter lag0 = kmonth+1

    Parameters
    ----------
    kmonth : INT
        Index of month at lag 0 (ex. Jan, kmonth=0)
    var : ARRAY [12,]
        Monthly variable
    lags : ARRAY [nlags]
        Lags to tile along

    Returns
    -------
    vartile : ARRAY [nlags]
        Tiled and shifted variable

    """
    vartile = np.tile(np.array(var),int(np.floor(len(lags)/12))) 
    vartile = np.concatenate([np.roll(vartile,-kmonth),[var[kmonth]]])
    return vartile

def xrdetrend(ds,timename='time',verbose=True):
    
    st          = time.time()
    if len(ds.shape) == 1: 
        ts_dt       = sp.signal.detrend(ds.data)
    else:
        # Simple Linear detrend along dimension 'time'
        tdim        = list(ds.dims).index(timename) # Locate Time Dim
        dt_dict     = detrend_dim(ds.values,tdim,return_dict=True) # ASSUME TIME in first axis
        ts_dt       = dt_dict['detrended_var']
    ds_anom_out = xr.DataArray(ts_dt,dims=ds.dims,coords=ds.coords,name=ds.name)
    if verbose:
        print("Detrended in %.2fs" % (time.time()-st))
    return ds_anom_out

def year2mon(ts,return_monxyr=True):
    """
    Separate mon x year from a 1D timeseries of monthly data
    """
    ts = np.reshape(ts,(int(np.ceil(ts.size/12)),12))
    if return_monxyr:
        ts = ts.T # [year x mon] --> [mon x year]
    return ts 


def yo_taper(x,pct,dim,debug=False):
    """
    
    YO_TAPER : TAPER
    ============================================================================
    
    USAGE : y=yo_taper(x,pct{,dim})
    
    DESCRIPTION : 
        Tapering the both ends of the input time series 
        
    INPUT :    x   [Array]    input time series (recommended to be detrended)
                              Can be 1-D or 2-D (multiple time series).
              pct  [Numeric]  percent of the series to be tapered (0.0<=pct<=1.0) 
              dim  [Int]      0 : each column is individual time series (default)
                              1 : each row is individual time series
             debug [Bool]     visualize tapering results
    
    OUTPUT :    y  [Array]    tapered output timeseries (1-D or 2-D)
    
    AUTHOR :  Young-Oh Kwon  (2003/12/15)
              Python Version: Glenn Liu (2021/02/11)
    
    NOTES  :  Still need to add error handling...
    
    """
    #------------------------------------------------------------------------
    # Check [pct]
    #------------------------------------------------------------------------
    if (pct<0) | (pct>1):
        print("pct should be between 0 and 1")
        exit
        
    #------------------------------------------------------------------------
    # Rotate the input vector if dim==1
    #------------------------------------------------------------------------
    if dim == 1:
        x = x.T
    
    #------------------------------------------------------------------------
    # Add dimension if x is 1-D
    #------------------------------------------------------------------------
    flag1D = False
    if len(x.shape) < 2:
        x = x[:,None]
        flag1D = True
    
    #------------------------------------------------------------------------
    # Taper
    #------------------------------------------------------------------------
    tslength = x.shape[0]                    # Length of each timeseries
    #numts    = x.shape[1]                   # Number of timeseries
    numtaper = int(tslength*pct/2)           # Number of points to be tapered
    
    # Remove mean
    tsavg    = x.mean(0)
    y        = x - tsavg[None,:] # Demean
    
    # Create tapering multiplers
    taperbeg = np.zeros(numtaper)
    taperend = np.zeros(numtaper)
    for i in range(numtaper):
        taperbeg[i] = (1 - np.cos( np.pi / (numtaper-1) * (i) )) /2
        taperend[i] = (1 + np.cos( np.pi / (numtaper-1) * (i) )) /2
        
    # Apply tapering to timeseries
    ytaper = y.copy()
    ytaper[:numtaper,:]   = ytaper[:numtaper,:]  * taperbeg[:,None]
    ytaper[-numtaper:,:]  = ytaper[-numtaper:,:] * taperend[:,None]
    
    if numtaper == 1:
        ytaper[0,:]  = 0
        ytaper[-1,:] = 0
    
    if debug: # Visualize tapering results
        t = 0
        
        # Plot Tapering Windows
        fig,ax = plt.subplots(1,1)
        ax.plot(taperbeg,color='k',label="Taper Beginning")
        ax.plot(taperend,color='r',label="Taper End")
        ax.plot(np.flip(taperend),color='orange',label="Taper End (Flipped)")
        ax.grid(True,ls='dotted')
        ax.legend()
        ax.set_title("Tapering Windows")
        
        # Plot Timeseries before and after tapering
        fig,ax = plt.subplots(1,1)
        ax.plot(y[:,t],color='k',label="Raw")
        ax.plot(ytaper[:,t],color='r',label="Tapered")
        ax.grid(True,ls='dotted')
        ax.legend()
        ax.set_title("Timeseries Tapering")
        
        # Zoom in on tapering results, beginning and end
        fig,axs = plt.subplots(2,1)
        ax = axs[0] # Beginning
        ax.plot(y[:numtaper,t],color='k',label="Raw")
        ax.plot(ytaper[:numtaper,t],color='r',label="Tapered")
        ax.grid(True,ls='dotted')
        ax.legend()
        ax.set_title("Beginning")
        
        ax = axs[1] # End
        ax.plot(y[-numtaper:,t],color='k',label="Raw")
        ax.plot(ytaper[-numtaper:,t],color='r',label="Tapered")
        ax.grid(True,ls='dotted')
        ax.legend()
        ax.set_title("End")
        plt.tight_layout()
    
    #------------------------------------------------------------------------
    # Rotate the output vector if dim==1
    #------------------------------------------------------------------------
    if dim == 1:
        y = ytaper.T
    else:
        y = ytaper
    
    #------------------------------------------------------------------------
    # Squeeze output if input was 1D
    #------------------------------------------------------------------------
    if flag1D:
        return y.squeeze()
    return y

def yo_spec(x,opt,nsmooth,pct,debug=True,verbose=True):
    """
    
    YO_SPEC : Calculating Auto-Spectrum
    ============================================================================
    
    USAGE : [P,freq,dof,r1]=yo_spec(x,opt,nsmooth,pct)
    
    DESCRIPTION : 
      Calculating Auto-Spectrum of input time series 

    INPUT :     x    =  a vector input time series  
               opt   =  detrending option (0: just demean, 1: demean+detrend)
             nsmooth =  # of adjacent frequencies over which smoothing to be 
                        performed on the raw periodogram estimates (>=1)
                          1    : No Smoothing
                          odds : All weights are 1/nsmooth except weight(1) and 
                                 weight(nsmooth) which are 1/(2*nsmooth)
                          even : Routine will force the nsmooth to be the next
                                 largest odd number
               pct   =  percent of the series to be tapered (0.0<=pct<=1.0) 
               
    OUTPUT :    P    =  auto-spectral density of x [x-units^2/cycle/sample-interval]
                        (spectrum is normalized so that the area under the curve
                        (P(1)+P(end))*df/2+sum(P(2:end-1))*df equals the variance 
                        of the detrended series, where df=1/npts=frequency spacing)
               freq  =  frequency [cycles/time]
               dof   =  degree of freedom
               r1    =  lag 1 correlation of input time series
               
    DEPENDENCIES:
        numpy, scipy.signal, scipy.fft
    """
    
    #------------------------------------------------------------------------
    # Check the dimensions of input time series
    #------------------------------------------------------------------------
    if len(x.shape) > 1:
        print("x should be 1-D vectors")
        return
    
    #------------------------------------------------------------------------
    # Check [opt]
    #------------------------------------------------------------------------
    if opt not in [0,1]:
        print("opt should be either 0 or 1")
        return
    
    #------------------------------------------------------------------------
    # Check [nsmooth]
    #------------------------------------------------------------------------
    if nsmooth != np.fix(nsmooth):
        print("nsmooth should be an integer")
        return
    elif nsmooth <=0: 
        print("nsmooth should be greater than zero")
        return
    #------------------------------------------------------------------------
    # Check [pct]
    #------------------------------------------------------------------------
    if (pct<0) | (pct>1):
        print("pct should be between 0 and 1")
        return
        
    #------------------------------------------------------------------------
    #  Make x a column vectors and count the number of data points
    #------------------------------------------------------------------------
    x    = x.flatten()
    npts = x.shape[0] 
    
    #------------------------------------------------------------------------
    # Demean, detrend, tapering
    #------------------------------------------------------------------------
    x    = x - x.mean()
    if opt == 1: 
        x = signal.detrend(x)
    
    VAR  = x.var()
    if pct != 0:
        x    = yo_taper(x,pct,0,debug=debug)
    
    #------------------------------------------------------------------------
    # Calculate the lag 1 correlation
    #------------------------------------------------------------------------
    r1 = np.corrcoef(x[1:],x[:-1])[0,1] # NOTE: Eventually change to yo_cor
    
    #------------------------------------------------------------------------
    # Calculate raw periodogram
    #------------------------------------------------------------------------
    X = fft.fft(x)/npts
    X = X[1:int(npts/2)+1]
    
    if np.fix(npts/2) == npts/2: # Even
        X[:-1] = 2*X[:-1]
    else:
        X = 2*X
    P0 = np.real(X * np.conj(X)) # sum(P0)= (2?) * var(x,1)
    
    #------------------------------------------------------------------------
    # Smooth the periodogram (reflective (symmetric) smoothing)
    #     with modified Daniell window (note: sum(win)=1)
    #------------------------------------------------------------------------
    #if np.fix(nsmooth/2) == np.fix((nsmooth+1)/2): # Original conditional
    if nsmooth%2 == 0: # for even nsmooth
        nsmooth+=1
    
    P0refl = np.hstack([P0,np.flip(P0[1:-1])])
    
    win = np.zeros(P0refl.shape) # Make window (in frequency space)
    #print(win.shape)
    if nsmooth > 1:
        kwin = int((nsmooth+1)/2)
        # first weights
        win[0:kwin]   = 1/(nsmooth-1)
        win[-kwin+1:] = 1/(nsmooth-1)
        
        # half of first weight
        win[kwin-1]  = 0.5*1/(nsmooth-1)
        win[-kwin+1] = 0.5*1/(nsmooth-1)
    else:
        win[1] = 1
    
    if win.sum() != 1:
        if verbose:
            print("Warning, window does not sum to 1!")
    
    # IFFT back to time, apply window, then FFT back (?)
    P1 = np.real( len(win) * fft.fft( fft.ifft(win) * fft.ifft(P0refl) ))
    P1 = P1[:len(P0)]
    
    # ------------------------------------------------------------------------
    # Normalize the smoothed periodogram, so that the area under the curve
    # (P(1)+P(end))*df/2+sum(P(2:end-1))*df = var of the detrended series
    #  where df=1/npts=frequency spacing 
    #------------------------------------------------------------------------
    
    df  = 1/npts          # frequency spacing
    
    WGT = ( (P1[0]+P1[-1]) / 2 + np.sum(P1[1:-1]) ) * df  # area under the spectrum curve
    
    P   = P1*VAR/WGT 
    
    #------------------------------------------------------------------------
    # Calculate degree of freedom with the smoothing and tapering factors
    #------------------------------------------------------------------------
    
    smoothfac = np.sum(win**2)
    taperfac  = 0.5 * (128-93*pct) / (8-5*pct)**2
    dof       = 2/smoothfac/taperfac
    
    #------------------------------------------------------------------------
    # Calculate Frequency grids
    #------------------------------------------------------------------------
    fbw  = dof*df/2                  #frequency bandwidth
    fmax = 0.5                       # Nyquist frequency
    freq = np.arange(1/npts,fmax+df,df);
    
    return P,freq,dof,r1


def yo_speccl(freq,P,dof,r1,clvl=[0.95,]):
    """
    
    YO_SPECCL : Calculating Confidence Curves for the Spectral Density Estimates
    
    ===========================================================================
    USAGE : CC = yo_speccl(freq,P,dof,r1,clvl=[0.95,])
    
    DESCRIPTION : 
        Calculating confidence curves of the spectral density estimates
        with the null hypothesis of the red noise spectrum  
    
    INPUT :   freq  =  frequecy vector as an independent variable
               P    =  spectral density as a dependent variable
              dof   =  degree of freedom in spectral density estimate
               r1   =  Lag 1 correlation of the original time series
              clvl  =  Confidence levels to each confidence curve (0<clvl[i]<1)
                      (default: clvl=[0.95,])
    
    OUTPUT :   CC   =  confidence curves : [len(freq),len(clvl)+1]
                       CC(:,1) is for the red noise spectrum

    AUTHOR :  Young-Oh Kwon  (2003/12/17) 
              Python Version: Glenn Liu (2021/21/2021)
              
    NOTES  :  Still need to add error handling...
    """
    #------------------------------------------------------------------------
    # Calculate the red noise spectrum
    #------------------------------------------------------------------------
    Pred  = 1/(1 + r1**2 - 2*r1*np.cos(2*np.pi*freq))
    scale = np.sum(P)/np.sum(Pred)
    CC    = Pred*scale
    
    # Tile CC along new dim for original spectrum + each confidence level
    nlvl  = len(clvl)
    CC    = np.tile(CC[:,None],nlvl+1) # [freq x (clvl+1)]
    
    #------------------------------------------------------------------------
    # Calculate the confidence curve for each cls
    #------------------------------------------------------------------------
    ## Matlab approach (checked for equivalence)
    #cc = (stats.chi2.ppf(clvl,df=dof)/dof)[None,:] # [1 x nlvl]
    #CC[:,1:] = CC[:,[0]] @ cc # [freq x nlvl] = [freq x 1] * [1 x nlvl]
    
    ## More pythonic way (?) using array broadcasting
    cc       = (stats.chi2.ppf(clvl,df=dof)/dof)
    CC[:,1:] *= cc[None,:]
    
    return CC


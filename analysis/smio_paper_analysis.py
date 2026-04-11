#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Finalized Script to Generate Figures in the Stochastic Model in Observations (SMIO) Paper

Based on the following scripts
Figure 1.A-C : viz_t2_updated_smio.py
Figure 1.D   : area_average_sensitivity.py
Figure 2-3   : area_avg_metrics
Figure 4     : cesm2_hierarchy_v_obs

Created on Tue Mar 10 10:31:33 2026

@author: gliu

"""

import time
import numpy as np
import numpy.ma as ma
import xarray as xr
import sys
import tqdm
import glob 
import scipy as sp
import cartopy.crs as ccrs
import cmcrameri.cm as cm

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#%% Set Paths

smio_utils_path = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/03_Scripts/smio/analysis/"
figpath         = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20260410_Draft06/"

# Path to smio_data_final
dpath           = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/smio_data_final/"

# Load Module with scripts
sys.path.append(smio_utils_path)
import smio_utils as sut

#%% Additional Functions

def setup_errorbar(mc_stds,stds,era5_last=False,include_era5=False):
    
    """
    Given the standard deviations from monte carlo of the simulations (excluding ERA5 for both!;
    or if ERA5 is included, use the [include_era5] toggle...)
    Set up the upper/lower bounds for the error bar (alpha=0.05, two-tailed)
    
    Inputs
    mc_stds      : LIST [ex][mciter]  Standard deviations from Monte Carlo simulations
    stds         : LIST [ex]          Standard deviation from full timeseries
    era5_last    : BOOL               False if ERA5 is first (default), True if ERA5 is last
    include_era5 : BOOL               False in ERA5 is excluded from mc_stds and stds (default), true otherwise
    
    Output
    errbar_var   : ARRAY [lower/upper,experiment] 95% Confidence Interval for barplot
    
    
    """ 
    
    # Preallocate Array
    if include_era5: # Assume ERA5 is included in mc_stds and stds
        nexps = len(mc_stds)
        if era5_last:
            id_experiment = np.arange(nexps)[:id_era] # If ERA is last
        else:
            id_experiment = np.arange(nexps)[(id_era+1):] # If ERA is first (reversed)
        stds_in = stds[id_experiment]
    else:
        nexps           = len(mc_stds) + 1 # Assume mcstds doesn't include ERA5
        stds_in = stds

    
    errbar_var      = np.zeros((2,nexps))
    
    # Convert MC Output to Array for experimental output
    mc_stds_arr      = np.array(mc_stds) # [exp,sample]
    
    # Get Upper and Lower Bounds for experimental output
    # Need to subtract stds to center
    lowervar        = np.abs(np.quantile(mc_stds_arr,0.025,axis=1) - stds_in) # Can't be negative
    uppervar        = np.quantile(mc_stds_arr,0.975,axis=1) - stds_in
    
    # Add space for ERA5 (no confidence interval computed...)
    if era5_last: # Assume ERA5 is the last Entry
        errbar_var[0,:] = np.hstack([lowervar,[None,]])
        errbar_var[1,:] = np.hstack([uppervar,[None,]])
    else: # Assume ERA5 is the first entry
        errbar_var[0,:] = np.hstack([[None,],lowervar])
        errbar_var[1,:] = np.hstack([[None,],uppervar])
    return errbar_var

#%% Plotting Options/Parameters

darkmode = False
if darkmode:
    dfcol = "w"
    bgcol = np.array([15,15,15])/256
    sp_alpha = 0.05
    transparent = True
    plt.style.use('dark_background')
    #mpl.rcParams['font.family']     = 'Avenir'
else:
    dfcol = "k"
    bgcol = "w"
    sp_alpha = 0.75
    transparent = False
    plt.style.use('default')

proj        = ccrs.PlateCarree()
mpl.rcParams['font.family'] = 'Arial'
mons3       = sut.get_monstr(nletters=3)

#%% Load Sea Ice Concentration File (Median 15%)

ncice   = "%sERA5_IceMask_Global_1979_2024_Median15.nc" % (dpath)
dsice   = xr.open_dataset(ncice)

dsmask  = dsice.median_mask_15pct
dsmask  = dsmask.rename({'latitude':'lat','longitude':'lon'})
dsmask  = sut.lon360to180_xr(dsmask)

#%% Also Load the MLD, regridded to ERA5 resolution using bilinear interp

ncmld   = "%sMIMOC_regridERA5_h_pilot.nc" % (dpath)
dsmld   = xr.open_dataset(ncmld).load()

# =============================================================================
#%% Load Hierarchy Data for Stochastic Model
# =============================================================================
"""
Area-Average Timeseries: for Figure 1D
For calculating results for Figures 2,3

"""

#% Standardized Names and Colors
comparename     = "Draft06"
expnames        = [
                    "SST_ERA5_1979_2024",
                    "Stochastic_Model_withReemergence",
                    "Stochastic_Model_NoReemergence",
                    "Stochastic_Model_Slablike",
                   ]
expnames_long   = ["Observations (ERA5)",
                   "Stochastic Model (with re-emergence)",
                   "Stochastic Model (no re-emergence)",
                   "Stochastic Model (slab-like)",
                   ]
expnames_short  = ["Obs.",
                   "Re.",
                   "No Re.",
                   "Slab"]
expcols         = ["k",
                   "gold",
                   "turquoise",
                   "salmon"]
expls           = ['solid',
                   'dashed',
                   'dashed',
                   'dotted']


# Region Info
regname             = "SPGNE"
bbsel               = [-40,-15,52,62]

# Global Mean Detrending Options
id_era = expnames.index("SST_ERA5_1979_2024")
detrend_obs_regression = True # Set to True to detrend obs using monthly regression to global mean
nc_gmsst               = "ERA5_GMSST_1979_2024.nc"

#% Load output area-averaged over the northeastern subpolar gyre (SPGNE)
nexps = len(expnames)
dsall = []
for ex in tqdm.tqdm(range(nexps)):
    expname = expnames[ex]
    ncname  = "%s%s_SST_SPGNE_Average.nc" % (dpath,expname)
    
    ds      = xr.open_dataset(ncname).load()
    dsall.append(ds)

#% Preprocessing (Deseason Only)
dsa      = [sut.xrdeseason(ds) for ds in dsall]
ssts     = [sp.signal.detrend(ds.SST.data) for ds in dsa] # Detrend for Stochastic Model
ssts_ds  = [sut.xrdetrend(ds.SST) for ds in dsa] # Detrend for Stochastic Model

#% Detrend ERA5 using Global Mean Regression Approach
if (detrend_obs_regression):
    
    sst_era = dsa[id_era].SST # Take undetrended ERA5 SST
    sst_era = sst_era.expand_dims(dim=dict(lon=1,lat=1))
    
    # Load GMSST
    ds_gmsst     = xr.open_dataset(dpath + nc_gmsst).GMSST_MaxIce.load()
    dsdtera5     = sut.detrend_by_regression(sst_era,ds_gmsst,regress_monthly=True)
    
    sst_era_dt = dsdtera5.SST.squeeze()
    
    ssts_ds[id_era] = sst_era_dt
    ssts[id_era]    = sst_era_dt.data


# =============================================================================
#%% Figure 1 T2 and Timeseries...
# =============================================================================

"""
Additional Notes on Figure 1:
    
Panels A-C
    
"""

#%% # Figure 1 Panels A-C (MLD and T2)
#% Load Necessary Data (autocorrelations) and compute T2

# Additional Options and Toggles
st             = time.time()

# Input Information
sm_lvl3     = dpath   + "Stochastic_Model_withReemergence_ACF_JFM.nc"
sm_lvl2     = dpath   + "Stochastic_Model_NoReemergence_ACF_JFM.nc"
sm_lvl1     = dpath   + "Stochastic_Model_Slablike_ACF_JFM.nc"
ncs_sm      = [sm_lvl3,sm_lvl2,sm_lvl1]

# Load NetCDFs for Stochastic Model Hierarchy
acfs_sm = [xr.open_dataset(nc).load() for nc in ncs_sm]

# For SM, integrate whole ACF due to long integration time
t2_sm   = [sut.calc_T2(ds.acf.squeeze(),axis=-1,verbose=True,ds=True) for ds in acfs_sm]
t2_sm   = [t2.mean(0) for t2 in t2_sm] # Take Mean across 10 stochastic model simulations

# Load Observational T2
print("Loading T2 Computed by Martha using AR-Fitting")
fn              = "ERA5_T2_SSTwint_Martha.nc"
ds2             = xr.open_dataset(dpath+fn).load()

# Adjust Lat/Lon to match Stochastic Model
lonm            = np.linspace(0,360,ds2.XC.shape[1])#ds2.XC#inds[0].lon
latm            = np.linspace(-90,90,ds2.YC.shape[0])#ds2.YC#inds[0].lat
coords          = dict(lat=latm,lon=lonm,)
ds2_martha      = xr.DataArray(np.flip(ds2.T2.data,0),coords=coords,dims=coords,name="T2")
ds2_martha      = sut.lon360to180_xr(ds2_martha)
ds2_martha      = sut.sel_region_xr(ds2_martha,[-80,0,0,90],verbose=True)

# Coordinates were somehow messed up (by sel_region_xr code, or maybe before...)
ds2_martha = ds2_martha.sortby('lon')
ds2_martha = ds2_martha.sortby('lat')

sut.printtime(st,"Loaded ACFs and calculated T2")

#%% Visualize 1.A-C
# This is the final version used in Figure 1 of the paper

pmesh           = False
fsz_axis        = 22
fsz_tick        = 18
fsz_title       = 28

# 
id_sm           = 0 # Index of Stochastic Model to plot
# 0 = With Re-emergence, 1 = No Re-emergence, 2 = Slab-Like for comparename=Draft05_ReverseOrder

bbox_spgne      = [-40,-15,52,62]

#Contour Settings
cints_t2_lab    = np.arange(1,6,1)
cints_t2        = np.arange(1,4.2,0.2)

# Bounding Box
bbsel           = [-65, -5, 45, 65] # [-40, -12, 50, 62]
centlat         = 55
centlon         = -35
fix_lon         = np.arange(-60,10,10)
fix_lat         = np.arange(45,70,5)

mld_cbticks     = np.arange(0,600,100)
t2_cbticks      = np.arange(1,5,1)

fig,axs,_       = sut.init_orthomap(1,3,bbsel,figsize=(24,12),centlat=centlat,centlon=centlon)

for a,ax in enumerate(axs):
    ii              = a
    ax              = sut.add_coast_grid(ax,bbox=bbsel,line_color='k',
                                        fill_color="lightgray",fontsize=fsz_tick,
                                        fix_lon=fix_lon,fix_lat=fix_lat)
    
    
    # Plot the t2
    if a == 1:

        plotvar     = ds2_martha 
        lon         = ds2_martha.lon
        lat         = ds2_martha.lat

        
        cmap        = cm.devon_r#'cmo.dense'#cm.acton_r
        cints       = cints_t2 #np.arange(0,5.5,0.5)
        cints_lab   = cints_t2_lab#cints[::2]
        clab        = "Decorrelation Timescale $T_2$ (Years)"
        cbticks     = t2_cbticks
        
    elif a == 0:
        plotvar     = dsmld.h.max('mon')# * dsmasksm
        lon         = plotvar.lon
        lat         = plotvar.lat
        cmap        = 'cmo.ice_r'
        cints       = np.arange(0,525,25)
        cints_lab   = cints[::4]
        clab        = "Maximum Climatological Mixed-Layer Depth (meters)"
        
        cbticks     = mld_cbticks
        
        
    elif a == 2:
        
        plotvar     = t2_sm[id_sm].T #* blowupmask_apply.data #* dsmaskera.data
        lon         = acfs_sm[id_sm].lon
        lat         = acfs_sm[id_sm].lat
        cmap        = cm.devon_r#'cmo.dense'#cm.acton_r
        cints       = cints_t2#np.arange(0,4.2,0.2)#np.arange(0,5.5,0.5)#np.arange(0,36,3)
        cints_lab   = cints_t2_lab #cints[::2]
        clab        = "Decorrelation Timescale $T_2$ (Years)"
        
        cbticks     = t2_cbticks
    
    
    cf      = ax.contourf(lon,lat,plotvar,
                          levels=cints,
                          transform=proj,cmap=cmap,zorder=-1,extend='both')
    
    cl      = ax.contour(lon,lat,plotvar,
                          levels=cints_lab,linewidths=.55,colors='w',
                          transform=proj,zorder=-1)
    ax.clabel(cl,fontsize=fsz_tick)

    
    if a == 0:
        cb = sut.hcbar(cf,ax=ax,fontsize=22,pad=0.0001,fraction=0.040)
        cb.set_label(clab,fontsize=fsz_axis)
        cb.set_ticks(cbticks)
    elif a == 2:
        cb = sut.hcbar(cf,ax=axs[1:].flatten(),fontsize=22,pad=0.0001,fraction=0.040)
        cb.set_label(clab,fontsize=fsz_axis)
        cb.set_ticks(cbticks)
    
    # Plot the Median Sea Ice Concentration
    plotvar = dsice
    icel    = ax.contour(plotvar.longitude,plotvar.latitude,plotvar.siconc_winter_median,levels=[0.15,],
                      colors='cyan',linewidths=4,transform=proj,linestyles='dotted')
    
    
    bb = sut.plot_box(bbox_spgne,ax=ax,color='purple',linewidth=4.5,proj=proj)
    sut.label_sp(a,case='lower',ax=ax,fig=fig,fontsize=fsz_title,labelstyle="%s",weight='bold')

figname = figpath + "Figure01_ABC_%s.png" % comparename 
plt.savefig(figname,dpi=300,bbox_inches='tight',transparent=transparent)

#%% Figure 1 Panel D
# Set up variables and determine index to plot (based on rmse)

#% Assume this is for the first 2 experiments in expnames (ERA5 and most complex Stochastic Model)
era_sst           = ssts[id_era].squeeze()
sm_sst_sel        = ssts[1]

era_sst_ds        = dsall[id_era]
sm_sst_ds         = dsall[1]

ntime_sm    = len(sm_sst_sel)
ntime_era   = len(era_sst)
find_imin   = False    #Set to True to manually set value
imin_in     = 713 * 12 #80545 # Index of Minimum (user input, set find_imin to False to recalculate)
lpcutoff    = 240

if find_imin: # Find Location Time with Best Overlap via RMSE
    # Find Time with Best Fit
    nsegments = ntime_sm - ntime_era
    rmse_all  = np.zeros((nsegments)) * np.nan
    for ii in tqdm.tqdm(range(nsegments)):
        idxin = np.arange(ii,ii+ntime_era)
        
            
        sstin            = sm_sst_sel[idxin]
        sstref           = era_sst
        if lpcutoff is not None:
            sstin  = sut.lp_butter(sstin,lpcutoff,6)
            sstref = sut.lp_butter(sstref,lpcutoff,6)
            
        rmse_all[ii]  = np.sqrt(np.nanmean(sstin - sstref)**2)
        
    # Apply 12-month Low Pass Filter to RMSE to smooth out seasonal variability
    rmsein = sut.lp_butter(rmse_all,12,6)
    
    # Find minimum
    imin   = np.nanargmin(rmsein).item()
    
else: # Use Selected Imin_in
    imin   = imin_in
    
#%% Plot Timeseries

# Copied section from area_average_sensitivity
remove_topright   = True
label_actual_year = True
ylims             = [-2,2]
ts_lw             = 2 # Timeseries Line Width

# Set ERA5 Time Information
xplot             = np.arange(ntime_era)
plotint           = 36
times             = era_sst_ds.time.data
years             = [str(t)[:4] for t in times]

# Get Stochastic Model Time Information
if label_actual_year:
    times_sm   = sm_sst_ds.time.data[imin:imin+ntime_era]
else:
    times_sm   = sm_sst_ds.time.data[:ntime_era]
years_sm   = [str(t)[:4] for t in times_sm]

# Set Fontsizes
fsz_legend        = 18
fsz_ticks         = 18
fsz_axis          = 22
fsz_title         = 22

# Initialize Fig -------------
fig,ax1           = plt.subplots(1,1,constrained_layout=True,figsize=(14,4.5))

# Plot ERA5 Timeseries
ax      = ax1
l1      = ax.plot(era_sst,label=expnames_long[-1],c=dfcol,lw=ts_lw)
era_lp  = sut.lp_butter(era_sst,120,6)

# Plot Stochastic Model Timeseries
ax2 = ax.twiny()
ax  = ax2
l2  = ax.plot(sm_sst_sel[imin:imin+ntime_era],label=expnames_long[1],c=expcols[1],lw=ts_lw)

# Set Legend
ax1.legend([l1[0],l2[0]],
           ['Observations (ERA5)','Stochastic Model (with Re-emergence)'],
           loc = 'lower left',
           fontsize=fsz_legend,ncol=2)

# # Label Axes
# for ax in axs:
#     ax.legend(fontsize=fsz_legend)
#     ax.set_xlim(xlims)
#     ax.set_ylim(ylims)
#     ax.set_ylabel("SST Anomaly [$\degree$C]",fontsize=fsz_axis)
#     ax.axhline([0],ls='dotted',c=dfcol,lw=0.75)
    
#     if remove_topright:
#         ax.spines[['right', 'top']].set_visible(False)

# Axes Formatting -----------
# Set xticks for ERA5
ax = ax1
ax.set_xticks(xplot[::plotint],labels=years[::plotint])
ax.set_xlim([xplot[0],xplot[-1]])

# Set xticks for SM
ax = ax2
ax.set_xticks(xplot[::plotint],labels=years_sm[::plotint])
ax.set_xlim([xplot[0],xplot[-1]])

# Set Y-axis Limits, add zero line
ax1.set_ylim(ylims)
ax.axhline([0],ls='dotted',c=dfcol,lw=0.75)

# Set Labels for ERA5
ax1.set_ylabel("SPGNE SST Anomaly ($\degree C$)",fontsize=fsz_axis)
ax1.tick_params(labelsize=fsz_ticks)
ax1.set_xlabel("Year (Observations)",fontsize=fsz_axis)

# Adjust Axes 2 color and size for Stochastic Model
ax2col = expcols[1]
if ax2col == "gold":
    ax2col = "goldenrod"
ax2.tick_params(labelsize=fsz_ticks,colors=ax2col)
ax1.spines['top'].set_color(ax2col)
ax2.set_xlabel("Stochastic Model Simulation Year",
               fontsize=fsz_axis,color=ax2col)
if remove_topright:
    ax.spines[['right', 'top']].set_visible(False)

# Label Subplot
sut.label_sp(3,case='lower',ax=ax,y=1.2,x=-.1, fig=fig,fontsize=fsz_title,labelstyle="%s",weight='bold')

figname = "%sFigure01D_Timeseries_%s_imin%i_lfp%03i.png" % (figpath,comparename,imin,lpcutoff)

plt.savefig(figname,dpi=150,bbox_inches='tight',transparent=transparent)

# =============================================================================
#%% Figure 2: Stochastic Model Autocorrelations
# =============================================================================

"""
Additional Notes on Figure 2:
    
"""

#%% Calculate necessary metrics

ssts         = [ds.data for ds in ssts_ds]
lags         = np.arange(61)
nsmooths     = [4,] + [250,] * (nexps-1)

metrics_out  = sut.compute_sm_metrics(ssts,nsmooth=nsmooths,lags=lags,detrend_acf=False)
monstds      = [ss.groupby('time.month').std('time') for ss in ssts_ds]

# Calculate Standard Deviations and Low-Pass Filtered Variance
stds         = np.array([ds.std('time').data.item() for ds in ssts_ds])
ssts_lp      = [sut.lp_butter(ts,120,6) for ts in ssts_ds]
stds_lp      = np.array([np.std(ds) for ds in ssts_lp])
vratio       = (stds_lp  / stds) * 100

# Calculate High Pass Filtered variance as well
ssts_hp      = [ssts_ds[ii] - ssts_lp[ii] for ii in range(nexps)]
stds_hp      = np.array([np.std(ds) for ds in ssts_hp])

#%% Bootstrap for Spectra

st      = time.time()
nsmooth = 4
mciter  = 10000
ex      = 1
pct     = 0.10

eraspec_dict = sut.quick_spectrum([ssts[id_era],],[nsmooth,],pct,return_dict=True)

stochmod_conts     = []

mc_specdicts       = []

for ex in np.arange(nexps):
    
    if ex == id_era:
        continue # Do not Bootstrap for ERA5
    
    stochmod_ts       = ssts[ex]
    
    #ntime_era         = len(ssts[id_era])
    mcdict            = sut.mcsampler(stochmod_ts,ntime_era,mciter)
    stochmod_samples  = [mcdict['samples'][ii,:] for ii in range(mciter)]
    
    stochmod_specdict = sut.quick_spectrum(stochmod_samples,[nsmooth,]*mciter,pct,return_dict=True,make_arr=True)
    specdict_cont     = sut.quick_spectrum([stochmod_ts,],[250,]*mciter,pct,return_dict=True)
    stochmod_conts.append(specdict_cont)
    mc_specdicts.append(stochmod_specdict)

sut.printtime(st,"Bootstrapping calculations for spectra completed")

#%% Bootstrap for Standard Deviations

st      = time.time()

# Take Samples, Skipping ERA5
mcstds,mcstds_lp,monstds_sample = sut.mcsample_stdev_metrics(ssts[(id_era+1):],ntime_era,mciter)
    
sut.printtime(st,"Bootstrapping calculations for monthly stdev completed")

#%% Get Confidence Interval for ERA5

n_eff           = sut.calc_dof(ssts[id_era],) # calculate effective dof

# Get theoretical chi^2 PDF using n_eff-1
xvariances      = np.linspace(0,1,100)
era5var_pdf     = sp.stats.chi2.pdf(xvariances,n_eff-1,)#loc=stds[-1]**2)

nu      = n_eff -1 
alpha   = 0.05
upperv  = sp.stats.chi2.isf(1-alpha/2,nu)
lowerv  = sp.stats.chi2.isf(alpha/2,nu)

lower   = nu / lowerv
upper   = nu / upperv

lowerbnd_era5 = np.sqrt(lower)
upperbnd_era5 = np.sqrt(upper)

#%% Set Up errorbars for barplot

errbar_var    = setup_errorbar(mcstds,stds[(id_era+1):],era5_last=False,include_era5=False)
errbar_var_lp = setup_errorbar(mcstds_lp,stds_lp[(id_era+1):],era5_last=False,include_era5=False)


#%% Plot Autocorrelation


fsz_title = 26
fsz_ticks = 14
plotkmons = [2,6]
use_neff  = True
conf      = 0.95
alpha     = 0.15 #0.15

monsfull = sut.get_monstr(nletters=None)
xtks     = lags[::6]

if nexps == 3:
    fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(8,6.5))
else:
    fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(8,7))

for ii in range(2):
    ax     = axs[ii]
    kmonth = plotkmons[ii]
    
    ax,ax2   = sut.init_acplot(kmonth,xtks,lags,ax=ax,title="")
    
    for ex in range(nexps):
        
        if ex == id_era:
            col_in = dfcol
        else:
            col_in = expcols[ex]
        
        plotvar = metrics_out['acfs'][kmonth][ex]
        
        zorders = [9,9,2,2,]
        zorder  = zorders[ii]
        
        ax.plot(lags,plotvar,
                label=expnames_long[ex],color=col_in,ls=expls[ex],lw=2.5,zorder=zorder)
        
        if use_neff:
            dof_in = sut.calc_dof(ssts[ex][kmonth::12])
        else:
            dof_in = len(ssts[ex])/12
        
        print("%s for mon %i, DOF In = %.2f" % (expnames_long[ex],kmonth+1,dof_in))
        
        cflag = sut.calc_conflag(plotvar,conf,2,dof_in)
        if ex == id_era:
            if darkmode:
                alpha = 0.15
            else:
                alpha = 0.05
        else:
            alpha = 0.15
        
        ax.fill_between(lags,cflag[:,0],cflag[:,1],alpha=alpha,color=col_in,zorder=12)
        
    if ii == 0:
        
        ax.set_xlabel("")
        
    else:
        ax.set_xlabel("Lag (months)")
        if nexps == 3:
            ax.legend(framealpha=0,fontsize=fsz_ticks,ncol=2)
        else:
            ax.legend(framealpha=0,fontsize=12,ncol=1)
            
    
    ax2.tick_params(labelsize=fsz_tick-4)
    ax.set_ylim([-.25,1])
    
    ax.set_title("")
    ax.set_ylabel("Correlation with \n %s Anomalies" % (monsfull[kmonth]),fontsize=fsz_tick)
    
    sut.label_sp(ii,case='lower',alpha=0.15,ax=ax,y=1.28,x=-.15,fig=fig,fontsize=fsz_title,labelstyle="%s",
                 weight='bold',fontcolor=dfcol)
    
    ax.tick_params(labelsize=fsz_ticks)
    
figname = "%sFigure02_ACF_%s_PaperOutline.png" % (figpath,comparename)

plt.savefig(figname,dpi=300,transparent=transparent)

# =============================================================================
#%% Figure 3: Stochastic Model Hierarchy Variance
# =============================================================================

"""
Additional Notes on Figure 3:

"""


#%% Visualize All Subpanels for Figure 3

# Set Figure Parameters
fsz_axis            = 16
fsz_ticks           = 16
fsz_legend          = 14
fsz_legend_spectra  = 16 # Reduced size for spectra plot

remove_topright     = True # True to remove top and right axlines of Panels A and B



# Initialize Figure using Gridspec
fig             = plt.figure(figsize=(14,10))
gs              = gridspec.GridSpec(8,12)

ax11            = fig.add_subplot(gs[:3,:3],) # Barplot
ax22            = fig.add_subplot(gs[:3,4:11])  # Month Std
ax33            = fig.add_subplot(gs[4:,:11]) # Spectra

# --------------------------------- # Barplot
ax               = ax11

# Panel A Options
expcols_bar          = np.array(expcols).copy()
expcols_bar[id_era]  = 'gray' # Set color to gray for ERA5
label_vratio         = False  # True to Label variance ratios
label_stds           = True   # True to label stdev on bars
ytks_var             = np.arange(0.2,1.2,0.2)

# Set Input Data
instd            = stds
instd_lp         = stds_lp

# Create Labels
if label_vratio:
    xlabs           = ["%s\n%.2f" % (expnames_long[ii],vratio[ii])+"%" for ii in range(len(vratio))]
else:
    xlabs  = expnames_short

# Plot Bars
braw            = ax.bar(np.arange(nexps),instd,color=expcols_bar,yerr=errbar_var,
                         error_kw=dict(ecolor='darkgray',
                                       barsabove=True,
                                       capsize=5,marker="o",markersize=25,mfc='None',
                                       ))
blp             = ax.bar(np.arange(nexps),instd_lp,color=dfcol,yerr=errbar_var_lp,
                         error_kw=dict(ecolor='w',
                                       barsabove=True,
                                       capsize=5,marker="d",markersize=25,mfc='None',))

# Label Standard Deviations
if darkmode:
    upperlbl_col = 'lightgray'
else:
    upperlbl_col = 'gray'
if label_stds:
    ax.bar_label(braw,fmt="%.02f",c=upperlbl_col,fontsize=fsz_axis)
    ax.bar_label(blp,fmt="%.02f",c='w',fontsize=fsz_axis,label_type='center')

# Make Fake Legend and place on plot
colorsf = {'Raw':'gray','10-year low-pass':'k',}         
labels  = list(colorsf.keys())
handles = [plt.Rectangle((0,0),1,1, color=colorsf[label]) for label in labels]
ax.legend(handles, labels,fontsize=fsz_legend,framealpha=0,
          bbox_to_anchor=(0.04, 0.82, 1., .102))


# Set X and Y Labels, Ticks
ax.set_xticks(np.arange(nexps),labels=xlabs,fontsize=fsz_tick,rotation=45)
ax.set_ylabel("SST \nStandard Deviation ($\degree$C)",fontsize=fsz_axis)
ax.set_ylim([0,1.0])
ax.set_yticks(ytks_var)
ax.tick_params(labelsize=fsz_tick)
if remove_topright:
    ax.spines[['right', 'top']].set_visible(False)

# Label Subplot
sut.label_sp(0,case='lower',alpha=0.15,ax=ax,y=1.10,x=-.45,fig=fig,fontsize=fsz_title,labelstyle="%s",
             weight='bold',fontcolor=dfcol)



# --------------------------------- # Monthly Variance
ax              = ax22

# Looping by Experiment
for ex in range(nexps):
    zorders = [9,9,2,2,]
    zorder = zorders[ex]
        
    # Plot Monthly standard deviation
    plotvar = monstds[ex]
    ax.plot(mons3,plotvar,label=expnames_long[ex],
            color=expcols[ex],lw=2.5,ls=expls[ex],marker="o",zorder=zorder)
    
    # Plot Confidence Interval (not for ERA5)
    if ex != id_era:
        exm1 = ex-1 # Didnt do Monte Carlo for ERA5
        plotmc = monstds_sample[exm1]
        bnds   = np.quantile(plotmc,[0.025,0.95],axis=0)
        

        
        ax.fill_between(mons3,bnds[0],bnds[1],color=expcols[ex],alpha=0.10,zorder=1)


# Set X and Y Labels, Ticks
ax.set_ylabel("Monthly SST\nStandard Deviation ($\degree$C)",fontsize=fsz_axis)
ax.set_xticklabels(mons3)
ax.set_xlim([0,11])
ax.set_ylim([0,0.75])
ax.set_yticks(np.arange(0.2,1.1,0.2))
ax.tick_params(labelsize=fsz_ticks)
if remove_topright:
    ax.spines[['right', 'top']].set_visible(False)

# Label Subplot
sut.label_sp(1,case='lower',alpha=0.15,ax=ax,y=1.10,x=-.15,fig=fig,fontsize=fsz_title,labelstyle="%s",
             weight='bold',fontcolor=dfcol)

# --------------------------------- # Power Spectra
ax = ax33

# Panel C Options
decadal_focus = True
obs_cutoff    = 10 # in years
obs_cutoff    = 1/(obs_cutoff*12)
dtmon_fix     = 60*60*24*30

# Set Xticks
if decadal_focus:
    xper            = np.array([20,10,5,2,1,0.5])
else:
    xper            = np.array([40,10,5,1,0.5])
xper_ticks      = 1 / (xper*12)

# Loop by experiments
for ii in range(nexps):
    if ii == id_era: # Use Default Color
        col_in = dfcol
    else:
        col_in = expcols[ii]
    
    # Get Variables
    plotspec        = metrics_out['specs'][ii] / dtmon_fix
    plotfreq        = metrics_out['freqs'][ii] * dtmon_fix
    CCs             = metrics_out['CCs'][ii] / dtmon_fix
    
    # Plot Cut Off Section for Obs
    if ii == id_era:
        iplot_hifreq = np.where(plotfreq > obs_cutoff)[0]
        ax.loglog(plotfreq,plotspec,label="",c=col_in,ls='dashed',lw=1.5)
        hiplotfreq     = plotfreq[iplot_hifreq]
        hiplotspec     = plotspec[iplot_hifreq]
        ax.loglog(hiplotfreq,hiplotspec,lw=2.5,label=expnames_long[ii],c=col_in,zorder=2)
        
    else:
        zorders = [2,9,3,3,]
        zorder = zorders[ii]
        
        # Plot the other spectra
        ax.loglog(plotfreq,plotspec,lw=2.5,label=expnames_long[ii],c=col_in,zorder=zorder)
        
    # Plot the 95% Confidence Interval (for stochastic model output)
    if ii > id_era:
        
        iim1 = ii-1 # Since ERA5 was skipped for MonteCarlo Testing
        
        plotspec1 = mc_specdicts[iim1]['specs'] / dtmon_fix
        plotfreq1 = mc_specdicts[iim1]['freqs'][0,:] * dtmon_fix
        
        bnds = np.quantile( plotspec1 ,[0.025,0.975],axis=0)
        
        ax.fill_between(plotfreq1,bnds[0],bnds[1],color=expcols[ii],alpha=0.15,zorder=1)
        #ax.loglog(plotfreq,bnds[0],ls='dotted',color='blue',label="95% Conf.")
        #ax.loglog(plotfreq,bnds[1],ls='dotted',color='blue')
    else:
        
        # Plot Confidence Interval (ERA5)
        alpha           = 0.05
        cloc_era        = [2e-2,6]
        dof_era         = metrics_out['dofs'][id_era]
        cbnds_era       = sut.calc_confspec(alpha,dof_era)
        ax.fill_between(plotfreq,cbnds_era[0]*plotspec,cbnds_era[1]*plotspec,color=expcols[ii],alpha=0.05,zorder=1)
        #proc.plot_conflog(cloc_era,cbnds_era,ax=ax,color=dfcol,cflabel=r"95% Confidence (ERA5)") #+r" (dof= %.2f)" % dof_era)
    
    #ax.loglog(plotfreq,CCs[:,0],ls='dotted',lw=0.5,c=expcols[ii])
    #ax.loglog(plotfreq,CCs[:,1],ls='dashed',lw=0.9,c=expcols[ii])

# Set Vertical Lines, Axes Labels
ax.set_xlim([xper_ticks[0],0.5])
ax.axvline([1/(6)],label="",ls='dotted',c='gray')
ax.axvline([1/(12)],label="",ls='dotted',c='gray')
ax.axvline([1/(5*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(2*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(10*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(20*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(40*12)],label="",ls='dotted',c='gray')

ax.set_xlabel("Frequency ($month^{-1}$)",fontsize=fsz_axis)
ax.set_ylabel("Power ($\degree C ^2 cycle \, per \, mon^{-1}$)",fontsize=fsz_axis)


# Add Legend
ax.legend(fontsize=fsz_legend_spectra,framealpha=0.5,edgecolor='none')

# Twin X-Axis for Period Labels
ax2 = ax.twiny()
ax2.set_xlim([xper_ticks[0],0.5])
ax2.set_xscale('log')
ax2.set_xticks(xper_ticks,labels=xper)
ax2.set_xlabel("Period (Years)",fontsize=fsz_axis)
for ax in [ax,ax2]:
    ax.tick_params(labelsize=fsz_ticks)

# Label Subplot
sut.label_sp(2,case='lower',alpha=0.15,ax=ax,y=1.10,x=-.1,fig=fig,fontsize=fsz_title,labelstyle="%s",
             weight='bold',fontcolor=dfcol)

# Output Figure
figname = "%sFigure03_SMHierarchy_Variance_%s.png" % (figpath,comparename)
plt.savefig(figname,dpi=300,transparent=transparent,bbox_inches='tight')

# =============================================================================
#%% Figure 4: CESM Hierarchy Variance
# =============================================================================

"""
Additional Notes on Figure 3:
"""


# Note: Data was loaded and area-averaged in the original script `cesm2_hierarchy_v_obs.py`

cesmnames           = ["FOM","MCOM","SOM"]

cesmnames_long      = ["Full Ocean Model (1800 years)",
                       "Multi-Column Ocean Model (300 years)",
                       "Slab Ocean Model (300 years)"]

cesmcols = [
            "deepskyblue",
            "forestgreen",
            "mediumorchid",
            ]


sst_cesm_raw = []
for cc in range(3):
    expname = cesmnames[cc]
    ncname = "%s/CESM2_%s_SST_SPGNE_Average.nc" % (dpath,expname)
    ds     = xr.open_dataset(ncname).TS.load()
    sst_cesm_raw.append(ds)

#% Preprocess Data (Detrend and deseason)
def preproc_cesm(ds):
    ds      = sut.fix_febstart(ds)
    dsa     = sut.xrdeseason(ds)
    dsa_dt  = sut.xrdetrend(dsa)
    return dsa_dt

sst_cesm_dt = [preproc_cesm(ds) for ds in sst_cesm_raw]

# Compute Necessary Metrics
nsmooths  = [250,100,100]
cssts     = [sst.data for sst in sst_cesm_dt]
cmetrics  = sut.compute_sm_metrics(cssts,nsmooth=nsmooths,lags=lags,detrend_acf=False)

cstd_metrics      = sut.calc_stds_sample(cssts)
cvratio           = cstd_metrics[-1]

#%% Compute Significance of Spectra

nsmooth = 4
mciter  = 10000
ex      = 0
pct     = 0.10

cesm_conts      = []
cesm_specdicts  = []
for ex in range(3):
    cesm_ts       = cssts[ex]
    
    mcdict        = sut.mcsampler(cesm_ts,ntime_era,mciter)
    csamples      = [mcdict['samples'][ii,:] for ii in range(mciter)]
    
    cesm_specdict = sut.quick_spectrum(csamples,[nsmooth,]*mciter,pct,return_dict=True,make_arr=True)
    cesm_cont     = sut.quick_spectrum([cesm_ts,],[nsmooths[ex],]*mciter,pct,return_dict=True)
    
    cesm_conts.append(cesm_cont)
    cesm_specdicts.append(cesm_specdict)

#%% Compute Significance of Stdev

# Get Standard Deviation Metrics
cstds          = cstd_metrics[1]
cstds_lp       = cstd_metrics[2]
cmonstds_spgne =  [ ds.groupby('time.month').std('time') for ds in sst_cesm_dt]

# Monte Carlo Computations for Standard Deviation Metrics
cesm_mcstds,cesm_mcstds_lp,cesm_mc_monstds = sut.mcsample_stdev_metrics(cssts,ntime_era,mciter)

#%% Setup Error Bars for barplot

cesm_errbar_var    = setup_errorbar(cesm_mcstds,cstds,era5_last=False,include_era5=False)
cesm_errbar_var_lp = setup_errorbar(cesm_mcstds_lp,cstds_lp,era5_last=False,include_era5=False)

#%% Plot Figure 4

# Set Figure Parameters
fsz_axis            = 16
fsz_ticks           = 16
fsz_legend          = 14
fsz_legend_spectra  = 16 # Reduced size for spectra plot
remove_topright     = True # True to remove top and right axlines of Panels A and B

# Set Up Labels (combine cesm_cols with ERA5)
expcols_bar         = ['gray',] + cesmcols # Set color to gray for ERA5
expcols_cesm        = ['k',]   + cesmcols
expnames_long_cesm  = [expnames_long[id_era],] + cesmnames_long
expnames_short_cesm = [expnames_short[id_era],] + cesmnames
vratio_cesm         = np.hstack([vratio[id_era],cvratio])

monstds_cesm        = [monstds[id_era],] + cmonstds_spgne

# Initialize Figure using Gridspec
fig             = plt.figure(figsize=(14,10))
gs              = gridspec.GridSpec(8,12)

ax11            = fig.add_subplot(gs[:3,:3],) # Barplot
ax22            = fig.add_subplot(gs[:3,4:11])  # Month Std
ax33            = fig.add_subplot(gs[4:,:11]) # Spectra

# --------------------------------- # Barplot
ax               = ax11

# Panel A Options

label_vratio         = False  # True to Label variance ratios
label_stds           = True   # True to label stdev on bars
ytks_var             = np.arange(0.2,1.3,0.2)

# Set Input Data
instd            = np.hstack([stds[0],cstds])
instd_lp         = np.hstack([stds_lp[0],cstds_lp])

# Create Labels
if label_vratio:
    xlabs           = ["%s\n%.2f" % (expnames_long_cesm[ii],vratio_cesm[ii])+"%" for ii in range(nexps)]
else:
    xlabs            = expnames_short_cesm

# Plot Bars
braw            = ax.bar(np.arange(nexps),instd,color=expcols_bar,yerr=cesm_errbar_var,
                         error_kw=dict(ecolor='darkgray',
                                       barsabove=True,
                                       capsize=5,marker="o",markersize=25,mfc='None',
                                       ))
blp             = ax.bar(np.arange(nexps),instd_lp,color=dfcol,yerr=cesm_errbar_var_lp,
                         error_kw=dict(ecolor='w',
                                       barsabove=True,
                                       capsize=5,marker="d",markersize=25,mfc='None',))

# Label Standard Deviations
if darkmode:
    upperlbl_col = 'lightgray'
else:
    upperlbl_col = 'gray'
if label_stds:
    ax.bar_label(braw,fmt="%.02f",c=upperlbl_col,fontsize=fsz_axis)
    ax.bar_label(blp,fmt="%.02f",c='w',fontsize=fsz_axis,label_type='center')

# Make Fake Legend and place on plot
colorsf = {'Raw':'gray','10-year low-pass':'k',}         
labels  = list(colorsf.keys())
handles = [plt.Rectangle((0,0),1,1, color=colorsf[label]) for label in labels]
ax.legend(handles, labels,fontsize=fsz_legend,framealpha=0,
          bbox_to_anchor=(0.04, 0.82, 1., .102))

# Set X and Y Labels, Ticks
ax.set_xticks(np.arange(nexps),labels=xlabs,fontsize=fsz_tick,rotation=45)
#ax.set_ylabel("$\sigma$(SST) [$\degree$C]",fontsize=fsz_axis)
ax.set_ylabel("SST\nStandard Deviation ($\degree$C)",fontsize=fsz_axis)
ax.set_ylim([0,1.2])
ax.set_yticks(ytks_var)
ax.tick_params(labelsize=fsz_tick)
if remove_topright:
    ax.spines[['right', 'top']].set_visible(False)

# Label Subplot
sut.label_sp(0,case='lower',alpha=0.15,ax=ax,y=1.10,x=-.45,fig=fig,fontsize=fsz_title,labelstyle="%s",
             weight='bold',fontcolor=dfcol)

#%


# --------------------------------- # Monthly Variance
ax              = ax22

# Looping by Experiment
for ex in range(nexps):
    
    # Plot Monthly standard deviation
    plotvar = monstds_cesm[ex]
    ax.plot(mons3,plotvar,label=expnames_long_cesm[ex],
            color=expcols_cesm[ex],lw=2.5,ls=expls[ex],marker="o",zorder=1)
    
    # Plot Confidence Interval (not for ERA5)
    if ex != id_era:
        exm1   = ex -1
        plotmc = cesm_mc_monstds[exm1] 
        bnds   = np.quantile(plotmc,[0.025,0.95],axis=0)
        ax.fill_between(mons3,bnds[0],bnds[1],color=expcols_cesm[ex],alpha=0.10,zorder=1)


# Set X and Y Labels, Ticks
#ax.set_ylabel("Monthly $\sigma(SST)$ [$\degree$C]",fontsize=fsz_axis)
ax.set_ylabel("Monthly SST\nStandard Deviation ($\degree$C)",fontsize=fsz_axis)
ax.set_xticklabels(mons3)
ax.set_xlim([0,11])
ax.set_ylim([0,1.2])
ax.set_yticks(np.arange(0.2,1.3,0.2))
ax.tick_params(labelsize=fsz_ticks)
if remove_topright:
    ax.spines[['right', 'top']].set_visible(False)

# Label Subplot
sut.label_sp(1,case='lower',alpha=0.15,ax=ax,y=1.10,x=-.15,fig=fig,fontsize=fsz_title,labelstyle="%s",
             weight='bold',fontcolor=dfcol)

# --------------------------------- # Power Spectra
ax = ax33

# Panel C Options
decadal_focus = True
obs_cutoff    = 10 # in years
obs_cutoff    = 1/(obs_cutoff*12)
dtmon_fix     = 60*60*24*30

# Set Xticks
if decadal_focus:
    xper            = np.array([20,10,5,2,1,0.5])
else:
    xper            = np.array([40,10,5,1,0.5])
xper_ticks      = 1 / (xper*12)

# Loop by experiments
for ii in range(nexps):
    
    if ii == id_era: # Use Default Color
        col_in = dfcol
        
        # Get Variables
        plotspec        = metrics_out['specs'][id_era] / dtmon_fix
        plotfreq        = metrics_out['freqs'][id_era] * dtmon_fix
        CCs             = metrics_out['CCs'][id_era] / dtmon_fix
        
    else:
        
        iim1            = ii - 1 # Since ERA5 was skipped in CESM spectra calculations
        col_in          = expcols_cesm[ii]
    
        # Get Variables
        plotspec        = cmetrics['specs'][iim1] / dtmon_fix
        plotfreq        = cmetrics['freqs'][iim1] * dtmon_fix
        CCs             = cmetrics['CCs'][iim1] / dtmon_fix
    
    
    # Plot Cut Off Section for Obs
    if ii == id_era:
        iplot_hifreq = np.where(plotfreq > obs_cutoff)[0]
        ax.loglog(plotfreq,plotspec,label="",c=col_in,ls='dashed',lw=1.5)
        hiplotfreq     = plotfreq[iplot_hifreq]
        hiplotspec     = plotspec[iplot_hifreq]
        ax.loglog(hiplotfreq,hiplotspec,lw=2.5,label=expnames_long_cesm[ii],c=col_in,zorder=2)
        
    else:
        zorders = [9,9,2,2,]
        zorder = zorders[ii]
        # Plot the other spectra
        ax.loglog(plotfreq,plotspec,lw=2.5,label=expnames_long_cesm[ii],c=col_in,zorder=zorder)
        
    # Plot the 95% Confidence Interval (for stochastic model output)
    if ii > id_era:
        
        iim1 = ii-1 # Since ERA5 was skipped for MonteCarlo Testing
        
        plotspec1 = cesm_specdicts[iim1]['specs'] / dtmon_fix
        plotfreq1 = cesm_specdicts[iim1]['freqs'][0,:] * dtmon_fix
        
        bnds       = np.quantile( plotspec1 ,[0.025,0.975],axis=0)
        
        ax.fill_between(plotfreq1,bnds[0],bnds[1],color=expcols_cesm[ii],alpha=0.15,zorder=1)
        #ax.loglog(plotfreq,bnds[0],ls='dotted',color='blue',label="95% Conf.")
        #ax.loglog(plotfreq,bnds[1],ls='dotted',color='blue')
    else:
        
        # Plot Confidence Interval (ERA5)
        alpha           = 0.05
        cloc_era        = [2e-2,6]
        dof_era         = metrics_out['dofs'][id_era]
        cbnds_era       = sut.calc_confspec(alpha,dof_era)
        ax.fill_between(plotfreq,cbnds_era[0]*plotspec,cbnds_era[1]*plotspec,color=expcols_cesm[ii],alpha=0.05,zorder=1)
        #proc.plot_conflog(cloc_era,cbnds_era,ax=ax,color=dfcol,cflabel=r"95% Confidence (ERA5)") #+r" (dof= %.2f)" % dof_era)
    
    #ax.loglog(plotfreq,CCs[:,0],ls='dotted',lw=0.5,c=expcols[ii])
    #ax.loglog(plotfreq,CCs[:,1],ls='dashed',lw=0.9,c=expcols[ii])

# Set Vertical Lines, Axes Labels
ax.set_xlim([xper_ticks[0],0.5])
ax.axvline([1/(6)],label="",ls='dotted',c='gray')
ax.axvline([1/(12)],label="",ls='dotted',c='gray')
ax.axvline([1/(5*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(2*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(10*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(20*12)],label="",ls='dotted',c='gray')
ax.axvline([1/(40*12)],label="",ls='dotted',c='gray')

ax.set_xlabel("Frequency ($month^{-1}$)",fontsize=fsz_axis)
ax.set_ylabel("Power ($\degree C ^2 cycle \, per \, mon^{-1}$)",fontsize=fsz_axis)

# Add Legend 
ax.legend(fontsize=fsz_legend_spectra,framealpha=0.5,edgecolor='none',loc='lower left')

# Twin X-Axis for Period Labels 
ax2 = ax.twiny()
ax2.set_xlim([xper_ticks[0],0.5])
ax2.set_xscale('log')
ax2.set_xticks(xper_ticks,labels=xper)
ax2.set_xlabel("Period (Years)",fontsize=fsz_axis)
for ax in [ax,ax2]:
    ax.tick_params(labelsize=fsz_ticks)

# Label Subplot
sut.label_sp(2,case='lower',alpha=0.15,ax=ax,y=1.10,x=-.1,fig=fig,fontsize=fsz_title,labelstyle="%s",
             weight='bold',fontcolor=dfcol)

# Output Figure
figname = "%sFigure04_CESMHierarchy_Variance_%s.png" % (figpath,comparename)

plt.savefig(figname,dpi=300,transparent=transparent,bbox_inches='tight')



#%% Supplemental Figures


#%% Plot t2 for other simulations (Figure S1)

pmesh           = False
fsz_axis        = 22
fsz_tick        = 18
fsz_title       = 28

# 
id_sm           = 0 # Index of Stochastic Model to plot
# 0 = With Re-emergence, 1 = No Re-emergence, 2 = Slab-Like for comparename=Draft05_ReverseOrder

bbox_spgne      = [-40,-15,52,62]

#Contour Settings
cints_t2_lab    = np.arange(1,6,1)
cints_t2        = np.arange(1,4.2,0.2)

# Bounding Box
bbsel           = [-65, -5, 45, 65] # [-40, -12, 50, 62]
centlat         = 55
centlon         = -35
fix_lon         = np.arange(-60,10,10)
fix_lat         = np.arange(45,70,5)

mld_cbticks     = np.arange(0,600,100)
t2_cbticks      = np.arange(1,5,1)

fig,axs,_       = sut.init_orthomap(1,2,bbsel,figsize=(20,12),centlat=centlat,centlon=centlon)

plot_ex = [1,2]


for a,ax in enumerate(axs):
    ii              = a
    ax              = sut.add_coast_grid(ax,bbox=bbsel,line_color='k',
                                        fill_color="lightgray",fontsize=fsz_tick,
                                        fix_lon=fix_lon,fix_lat=fix_lat)
    
    
    ex          = plot_ex[a]
    
    plotvar     = t2_sm[ex].T #* blowupmask_apply.data #* dsmaskera.data
    lon         = acfs_sm[ex].lon
    lat         = acfs_sm[ex].lat
    cmap        = cm.devon_r#'cmo.dense'#cm.acton_r
    clab        = "Decorrelation Timescale $T_2$ (Years)"

    cints       = cints_t2#np.arange(0,4.2,0.2)#np.arange(0,5.5,0.5)#np.arange(0,36,3)
    cints_lab   = cints_t2_lab #cints[::2]
    
    cbticks     = t2_cbticks
        
    
    cf      = ax.contourf(lon,lat,plotvar,
                          levels=cints,
                          transform=proj,cmap=cmap,zorder=-1,extend='both')
    
    cl      = ax.contour(lon,lat,plotvar,
                          levels=cints_lab,linewidths=.55,colors='w',
                          transform=proj,zorder=-1)
    ax.clabel(cl,fontsize=fsz_tick)

    
    ax.set_title(expnames_long[ex+1],fontsize=fsz_axis)
    
    # Plot the Median Sea Ice Concentration
    plotvar = dsice
    icel    = ax.contour(plotvar.longitude,plotvar.latitude,plotvar.siconc_winter_median,levels=[0.15,],
                      colors='cyan',linewidths=4,transform=proj,linestyles='dotted')
    
    
    bb = sut.plot_box(bbox_spgne,ax=ax,color='purple',linewidth=4.5,proj=proj)
    sut.label_sp(a,case='lower',ax=ax,fig=fig,fontsize=fsz_title,labelstyle="%s",weight='bold')
    
    
cb = sut.hcbar(cf,ax=axs.flatten(),fontsize=22,pad=0.0001,fraction=0.040)
cb.set_label(clab,fontsize=fsz_axis)
cb.set_ticks(cbticks)


figname = figpath + "FigureS1_T2_SM_%s.png" % comparename 
plt.savefig(figname,dpi=300,bbox_inches='tight',transparent=transparent)

#%% Figure S2 Autocorrelation For Each Month

#alpha = 0.001

plot_im         = np.roll(np.arange(12),1)
fig,axs         = plt.subplots(4,3,constrained_layout=True,figsize=(10,8))

alphalist       = list(map(chr, range(97, 123)))
alphalist_upper = [s.upper() for s in alphalist]
dofs_eff        = np.zeros((nexps,12)) * np.nan

fsz_splab = 12

handles = [] # For legend
for mm in range(12):
    
    ax     = axs.flatten()[mm]
    kmonth = plot_im[mm]
    lab    = "%s. %s" % (alphalist[mm],mons3[kmonth])
    
    sut.label_sp(lab,case='lower',alpha=0.15,ax=ax,x=0,fig=fig,fontsize=fsz_splab,labelstyle="%s",
                 weight='bold',fontcolor=dfcol,usenumber=True)
    
    if mm == 10:
        ax.set_xlabel("Lag (month)")
    
    # Plot ACFs (copied from above) ===========================================
    for ex in range(nexps):
        if ex == id_era:
            col_in = dfcol
        else:
            col_in = expcols[ex]
            
        zorders = [9,9,2,2,]
        zorder  = zorders[ex]
        
        plotvar = metrics_out['acfs'][kmonth][ex]
        
        lll = ax.plot(lags,plotvar,
                label=expnames_long[ex],color=col_in,ls=expls[ex],lw=2.5,zorder=zorder)
        
        # Calcualate Confidence Interval
        if use_neff:
            plotvar_mon = ssts[ex][kmonth::12]
            dof_in      = sut.calc_dof(plotvar_mon,calc_r1=True,r1_in=None)
            dofs_eff[ex,kmonth] = dof_in
        else:
            dof_in = len(ssts[ex])/12
        cflag = sut.calc_conflag(plotvar,conf,2,dof_in)
        if ex == 2:
            if darkmode:
                alpha = 0.15
            else:
                alpha = 0.05
        else:
            alpha = 0.10
        
        ax.fill_between(lags,cflag[:,0],cflag[:,1],alpha=alpha,color=col_in,zorder=3)
        if mm == 0:
            handles.append(lll[0]) # Need to take first element of list for handle
        
    # =========================================================================
    
    ax.set_xlim([0,60])
    ax.set_ylim([-0.25,1.25])
    ax.set_xticks(np.arange(0,66,6))
    ax.set_yticks(np.arange(0,1.25,0.25))
    ax.axhline([0],ls='dashed',c='k',lw=0.55)


axs[0,0].set_ylabel("Correlation")

fig.legend(handles,expnames_long,bbox_to_anchor=(0.5, 1.03),loc ='center',ncol=4)

figname = "%sFigureS00_ACF_%s_neff%i_AllMonths.png" % (figpath,comparename,use_neff)
plt.savefig(figname,dpi=150,transparent=transparent,bbox_inches='tight') 







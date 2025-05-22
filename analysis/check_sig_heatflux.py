#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Check points with insignificant or positive heat flux feedback
in the SPGNE Box

Copied Upper Section of [prep_inputs_obs]

Created on Fri May  9 11:06:08 2025

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
import tqdm
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

figpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20250513/"

proc.makedir(figpath)

#%% Load Ice Edge and other variables to plot

icemask = dl.load_mask("ERA5").mask
plotmask = xr.where(np.isnan(icemask),0,icemask)


# # Load Sea Ice Masks
dpath_ice   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/reanalysis/proc/NATL_proc_obs/"
nc_masks    = dpath_ice + "OISST_ice_masks_1981_2020.nc"
ds_masks    = xr.open_dataset(nc_masks).load()
ds_ice_era5 = dl.load_mask("ERA5").mask
dsice_era5_plot = xr.where(np.isnan(ds_ice_era5),0,1)

# Load AVISO
dpath_aviso = dpath_ice + "proc/"
nc_adt      = dpath_aviso + "AVISO_adt_NAtl_1993_2022_clim.nc"
ds_adt      = xr.open_dataset(nc_adt).load()
cints_adt   = np.arange(-100, 110, 10)

# Load Mixed-Layer Depth
dpath_mld   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/"
nc_mld      = dpath_mld + "MIMOC_regridERA5_h_pilot.nc"
ds_mld      = xr.open_dataset(nc_mld).load().h
cints_mld   = np.arange(0,510,10)

# Make a plotting function
def plot_ice_ssh(fsz_ticks=20-2,label_ssh=False):
    # Requires ds_masks and ds_adt to be loaded
    ax = plt.gca()
    
    # # Plot Sea Ice
    plotvar = dsice_era5_plot#ds_masks.mask_mon
    cl = ax.contour(plotvar.lon, plotvar.lat,
                    plotvar, colors="cyan",
                    linewidths=2, transform=proj, levels=[0, 1], zorder=1)
    
    # Plot the SSH
    plotvar = ds_adt.mean('time')
    cl = ax.contour(plotvar.lon, plotvar.lat, plotvar.adt*100, colors="k",alpha=0.8,
                    linewidths=0.75, transform=proj, levels=cints_adt)
    if label_ssh:
        ax.clabel(cl,fontsize=fsz_ticks)
    return None



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


"""
signame = "noPositive"#"noPositive"#"pilot" #"noPositive" # 
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

# dof was set above

# Compute and Apply Mask
st = time.time()
dampingmasked, freq_success, sigmask = scm.prep_HF(hff, rsst, rflx,
                                                   setdict['p'], setdict['tails'], dof, setdict['method'],
                                                   returnall=True)  # expects, [month x lag x lat x lon], should generalized with ensemble dimension?
print("Completed significance testing in %.2fs" % (time.time()-st))



dampingout = dampingmasked[:,setdict['sellags'],:,:].squeeze()
dampingout = xr.where(np.isnan(dampingout),0.,dampingout)

#%% Check Points in SPGNE Region
fsz_title = 24 
bbsim     = [-40,-15,52,62]
bbspgne   = [-40,-15,52,62]
ilag      = 0

im = 'min'


for im in range(12):
    if im is 'min':
        plotvar   = hff.isel(lag=ilag).min('month') #dampingout.isel(month=imon)#
    else:
        plotvar   = hff.isel(lag=ilag,month=im)
    
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
    
    
    # Plot Additional Features
    
    # Region Bounding Box
    viz.plot_box(bbsim,ax=ax,color='limegreen',linewidth=2.5,proj=proj)
    
    
    if im is "min":
        # plot SSH
        plot_ice_ssh()
        
        # Plot Mixed-layer depth
        plotvar = ds_mld.max('mon')
    else:
        
        # # Plot the SSH
        # cints_adt2 = np.arange(-200,205,5)
        # plotvar = ds_adt.isel(time=im)#('time')
        # cl = ax.contour(plotvar.lon, plotvar.lat, plotvar.adt*100, colors="k",alpha=0.8,
        #                 linewidths=0.75, transform=proj, levels=cints_adt2)
            
            
        # Plot Mixed-layer depth
        plotvar = ds_mld.isel(mon=im)
    
    # Plot Mixed-layer depth
    clmld = ax.contour(plotvar.lon,plotvar.lat,plotvar,linewidths=0.75,
                        levels=cints_mld,transform=proj,colors='dimgray')
    
    ax.clabel(clmld)
    # lonf = -49
    # latf = -60
    # ax.plot(lonf,latf,marker="x",c='black',
    #         markersize=25,transform=proj)
    
    
    # Print some information
    isneg_reg = proc.sel_region_xr(isneg,bbsim)
    total_neg = isneg_reg.data.sum((0,1))
    total_pts = np.prod(np.array(np.shape(isneg_reg)))
    ptcount   = "%s/%s pts (%.2f" % (total_neg,total_pts,total_neg/total_pts*100) + "%) are negative"
    
    if im is "min":
        title = "ERA5 Minimum Heat Flux Feedback, Lag %02i\n%s" % (ilag+1,ptcount)
        figname = "%sHFF_Check_SPGNE_ERA5_%s_%s_monMIN_lag%i.png" % (figpath,flxname,signame,ilag+1)
    else:
        title = "ERA5 %s Heat Flux Feedback, Lag %02i\n%s" % (mons3[im],ilag+1,ptcount)
        figname = "%sHFF_Check_SPGNE_ERA5_%s_%s_mon%02i_lag%i_mld.png" % (figpath,flxname,signame,im+1,ilag+1)
    ax.set_title(title,fontsize=fsz_title)
    
    
    plt.savefig(figname,dpi=150,bbox_inches='tight')
    
    
#%% Compute the P-Value, and plot it as contour lines (with HFF)


ilag      = 0
im        = 1
rho       = rflx.isel(lag=ilag,month=im).data

pval      = proc.calc_pval_rho(rho,123)
cints_p   = np.arange(0,1.05,0.05)

fig,ax,_  = viz.init_orthomap(1,1,bboxplot=bbplot2,figsize=(12,8),centlon=-25)

# Plot the Heat Flux Damping
plotvar   = hff.isel(lag=ilag,month=im) #dampingout.isel(month=imon)#
isneg     = xr.where(plotvar<0,1,0)
pcm       = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                    transform=proj,
                    cmap='cmo.balance',vmin=-35,vmax=35,zorder=-2)

# Plot Negative Points
viz.plot_mask(plotvar.lon,plotvar.lat,isneg.T,reverse=True,
              geoaxes=True,proj=proj,ax=ax,color='k',markersize=0.2)

plotvar   = plotmask
ax.contour(plotvar.lon,plotvar.lat,plotvar,levels=[0,1],transform=proj,colors='cyan',zorder=-1)

ax        = viz.add_coast_grid(ax,bbox=bbplot2,proj=proj,fill_color="k")
viz.plot_box(bbsim,ax=ax,color='limegreen',linewidth=2.5,proj=proj)

cl = ax.contour(plotvar.lon,plotvar.lat,pval,
                    transform=proj,colors="k",levels=cints_p,linewidths=0.75)

ax.clabel(cl)


#%% Calculate P=value for all months and lags

pval       = proc.calc_pval_rho(rflx,123)

#%%Plot just the pvalue (max for all months)
max_all_mon = False
im          = 1


iimax = 0
for im in range(12):
    
    if max_all_mon:
        plot_pval = pval[:,ilag,...].max(0)
    else:
        plot_pval = pval[im,ilag,...]

    
    
    if max_all_mon:
        if iimax > 0:
            continue
    fig,ax,_  = viz.init_orthomap(1,1,bboxplot=bbplot2,figsize=(12,8),centlon=-25)
    
    # Plot the Heat Flux Damping
    plotvar   = hff.isel(lag=ilag,month=im) #dampingout.isel(month=imon)#
    
    pcm       = ax.pcolormesh(plotvar.lon,plotvar.lat,plot_pval,
                        transform=proj,
                        cmap='cmo.tempo',vmin=0,vmax=1,zorder=-2)
    
    
    ax        = viz.add_coast_grid(ax,bbox=bbplot2,proj=proj,fill_color="k")
    viz.plot_box(bbsim,ax=ax,color='limegreen',linewidth=2.5,proj=proj)
    
    cb = viz.hcbar(pcm,ax=ax,)
    if max_all_mon:
        clabel = "%i-lag Cross-Correlation P-Value (Max), DOF-%i" % (ilag+1,dof)
    else:
        clabel = "%s %i-lag Cross-Correlation P-Value (Max), DOF-%i" % (mons3[im],ilag+1,dof)
    cb.set_label(clabel,fontsize=fsz_title)
    cl = ax.contour(plotvar.lon,plotvar.lat,plot_pval,
                        transform=proj,colors="white",levels=np.arange(0,1.1,0.1),linewidths=0.75)
    
    clbl = ax.clabel(cl)
    viz.add_fontborder(clbl,c="k",w=2.5)
    
    if max_all_mon:
        figname = "%sPvalue_Crosscorrelation_lag%02i_monMAX.png" % (figpath,ilag)
    else:
        figname = "%sPvalue_Crosscorrelation_lag%02i_mon%02i.png" % (figpath,ilag,im+1)
    plt.savefig(figname,dpi=150,bbox_inches='tight')
    
    iimax += 1
    

#%% Load raw heat flux and and SST that was used for the calculation

ensorem_path = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/hff_calc/anom/"
vnames = ['sst','qnet']

dsanom = []
for vv in range(2):
    vname  = vnames[vv]
    ncname = "%sERA5_%s_NAtl_1979to2024_detrend1_ENSOrem_lag1_pcs3_monwin3.nc" % (ensorem_path,vname)
    ds     = xr.open_dataset(ncname)[vname].load()
    dsanom.append(ds)
    

#%% Get Point Index

def get_closest_indices(lonf,latf,lon,lat,debug=False,return_lin=False):
    # Return x and y indices of closest point to lonf,latf using meshgrid
    # Given x-coordinate [lon] and y-coordinate [lat]
    # Option to return linear indices [return_lin]
    
    # Make a meshgrid
    xx,yy   = np.meshgrid(lon,lat)
    
    # Find the closest point
    londiff = np.abs(xx - lonf)
    latdiff = np.abs(yy - latf)
    diffsum = latdiff + londiff
    idxlin  = np.nanargmin(diffsum)
    if return_lin:
        return idxlin
    
    # Get the indices
    nlat    = len(lat)
    nlon    = len(lon)
    idx,idy = np.unravel_index(idxlin,[nlat,nlon])
    
    # Visualize to make sure
    if debug:
        fig,ax = plt.subplots(1,1)
        pcm    = ax.pcolormesh(xx,yy,diffsum,cmap='inferno')
        cb = fig.colorbar(pcm,ax=ax,pad=0.01,fraction=0.025)
        cb.set_label("Difference from Target Lat/Lon")
        ax.scatter(lon[idx],lat[idy],c='white',marker='x',label="Found Point")
        ax.legend()
        
    return idx,idy


lonf  = -35
latf  = 57
lon   = dsreg[0].lon.data
lat   = dsreg[0].lat.data
debug = True
xxx,yyy = get_closest_indices(lonf,latf,lon,lat,debug=True)


#%%




#%%


# # 
# np.where(xx == lonf)



# def get_point_index(lonf,latf,lon,lat):
#     nlat,nlon = len(lat),len(lon)
    
#     latidx      = np.arange(nlat)
#     lonidx      = np.arange(nlon)
#     lin_idx     = np.arange(nlat*nlon)
    
#     #
#     #test = np.ravel_multi_index([latidx,lonidx])
    
    
    
    
    
#     test = np.unravel_index(lin_idx,[nlat,nlon])
    
#     # Get Original Index of Lat/Lon in Unraveled Index
#     klat,klon   = proc.find_latlon(lonf,latf,lon,lat)
    
#     arr        = np.array([[klat,],[klon,]])
#     flat_index = np.ravel_multi_index(arr, (nlat,nlon,))
    
#     #lin_idx     = np.
    
    
    
    
    
#     #lat = dsreg.lat.data
#     #lon = dsreg.lon.data



#%% Recalculate Lag Covariance

bbsim = [-50,0,50,65] #[-40,-15,52,62]

# Select Region
dsreg = [proc.sel_region_xr(ds,bbsim) for ds in dsanom]

# Reshape to [yr x mon x space]
ntime,nlat,nlon = dsreg[0].shape
nspace          = nlat*nlon
nyr             = int(ntime/12)
invars          = [ds.data.reshape(nyr,12,nspace) for ds in dsreg]
sst,flx         = invars

# Also select region and convert to arr
isneg_reg = proc.sel_region_xr(isneg,bbsim).data
isneg_reg = isneg_reg.reshape(nspace)

monwin   = 3
lags     = np.arange(12)
nlags    = len(lags)
covall  = np.zeros((nlags,12,nspace)) * np.nan
autoall = np.zeros((nlags,12,nspace)) * np.nan

covleadall = covall.copy()


# Analyze Lags
for l in range(nlags):
    lag = lags[l]
    for m in range(12):
        lm = m-lag # Get Lag Month
        
        #
        flxmon = scm.indexwindow(flx,m,monwin,combinetime=True,verbose=False)
        sstmon = scm.indexwindow(sst,m,monwin,combinetime=True,verbose=False)
        sstlag = scm.indexwindow(sst,lm,monwin,combinetime=True,verbose=False)
        
        # Calculate covariance ----
        cov     = proc.covariance2d(flxmon,sstlag,0)
        autocov = proc.covariance2d(sstmon,sstlag,0)
        
        covall[l,m,:]  = cov.copy()
        autoall[l,m,:] = autocov.copy()
        
        
        flxlag  = scm.indexwindow(flx,lm,monwin,combinetime=True,verbose=False)
        covlead = proc.covariance2d(flxlag,sstmon,0)
        covleadall[l,m,:] = covlead.copy()

#%% Scatterplot for a particular Point


lonreg = dsreg[0].lon.data
latreg = dsreg[0].lat.data
lonf   = -28
latf   = 61
pt     = get_closest_indices(lonf,latf,lonreg,latreg,return_lin=True)#252

m      = 0
lag    = 1
lm     = m-lag # Get Lag Month
monwin = 3

locfn,loctitle = proc.make_locstring(lonf,latf,fancy=True)


#pt   = 242

# This section is the calcualion for HFF --------------------------------------
flxmon = scm.indexwindow(flx,m,monwin,combinetime=True,verbose=False)
sstmon = scm.indexwindow(sst,m,monwin,combinetime=True,verbose=False)
sstlag = scm.indexwindow(sst,lm,monwin,combinetime=True,verbose=False)

cov     = proc.covariance2d(flxmon,sstlag,0)*-1
autocov = proc.covariance2d(sstmon,sstlag,0)
hff     = cov/autocov
# --------------------------------------


# Calculate and plot line of best fit ----------------------
xin     = flxmon[:,pt]
yin     = sstlag[:,pt]
coeffs,newmodel,res = proc.polyfit_1d(xin,yin,1)

numer = np.nansum(res**2)
denom = np.nansum((yin - yin.mean())**2)
R2    = 1-numer/denom 

xplot     = np.linspace(np.nanmin(xin),np.nanmax(xin),100)
plotmodel = xplot * coeffs[0] + coeffs[1]

fig,ax = plt.subplots(1,1,constrained_layout=True)
ax.scatter(flxmon[:,pt],sstlag[:,pt],)


ax.plot(xplot,plotmodel,lw=1,c='r',
        label="Linear Fit ($R^2$: %.4f, Slope: %.2e, Intercept: %.2e)" % (R2,coeffs[0],coeffs[1]))
ax.legend()

ax.set_xlabel("Qnet [W/m2]")
ax.set_ylabel("SST [degC]")

ax.axhline([0],c="k",lw=0.55)
ax.axvline([0],c="k",lw=0.55)

title_top = "Basemonth %s, Lag %02i @ %s" % (mons3[m-1],lag,loctitle)
title_bot = "Cov = %.3f, Autocov = %.3f, HFF = %.2f" % (cov[pt],autocov[pt],hff[pt])

ax.set_title("%s\n%s"  % (title_top,title_bot))

ax.set_xlim([-200,200])
ax.set_ylim([-3,3])

#%% Plot Heat Flux Feedback of these points

hfftest = covall/autoall

im      = 0

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,4.5))

for pt in tqdm.tqdm(range(nspace)):
    
    if isneg_reg[pt] == True:
        c = "blue"
    else:
        c = "red"
    
    plotvar = hfftest[:,im,pt] * -1#covall[:,im,pt]
    ax.plot(lags,plotvar,c=c,alpha=0.05)
    
    
    


ax.set_xlim([0,12])
ax.set_xticks(lags)
ax.axhline([0],c="k",lw=0.55)



#%% Plot Lead Lag at each point, color coding pos/neg values

mons3 = proc.get_monstr()
im    = 0
for im in range(12):
    
    fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,4.5))
    
    ll_pos = []
    ll_neg = []
    
    for pt in tqdm.tqdm(range(nspace)):
        
       
        
        plotlag  = covall[:,im,pt] * -1
        plotlead = np.flip(covleadall[:,im,pt]) * -1
        
        plotleadlag = np.hstack([plotlead[:-1],plotlag])
        
        if isneg_reg[pt] == True:
            c = "blue"
            ll_neg.append(plotleadlag)
            
        else:
            c = "red"
            ll_pos.append(plotleadlag)
            
            
        
        
        ax.plot(lags,plotlag,c=c,alpha=0.05,label="")
        ax.plot(leads,plotlead,c=c,alpha=0.05,label="")
        
        
    ll_neg = np.array(ll_neg)
    ll_pos = np.array(ll_pos)
    
    muneg    = ll_neg.mean(0)
    mupos    = ll_pos.mean(0)
    stdneg   = ll_neg.std(0)
    stdpos   = ll_pos.std(0)
    
    ax.plot(leadlags,muneg,label="-$\lambda^a$, n=%i" % ll_neg.shape[0],c='midnightblue',marker="o")
    ax.plot(leadlags,ll_pos.mean(0),label="+$\lambda^a$, n=%i" % ll_pos.shape[0],c='saddlebrown',marker="d")
    
    
    ax.fill_between(leadlags,muneg-stdneg,muneg+stdneg,label="",color='midnightblue',alpha=0.2)
    ax.fill_between(leadlags,mupos-stdpos,mupos+stdpos,label="",color='saddlebrown',alpha=0.2)
    
    ax.set_ylim([-20,20])
    
    ax.set_xlim([-12,12])
    ax.set_xticks(leadlags)
    ax.axhline([0],c="k",lw=0.55)
    ax.axvline([1],c="cornflowerblue",lw=1.5,ls='dashed')
    ax.axvline([0],c="k",lw=0.55)
    
    ax.set_title("%s $SST$-$Q_{net}$ Lead-Lag Covariance" % mons3[im])
    ax.set_xlabel("<< Qnet Leads | SST Leads >>")
    ax.set_ylabel("Covariance (W/m$^2$ $\degree$C)")
    ax.legend()
    
    savename = "%sSST_FLX_Covariance_SPGNE_mon%02i.png" % (figpath,im+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')


#%%

hfftestin = hfftest[1,1,:].reshape(nlat,nlon)
lat       = dsreg[0].lat.data
lon       = dsreg[0].lon.data

fig,ax,_  = viz.init_orthomap(1,1,bboxplot=bbplot2,figsize=(12,8),centlon=-25)
ax        = viz.add_coast_grid(ax,bbox=bbplot2,proj=proj,fill_color="k")

ax.pcolormesh(lon,lat,hfftestin,cmap='cmo.balance',vmin=-35,vmax=35,transform=proj)


viz.plot_box(bbsim,ax=ax,color='limegreen',linewidth=2.5,proj=proj)


#%% Reshape Lead/Lags into DataArray

leadlags       = np.hstack([-1*np.flip(lags)[:-1],lags])
leadsflipcovar = np.flip(covleadall[1:,:,:],0) #* -1
lagscovar      = covall #* -1
leadlagscovar  = np.concatenate([leadsflipcovar,lagscovar],axis=0) * -1

leadlagscovar_rs = leadlagscovar.reshape(23,12,nlat,nlon)

# Place into Data Array
coords              = dict(lags=leadlags,mon=np.arange(1,13,1),lat=dsreg[0].lat,lon=dsreg[0].lon)
da_leadlagscovar    = xr.DataArray(leadlagscovar_rs,coords=coords,dims=coords,name='sst_flx_covar') 
#%% Try lead/lag joining

fig,ax   = plt.subplots(1,1,constrained_layout=True,figsize=(8,4.5))

leadlags = np.hstack([-1*np.flip(lags)[:-1],lags]) #* -1
leads    = -1 * np.flip(lags)
pt       = 0
plotlag  = covall[:,im,pt] * -1
plotlead = np.flip(covleadall[:,im,pt]) * -1

# Check to make sure they are overalapping
ax.plot(lags,plotlag)
ax.plot(leads,plotlead)
ax.plot(leadlags,leadlagscovar[:,im,pt],color='red',ls='dashed')

#%% For a given month, plot the lead/lag pattern

def monfromlag(basemonth_index,lag,debug=True):
    # Get month index [im], given the [basemonth_index], and [lag] from basemonth.
    mons3   = proc.get_monstr()
    monin   = ((basemonth_index+1) + lag) % 12
    im      = monin - 1
    print("Lag is %02i, Month is %02i, Indexed: %s" % (lag,monin,mons3[monin-1]))
    return im

basemonth_index = 1
plotlags        = np.arange(-11,12,1)

for il in range(len(plotlags)):
    
    plotlag  = plotlags[il]
    
    im       = monfromlag(basemonth_index,plotlag,debug=True)
    
    # In script version (non-function)
    #monin   = ((basemonth_index+1) + plotlag)%12
    #im      = monin-1
    #print("Lag is %02i, Month is %02i, Indexed: %s" % (plotlag,monin,mons3[monin-1]))
    
    fig,ax,_  = viz.init_orthomap(1,1,bboxplot=bbplot2,figsize=(12,8),centlon=-25)
    ax        = viz.add_coast_grid(ax,bbox=bbplot2,proj=proj,fill_color="k")
    
    plotlag = plotlags[il]
    plotvar = da_leadlagscovar.sel(lags=plotlag).isel(mon=im)
    
    isneg   = xr.where(plotvar<0,1,0)
    
    pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                            cmap='cmo.balance',vmin=-25,vmax=25)
    
    ax.set_title("%s Lag: %02i\nBasemonth: %s" % (mons3[im],plotlag,mons3[basemonth_index]),fontsize=fsz_title)
    
    # Plot Mask for negative values
    viz.plot_mask(plotvar.lon,plotvar.lat,isneg.T,reverse=True,
                  geoaxes=True,proj=proj,ax=ax,color='k',markersize=0.2)
    
    # Plot some additional figures ---------------------------
    # Plot the SPGNE Box
    viz.plot_box(bbspgne,ax=ax,color='limegreen',linewidth=2.5,proj=proj)
    
    # Plot the SSH for a given month
    plotvar = ds_adt.isel(time=im)
    cl = ax.contour(plotvar.lon, plotvar.lat, plotvar.adt*100, colors="k",alpha=0.8,
                    linewidths=0.75, transform=proj, levels=cints_adt)
    
    figname = "%sSST_FLX_Lag_Covar_Map_kmonth%02i_lag%02i.png" % (figpath,basemonth_index,plotlag)
    plt.savefig(figname,dpi=150,bbox_inches='tight')
    
    
    
#%% Examine Scatterplot




#%%






#dampingout[np.isnan(dampingout)] = 0

# #%% Sanity Check for the damping...

# imon    = 1
# ilag    = 0

# for imon in range(12):
#     plotvar = hff.isel(lag=ilag,month=imon) #dampingout.isel(month=imon)#
#     #isneg    = xr.where(plotvar<0,1,0)
#     plotmask = sigmask[imon,ilag]
    
#     if np.any(np.isnan(plotmask)):
#         plotmask = xr.where(np.isnan(plotmask),0,1)
#     else:
#         plotmask = plotmask#xr.where(plotmask==0,0,1)
    
#     bbsel   = [-80,0,50,65]
    
#     fsz_title = 24
    
    
#     # Get Count of points in neative region
#     bbsim   = [-40,-15,52,62]
#     isneg_reg,_,_ = proc.sel_region(plotmask.T,plotvar.lon.data,plotvar.lat.data,bbsim)
#     isneg_reg     = np.logical_not(isneg_reg,dtype=bool)
#     total_neg = isneg_reg.sum((0,1))
#     total_pts = np.prod(np.array(np.shape(isneg_reg)))
#     ptcount   = "%s/%s pts (%.2f" % (total_neg,total_pts,total_neg/total_pts*100) + "%) are set to 0"
    
    
#     fig,axs,_ = viz.init_orthomap(2,1,bboxplot=bbsel,figsize=(12,8))
    
#     # Plot Before with significance dots
#     ax = axs[0]
#     ax = viz.add_coast_grid(ax,bbox=bbsel,proj=proj)
#     pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
#                         transform=proj,
#                         cmap='cmo.balance',vmin=-35,vmax=35)
#     viz.plot_mask(plotvar.lon,plotvar.lat,plotmask.T,geoaxes=True,proj=proj,ax=ax,color='gray',markersize=0.2)
#     ax.set_title('Before Mask (Lag %i, Month %s)\n%s' % (ilag+1,mons3[imon],ptcount),fontsize=fsz_title)
    
#     # PLot After Masking
#     ax = axs[1]
#     plotvar = dampingout.isel(month=imon)
#     ax = viz.add_coast_grid(ax,bbox=bbsel,proj=proj)
#     pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
#                         transform=proj,
#                         cmap='cmo.balance',vmin=-35,vmax=35)
#     viz.plot_mask(plotvar.lon,plotvar.lat,plotmask.T,geoaxes=True,proj=proj,ax=ax,color='gray',markersize=0.2)
    
    
#     # Set Title and Colorbar
#     ax.set_title('After Mask',fontsize=fsz_title)
#     cb = viz.hcbar(pcm,ax=axs.flatten())
    
    
#     for ax in axs:
        
#         # Plot Bounding Box for analysis
#         viz.plot_box(bbsim,ax=ax,color='limegreen',linewidth=2.5,proj=proj)
    
#     figname = "%sHFF_Check_ERA5_%s_%s_mon%02i_lag%i.png" % (figpath,flxname,signame,imon+1,ilag+1)
#     plt.savefig(figname,dpi=150,bbox_inches='tight')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize T2 (Annually Averaged) using output from annual_acf_pointwise.py

Created on Fri Oct 17 15:14:10 2025

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

#%% Load Sea Ice Concentration File

dpath   = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
ncice   = "%sERA5_IceMask_Global_1979_2024_Median15.nc" % (dpath)
dsice   = xr.open_dataset(ncice)

dsmask  = dsice.median_mask_15pct
dsmask  = dsmask.rename({'latitude':'lat','longitude':'lon'})
dsmask  = proc.lon360to180_xr(dsmask)

#%% Load the autocorrelation functions

acfpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"

comparename = "smioOct2025"
smnc        = acfpath + "SM_SST_ORAS5_avg_GMSST_EOF_usevar_NATL_lag00to40_JFM_ensALL.nc"
eranc       = acfpath + "ERA5_NAtl_1979to2025_lag00to40_JFM_ensALL.nc"

sm_monnc    = acfpath + "SM_SST_ORAS5_avg_GMSST_EOFmon_usevar_NATL_lag00to40_JFM_ensALL.nc"

exppath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/SST_ORAS5_avg_GMSST_EOFmon_usevar_NATL/Metrics/"
blowupnc    = exppath + "Blowup_Points.nc"

dsera       = xr.open_dataset(eranc).load()
dssm        = xr.open_dataset(smnc).load()
dssm_mon    = xr.open_dataset(sm_monnc).load()
dsblowup    = xr.open_dataset(blowupnc).load()


# Also load the monthly version

sm_monthly_nc = acfpath + "SM_SST_ORAS5_avg_GMSST_EOFmon_usevar_NATL_lag00to60_ALL_ensALL.nc"
dsmonthly_nc  = xr.open_dataset(sm_monthly_nc).load()


figpath     = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20251106/"
proc.makedir(figpath)

#%% Also Load the MLD

ncmld       = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/MIMOC_regridERA5_h_pilot.nc"
dsmld       = xr.open_dataset(ncmld).load()

#%%
fpath="/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
fn = "T2_SSTwint_regOutGM_arYW_AICr_remPolyOrder0_1950-2024.nc"
ds2 = xr.open_dataset(fpath+fn).load()

lonm         = np.linspace(0,360,ds2.XC.shape[1])#ds2.XC#inds[0].lon
latm         = np.linspace(-90,90,ds2.YC.shape[0])#ds2.YC#inds[0].lat
coords     = dict(lat=latm,lon=lonm,)
ds2_martha  = xr.DataArray(np.flip(ds2.T2.data,0),coords=coords,dims=coords,name="T2")
ds2_martha  = proc.lon360to180_xr(ds2_martha)
ds2_martha = proc.sel_region_xr(ds2_martha,[-80,0,0,90])

#ds2_martha   = ds2_martha.reindex(lat=list(reversed(ds2_martha.lat)))



#%% Calculate T2

t2_era = proc.calc_T2(dsera.acf.sel(lags=slice(0,10)).squeeze() * dsmask,axis=-1,verbose=True,ds=True)
t2_sm  = proc.calc_T2(dssm.acf.squeeze() * dsmask,axis=-1,verbose=True,ds=True)
t2_sm_mon =  proc.calc_T2(dssm_mon.acf.squeeze(),axis=-1,verbose=True,ds=True)

t2_monthly = proc.calc_T2(dsmonthly_nc.acf.squeeze(),axis=-1,ds=True)


_,dsmasksm  = proc.resize_ds([dssm.acf,dsmask])
_,dsmaskera  = proc.resize_ds([dsera.acf.isel(lags=0).squeeze(),dsmask])


#%% Visualize T2

fsz_tick = 12
fsz_axis = 14

inds = [dsera,dssm]

t2s  = [t2_era,t2_sm.mean(0)]

import cmcrameri.cm as cm

proj = ccrs.PlateCarree()

plt.style.use('default')


cmap            = cm.acton_r
pmesh           = True
cints           = np.arange(0,4.4,0.4)#np.arange(0,36,3)
cints_t2_major  = cints#np.arange(0,48,6)

centlat         = 50
bbsel           = [-80, -0, 40, 65] # [-40, -12, 50, 62]
fig,axs,_       = viz.init_orthomap(1,2,bbsel,figsize=(20,12),centlat=centlat)

for a,ax in enumerate(axs):
    ii              = a
    ax              = viz.add_coast_grid(ax,bbox=bbsel,line_color='dimgray',
                                        fill_color="lightgray",fontsize=18)
    
    
    # Plot the t2
    plotvar = t2s[a].T
    lon     = inds[a].lon
    lat     = inds[a].lat
    
    if pmesh:
        # cf = ax.pcolormesh(lon,lat,plotvar,
        #                       vmin=cints[0],vmax=cints[-1],
        #                       transform=proj,cmap=cmap,zorder=-1)
        
        
        cl = ax.contourf(lon,lat,plotvar,
                              levels=cints_t2_major,
                              transform=proj,cmap=cmap,zorder=-1)
        # cl = ax.contour(lon,lat,plotvar,
        #                       levels=cints_t2_major,
        #                       transform=proj,colors='w',zorder=-1)
        
        #ax.clabel(cl,fontsize=fsz_tick+6)
        
    else:
        cf      = ax.contourf(plotvar.lon,plotvar.lat,plotvar,
                              levels=cints,
                              transform=proj,cmap=cmap,zorder=-1)
    
    # Plot the Median Sea Ice Concentration
    plotvar = dsice
    icel = ax.contour(plotvar.longitude,plotvar.latitude,plotvar.siconc_winter_median,levels=[0.15,],
                      colors='cyan',linewidths=2,transform=proj,linestyles='dotted')
        
cb = viz.hcbar(cl,ax=axs.flatten(),fontsize=22,pad=0.0001,fraction=0.045)
cb.set_label("Decorrelation Timescale $T^2$ [Years]",fontsize=fsz_axis)  


#%% Just Plot the Case for the Stochastic Model

plot_mon        = True
bbox_spgne      = [-40,-15,52,62]
cints_lab       = np.arange(0,6,1)
cints           = np.arange(0,4.2,0.2)#np.arange(0,36,3)
fig,ax,_        = viz.init_orthomap(1,1,bbsel,figsize=(18,12),centlat=centlat)
ax              = viz.add_coast_grid(ax,bbox=bbsel,line_color='dimgray',
                                    fill_color="lightgray",fontsize=18)

# Plot the T2
if plot_mon:
    plotvar = t2_sm_mon.mean(0).T
else:
    plotvar = t2s[a].T
lon     = inds[a].lon
lat     = inds[a].lat
cf      = ax.contourf(lon,lat,plotvar,
                      levels=cints_t2_major,
                      transform=proj,cmap=cmap,zorder=-1,extend='both')
cl = ax.contour(lon,lat,plotvar,
                      levels=cints_lab,
                      transform=proj,colors="w",zorder=-1,linewidths=0.5)


clbl    = ax.clabel(cl,fontsize=fsz_tick+6)
#viz.add_fontborder(clbl)

# Plot the Median Sea Ice Concentration
plotvar = dsice
icel    = ax.contour(plotvar.longitude,plotvar.latitude,plotvar.siconc_winter_median,levels=[0.15,],
                  colors='cyan',linewidths=2,transform=proj,linestyles='dotted')

cb      = viz.hcbar(cf,ax=ax,fontsize=22,pad=0.0001,fraction=0.045)
cb.set_label("Decorrelation Timescale $T^2$ [Years]",fontsize=24)  

bb = viz.plot_box(bbox_spgne,ax=ax,color='magenta',linewidth=2.5,proj=proj)
 
figname = figpath + "Stochastic_Model_AllMonthReg_T2_plotmon%i.png" % plot_mon
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Plot, side by side, the Max Wintertime MLD and the Persistence of ERA5 SST Anomalies

pmesh           = False
fsz_axis        = 22
fsz_tick        = 18
fig,axs,_       = viz.init_orthomap(1,2,bbsel,figsize=(20,12),centlat=centlat)

for a,ax in enumerate(axs):
    ii              = a
    ax              = viz.add_coast_grid(ax,bbox=bbsel,line_color='dimgray',
                                        fill_color="lightgray",fontsize=18)
    
    # Plot the t2
    if a == 0:
        plotvar     = t2s[a].T #* dsmaskera.data
        lon         = inds[a].lon
        lat         = inds[a].lat
        cmap        = cm.devon_r#'cmo.dense'#cm.acton_r
        cints       = np.arange(0,5.5,0.5)#np.arange(0,36,3)
        cints_lab   = cints[::2]
        clab        = "Decorrelation Timescale $T^2$ [Years]"
    elif a == 1:
        plotvar = dsmld.h.max('mon')# * dsmasksm
        lon     = plotvar.lon
        lat     = plotvar.lat
        cmap    = 'cmo.ice_r'
        cints   = np.arange(0,525,25)
        cints_lab = cints[::4]
        clab    = "Maximum Seasonal Mixed-Layer Depth [meters]"
    
    cf      = ax.contourf(lon,lat,plotvar,
                          levels=cints,
                          transform=proj,cmap=cmap,zorder=-1)
    
    cl      = ax.contour(lon,lat,plotvar,
                          levels=cints_lab,linewidths=.55,colors='w',
                          transform=proj,zorder=-1)
    ax.clabel(cl,fontsize=fsz_tick)
    
    # clbl    = ax.clabel(cf,levels=cints[:-2:2])
    # viz.add_fontborder(clbl,w=1)
    
    cb = viz.hcbar(cf,ax=ax,fontsize=22,pad=0.0001,fraction=0.045)
    cb.set_label(clab,fontsize=fsz_axis)  
    
    # Plot the Median Sea Ice Concentration
    plotvar = dsice
    icel    = ax.contour(plotvar.longitude,plotvar.latitude,plotvar.siconc_winter_median,levels=[0.15,],
                      colors='cyan',linewidths=4,transform=proj,linestyles='dotted')
    
    
    bb = viz.plot_box(bbox_spgne,ax=ax,color='purple',linewidth=4.5,proj=proj)
    
figname = figpath + "ERAT2_MLD_DraftPlot.png" 
plt.savefig(figname,dpi=150,bbox_inches='tight',transparent=True)


#%% 2025.10.24, Check Monthly vs Annual Average values

kmonths     = [0,1,2]
dsann       = dssm_mon.acf.squeeze()
dsmon       = dsmonthly.acf.squeeze()
mons3       = proc.get_monstr()
lonf        = -30
latf        = 58
lw          = 2.5

crop_lagyear = 6

moncols = ['orange',"limegreen","magenta"]
locfn,locstr = proc.make_locstring(lonf,latf)

fig,ax = plt.subplots(1,1,figsize=(12.5,4.5,),constrained_layout=True)

xtks = dsmon.lags.data[::3]


acfsel = []
for k,kmonth in enumerate(kmonths):
    plotvar = proc.selpt_ds(dsmon.isel(mons=kmonth),lonf,latf).mean('ens')
    t2plot  = proc.calc_T2(plotvar) 
    
    ax.plot(plotvar.lags,plotvar,label="Monthly ACF %s ($T_2=%.2f$)" % (mons3[kmonth],t2plot/12),c=moncols[k],lw=lw)
    
    acfsel.append(plotvar.data)

t2plot = proc.calc_T2(np.array(acfsel).mean(0)) 
ax.plot(plotvar.lags,np.array(acfsel).mean(0),color="k",label="Mean Monthly ACF (JFM), ($T_2=%.2f$)" % (t2plot/12),lw=lw,ls='dashed')





plotvar = proc.selpt_ds(dsann.sel(lags=slice(0,crop_lagyear)),lonf,latf).mean('ens') # Slice to 5 lags
annlags = plotvar.lags.data * 12
t2plot  = proc.calc_T2(plotvar) 
ax.plot(annlags,plotvar,color='blue',label="JFM Annual ACF, ($T_2=%.2f$)" % (t2plot),marker="o",lw=lw)


ax.legend()
ax.set_xlim([0,60])
ax.set_ylim([0,1.1])
ax.set_xticks(xtks)
ax.set_xlabel("Lag (Months)")
ax.set_ylabel("Correlation")
ax.grid(True,ls='dotted',color='gray')

ax.set_title("ACF Comparison @ %s" % locstr)

figname = figpath + "ACF_Comparison_%s.png" % locfn
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Draft 2 Version: plot, side by side, the Max Wintertime MLD and the Persistence of ERA5 SST Anomalies

pmesh           = False
fsz_axis        = 22
fsz_tick        = 18
fig,axs,_       = viz.init_orthomap(1,2,bbsel,figsize=(20,12),centlat=centlat)

for a,ax in enumerate(axs):
    ii              = a
    ax              = viz.add_coast_grid(ax,bbox=bbsel,line_color='dimgray',
                                        fill_color="lightgray",fontsize=18)
    
    # Plot the t2
    if a == 0:
        plotvar = t2s[a].T #* dsmaskera.data
        lon     = inds[a].lon
        lat     = inds[a].lat
        cmap    = cm.acton_r
        cints   = np.arange(0,5.5,0.5)#np.arange(0,36,3)
        cints_lab = cints[::2]
        clab    = "Decorrelation Timescale $T^2$ [Years]"
    elif a == 1:
        plotvar = dsmld.h.max('mon')# * dsmasksm
        lon     = plotvar.lon
        lat     = plotvar.lat
        cmap    = "cmo.ice_r"
        cints   = np.arange(0,525,25)
        cints_lab = cints[::4]
        clab    = "Maximum Seasonal Mixed-Layer Depth [meters]"
    
    cf      = ax.contourf(lon,lat,plotvar,
                          levels=cints,
                          transform=proj,cmap=cmap,zorder=-1)
    
    cl      = ax.contour(lon,lat,plotvar,
                          levels=cints_lab,linewidths=.55,colors='w',
                          transform=proj,zorder=-1)
    ax.clabel(cl,fontsize=fsz_tick)
    
    # clbl    = ax.clabel(cf,levels=cints[:-2:2])
    # viz.add_fontborder(clbl,w=1)
    
    cb = viz.hcbar(cf,ax=ax,fontsize=22,pad=0.0001,fraction=0.045)
    cb.set_label(clab,fontsize=fsz_axis)  
    
    # Plot the Median Sea Ice Concentration
    plotvar = dsice
    icel    = ax.contour(plotvar.longitude,plotvar.latitude,plotvar.siconc_winter_median,levels=[0.15,],
                      colors='yellow',linewidths=4,transform=proj,linestyles='dotted')
    
    
    bb = viz.plot_box(bbox_spgne,ax=ax,color='magenta',linewidth=2.5,proj=proj)
    
figname = figpath + "ERAT2_MLD_DraftPlot.png" 
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Demarcate points blown up

nanids           = dsblowup.min('run').nanid
blowupmask       = xr.where(nanids > 0,1,0)
blowupmask_apply = xr.where(nanids > 0,np.nan,1)

#%% Fix this fucking image

fig,ax,_       = viz.init_orthomap(1,1,bbsel,figsize=(24,12),centlat=centlat,centlon=centlon)
ax             = viz.add_coast_grid(ax,bbox=bbsel,line_color='k',
                                    fill_color="lightgray",fontsize=fsz_tick,
                                    fix_lon=fix_lon,fix_lat=fix_lat)


plotvar     = ds2_martha#.T2 #ds2.T2.data #t2s[0].T #* dsmaskera.data
#plotvar     = proc.sel_region_xr(plotvar,[-80,0,0,65])
lon         = ds2_martha.lon#np.linspace(0,360,ds2.XC.shape[1])#ds2.XC#inds[0].lon
lat         = ds2_martha.lat#np.linspace(90,-90,ds2.YC.shape[0])#ds2.YC#inds[0].lat
 
pcm         = ax.contourf(lon,lat,plotvar,transform=proj,cmap=cmap)



#%%
pmesh           = False
fsz_axis        = 22
fsz_tick        = 18
fsz_title       = 28

use_marthas_t2 = True

#Contour Settings
cints_t2_lab    = np.arange(0,6,1)
cints_t2        = np.arange(0,4.2,0.2)

# Bounding Box
bbsel           = [-65, -5, 45, 65] # [-40, -12, 50, 62]
centlat         = 55
centlon         = -35
fix_lon         = np.arange(-65,5,5)
fix_lat         = np.arange(45,70,5)

mld_cbticks     = np.arange(0,600,100)
t2_cbticks      = np.arange(0,5,1)

fig,axs,_       = viz.init_orthomap(1,3,bbsel,figsize=(24,12),centlat=centlat,centlon=centlon)

for a,ax in enumerate(axs):
    ii              = a
    ax              = viz.add_coast_grid(ax,bbox=bbsel,line_color='k',
                                        fill_color="lightgray",fontsize=fsz_tick,
                                        fix_lon=fix_lon,fix_lat=fix_lat)
    
    # if a != 2:
    #     continue
    # Plot the t2
    if a == 1:
        if use_marthas_t2:
            plotvar     = ds2_martha#.T2 #ds2.T2.data #t2s[0].T #* dsmaskera.data
           #plotvar     = proc.sel_region_xr(plotvar,[-80,0,0,65])
            lon         = ds2_martha.lon#np.linspace(0,360,ds2.XC.shape[1])#ds2.XC#inds[0].lon
            lat         = ds2_martha.lat#np.linspace(90,-90,ds2.YC.shape[0])#ds2.YC#inds[0].lat
        else:
            plotvar     = t2s[0].T #* dsmaskera.data
            lon         = inds[0].lon
            lat         = inds[0].lat
        
        cmap        = cm.devon_r#'cmo.dense'#cm.acton_r
        cints       = cints_t2 #np.arange(0,5.5,0.5)
        cints_lab   = cints_t2_lab#cints[::2]
        clab        = "Decorrelation Timescale $T^2$ [Years]"
        cbticks     = t2_cbticks
        
    elif a == 0:
        plotvar     = dsmld.h.max('mon')# * dsmasksm
        lon         = plotvar.lon
        lat         = plotvar.lat
        cmap        = 'cmo.ice_r'
        cints       = np.arange(0,525,25)
        cints_lab   = cints[::4]
        clab        = "Maximum Climatological Mixed-Layer Depth [meters]"
        
        cbticks     = mld_cbticks
        
        
    elif a == 2:
        
        plotvar     = t2s[1].T * blowupmask_apply.data #* dsmaskera.data
        lon         = inds[1].lon
        lat         = inds[1].lat
        cmap        = cm.devon_r#'cmo.dense'#cm.acton_r
        cints       = cints_t2#np.arange(0,4.2,0.2)#np.arange(0,5.5,0.5)#np.arange(0,36,3)
        cints_lab   = cints_t2_lab #cints[::2]
        clab        = "Decorrelation Timescale $T^2$ [Years]"
        
        cbticks     = t2_cbticks
    

        

    
    cf      = ax.contourf(lon,lat,plotvar,
                          levels=cints,
                          transform=proj,cmap=cmap,zorder=-1,extend='both')
    
    cl      = ax.contour(lon,lat,plotvar,
                          levels=cints_lab,linewidths=.55,colors='w',
                          transform=proj,zorder=-1)
    ax.clabel(cl,fontsize=fsz_tick)
    
    # clbl    = ax.clabel(cf,levels=cints[:-2:2])
    # viz.add_fontborder(clbl,w=1)
    
    if a == 0:
        cb = viz.hcbar(cf,ax=ax,fontsize=22,pad=0.0001,fraction=0.040)
        cb.set_label(clab,fontsize=fsz_axis)
        cb.set_ticks(cbticks)
    elif a == 2:
        cb = viz.hcbar(cf,ax=axs[1:].flatten(),fontsize=22,pad=0.0001,fraction=0.040)
        cb.set_label(clab,fontsize=fsz_axis)
        cb.set_ticks(cbticks)
    
    # Plot the Median Sea Ice Concentration
    plotvar = dsice
    icel    = ax.contour(plotvar.longitude,plotvar.latitude,plotvar.siconc_winter_median,levels=[0.15,],
                      colors='cyan',linewidths=4,transform=proj,linestyles='dotted')
    
    # # Plot Blow Up Points (for Stochastic Model Simulation)
    # if a == 2:
    #     plotvar = blowupmask
    #     viz.plot_mask(plotvar.lon,plotvar.lat,plotvar.T,
    #                   reverse=True,
    #                   geoaxes=True,proj=proj,ax=ax,marker="x",markersize=10,color='hotpink')
    
    bb = viz.plot_box(bbox_spgne,ax=ax,color='purple',linewidth=4.5,proj=proj)
    viz.label_sp(a,ax=ax,fig=fig,fontsize=fsz_title)
    

figname = figpath + "ERA_SM_T2_MLD_Draft02Plot.png" 
plt.savefig(figname,dpi=150,bbox_inches='tight',transparent=True)


plotvar     = ds2.T2.data #t2s[0].T #* dsmaskera.data
lon         = np.linspace(0,360,ds2.XC.shape[1])#ds2.XC#inds[0].lon
lat         = np.linspace(-90,90,ds2.YC.shape[0])#ds2.YC#inds[0].lat

plt.pcolormesh(lon,lat,plotvar)


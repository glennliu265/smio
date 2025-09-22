#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare stochastic model inputs

based on "viz_inputs_paper_draft"

Created on Mon Sep 22 09:56:53 2025

@author: gliu

"""

import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import os
import matplotlib.patheffects as PathEffects
from cmcrameri import cm

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine     = "Astraeus"

# First Load the Parameter File
cwd         = os.getcwd()
sys.path.append(cwd+ "/..")
sys.path.append("../")
import reemergence_params as rparams

# Paths and Load Modules
machine     = "Astraeus"
pathdict    = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])
from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx

# Set needed paths
figpath     = pathdict['figpath']
proc.makedir(figpath)
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
rawpath     = pathdict['raw_path']

# ----------------------------------
#%% User Edits
# ----------------------------------

# Indicate the experiment
expname_sss         = "SSS_Revision_Qek_TauReg"#"#"SSS_Draft03_Rerun_QekCorr"#SSS_EOF_LbddCorr_Rerun_lbdE_neg" #"SSS_EOF_Qek_LbddEnsMean"#"SSS_EOF_Qek_LbddEnsMean"
expname_sst         = "SST_Revision_Qek_TauReg"#"SST_Draft03_Rerun_QekCorr"#"SST_EOF_LbddCorr_Rerun"


# Constants
dt                  = 3600*24*30 # Timestep [s]
cp                  = 3850       # 
rho                 = 1026    #`23      # Density [kg/m3]
B                   = 0.2        # Bowen Ratio, from Frankignoul et al 1998
L                   = 2.5e6      # Specific Heat of Evaporation [J/kg], from SSS model document

fsz_tick            = 18
fsz_title           = 24
fsz_axis            = 22


debug       = False



#%% Add some functions to load (and convert) inputs

def stdsqsum(invar,dim):
    return np.sqrt(np.nansum(invar**2,dim))

def stdsq(invar):
    return np.sqrt(invar**2)

def stdsqsum_da(invar,dim):
    return np.sqrt((invar**2).sum(dim))

def convert_ds(invar,lat,lon,):
    
    if len(invar.shape) == 4: # Include mode
        nmode = invar.shape[0]
        coords = dict(mode=np.arange(1,nmode+1),mon=np.arange(1,13,1),lat=lat,lon=lon)
    else:
        coords = dict(mon=np.arange(1,13,1),lat=lat,lon=lon)
    
    return xr.DataArray(invar,coords=coords,dims=coords)

def compute_detrain_time(kprev_pt):
    
    detrain_mon   = np.arange(1,13,1)
    delta_mon     = detrain_mon - kprev_pt#detrain_mon - kprev_pt
    delta_mon_rev = (12 + detrain_mon) - kprev_pt # Reverse case 
    delta_mon_out = xr.where(delta_mon < 0,delta_mon_rev,delta_mon) # Replace Negatives with 12+detrain_mon
    delta_mon_out = xr.where(delta_mon_out == 0,12.,delta_mon_out) # Replace deepest month with 12
    delta_mon_out = xr.where(kprev_pt == 0.,np.nan,delta_mon_out)
    
    return delta_mon_out


#%% Plotting Params

mpl.rcParams['font.family'] = 'Avenir'
bboxplot                    = [-80,0,20,65]
proj                        = ccrs.PlateCarree()
#lon                         = daspecsum.lon.values
#lat                         = daspecsum.lat.values
mons3                       = proc.get_monstr()


plotver = "rev1" # [sub1]

#%% Load BSF and Ice Mask (copied from compare_detrainment_damping)

bsf      = dl.load_bsf()

# Load Land Ice Mask
icemask  = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")

# Resize
bsf,icemask    = proc.resize_ds([bsf,icemask])
bsf_savg = proc.calc_savg_mon(bsf)

#
mask = icemask.MASK.squeeze()
mask_plot = xr.where(np.isnan(mask),0,mask)#mask.copy()

mask_apply = icemask.MASK.squeeze().values
#mask_plot[np.isnan(mask)] = 0


# Load Gulf Stream
ds_gs   = dl.load_gs()
ds_gs   = ds_gs.sel(lon=slice(-90,-50))
ds_gs2  = dl.load_gs(load_u2=True)

# Load velocities
ds_uvel,ds_vvel = dl.load_current()
tlon  = ds_uvel.TLONG.mean('ens').data
tlat  = ds_uvel.TLAT.mean('ens').data

# Load Region Information
regionset       = "SSSCSU"
regiondicts     = rparams.region_sets[regionset]
bboxes          = regiondicts['bboxes']
regions_long    = regiondicts['regions_long']
rcols           = regiondicts['rcols']
rsty            = regiondicts['rsty']
regplot         = [0,1,3]
nregs           = len(regplot)


# Get Point Info
pointset        = "PaperDraft02"
ptdict          = rparams.point_sets[pointset]
ptcoords        = ptdict['bboxes']
ptnames         = ptdict['regions']
ptnames_long    = ptdict['regions_long']
ptcols          = ptdict['rcols']
ptsty           = ptdict['rsty']


#%% Check and Load Params (copied from run_SSS_basinwide.py on 2024-03-04)

# Load the parameter dictionary
expparams_byvar     = []
paramset_byvar      = []
convdict_byvar      = []
convda_byvar        = []
for expname in [expname_sst,expname_sss]:
    
    print("Loading inputs for %s" % expname)
    
    expparams_raw   = np.load("%s%s/Input/expparams.npz" % (output_path,expname),allow_pickle=True)
    
    expparams       = scm.repair_expparams(expparams_raw)
    
    # Get the Variables (I think only one is really necessary)
    #expparams_byvar.append(expparams.copy())
    
    # Load Parameters
    paramset = scm.load_params(expparams,input_path)
    inputs,inputs_ds,inputs_type,params_vv = paramset
    
    
    # Convert to the same units
    convdict                               = scm.convert_inputs(expparams,inputs,return_sep=True)
    
    # Get Lat/Lon
    ds = inputs_ds['h']
    lat = ds.lat.data
    lon = ds.lon.data
    
    # Convert t22o DataArray
    varkeys = list(convdict.keys())
    nk = len(varkeys)
    conv_da = {}
    for nn in range(nk):
        #print(nn)
        varkey = varkeys[nn]
        invar  = convdict[varkey]
        conv_da[varkey] =convert_ds(invar,lat,lon)
        
    
    # Append Output
    expparams_byvar.append(expparams)
    paramset_byvar.append(paramset)
    convdict_byvar.append(convdict)
    convda_byvar.append(conv_da)

# --------------------------------------
#%% Load kprev and compute convert lbd-d
# --------------------------------------

lbdd_sst    = paramset_byvar[0][1]['lbd_d']
lbdd_sss    = paramset_byvar[1][1]['lbd_d']

# Compute Detrainment Times
ds_kprev    = xr.open_dataset(input_path + "mld/CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc")
delta_mon   = xr.apply_ufunc(
        compute_detrain_time,
        ds_kprev.kprev,
        input_core_dims=[['mon']],
        output_core_dims=[['mon']],
        vectorize=True,
        )

lbdd_sst_conv = -delta_mon / np.log(lbdd_sst)
lbdd_sss_conv = -delta_mon / np.log(lbdd_sss)

#%% Load (or compute) the SST Evaporation Feedback

# Load lbd_e
lbd_e    = xr.open_dataset(input_path + "forcing/" + expparams_byvar[1]['lbd_e']).lbd_e.load() # [mon x lat x lon]
lbd_e    = proc.sel_region_xr(lbd_e,bbox=expparams_byvar[1]['bbox_sim'])

# Convert [sec --> mon]
lbd_emon = lbd_e * dt
#lbd_emon = lbd_emon.transpose('lon','lat','mon')#.values


# --------------------------------------
#%% Load Raw Variables for Comparison
# --------------------------------------
mvpath      = rawpath + "monthly_variance/"
ncnames_raw = [
    
    #"CESM1LE_SST_NAtl_19200101_20050101_bilinear_stdev.nc",
    #"CESM1LE_SSS_NAtl_19200101_20050101_bilinear_stdev.nc",
    "CESM1_HTR_FULL_Fprime_timeseries_nomasklag1_nroll0_NAtl_stdev.nc",
    "CESM1_HTR_FULL_Eprime_timeseries_LHFLXnomasklag1_nroll0_NAtl_stdev.nc",
    "CESM1LE_PRECTOT_NAtl_19200101_20050101_stdev.nc",
    "CESM1LE_Qek_SST_NAtl_19200101_20050101_bilinear_stdev.nc",
    "CESM1LE_Qek_SSS_NAtl_19200101_20050101_bilinear_stdev.nc"
    
    ]
rawvarname = ["Fprime","LHFLX","PRECTOT","Qek","Qek"]
dsmonvar_all = [xr.open_dataset(mvpath + ncnames_raw[ii])[rawvarname[ii]].load() for ii in range(len(rawvarname))]


# <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0>
#%% Plot the Output (Detrainment Damping Only)                         |
# <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0>

if plotver == "rev1":
    
    fsz_title       = 42 
    fsz_axis        = 32 
    fsz_tick        = 25 
    figsize         = (28,15)
    clab            = r"Subsurface Memory Timescale [$\tau^d$,months]"
    figname         = "%sDeepDamping" % (figpath)
    
else: # sub1
    fsz_title       = 42 # was 24 before
    fsz_axis        = 32 # was 22 before
    fsz_tick        = 26 # was 20 before
    figsize         = (28,15)
    clab            = r"Deep Decorrelation Timescale [$\tau^d$,months]"
    figname         = "%sInputs_Deep_Damping.png" % (figpath)

selmons         = [[6,7,8],[9,10,11],[0,1,2]]
plotvars        = [lbdd_sst_conv,lbdd_sss_conv]
plotvars_corr   = [lbdd_sst,lbdd_sss]
vlms            = [0,60]
cints_corr      = np.arange(0,1.1,.1)
cmap            = 'inferno'
fig,axs,mdict   = viz.init_orthomap(2,3,bboxplot=bboxplot,figsize=figsize)
ylabs_dd        = [u"$\lambda^d_T$" + " (SST)", u"$\lambda^d_S$" + " (SSS)"]
#ylabs_dd        = ["SST","SSS"]
ii              = 0

tau_cints = np.arange(0,66,6)
for vv in range(2):
    
    for mm in range(3):
        
        ax = axs[vv,mm]
        selmon = selmons[mm]
        
        ax = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=fsz_tick,
                                fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
        
        if vv == 0:
            ax.set_title(proc.mon2str(selmon),fontsize=fsz_title)
        if mm == 0:
            viz.add_ylabel(ylabs_dd[vv],ax=ax,fontsize=fsz_title,x=-.065,y=0.6)
        
        # Plot the Timescales
        plotvar = plotvars[vv].isel(mon=selmon).mean('mon') * mask
        pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                                transform=proj,vmin=vlms[0],vmax=vlms[1],cmap=cmap)
        # pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,
        #                         transform=proj,levels=tau_cints,cmap=cmap)
        
        # Plot the Correlation
        plotvar = plotvars_corr[vv].isel(mon=selmon).mean('mon') * mask
        cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                                transform=proj,levels=cints_corr,
                                colors="lightgray",linewidths=1.5)
        cl_lab = ax.clabel(cl,fontsize=fsz_tick,colors='k')
        [tt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')]) for tt in cl_lab]
        
        # Add other features
        # Plot Ice Mask
        ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=1,
                   transform=mdict['noProj'],levels=[0,1],zorder=-1)
        
        # Plot Gulf Stream Position
        gss = ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=1.75,c='k',ls='dashdot')
        gss[0].set_path_effects([PathEffects.withStroke(linewidth=4, foreground='lightgray')])
                             
        # Label Subplot
        viz.label_sp(ii,alpha=0.75,ax=ax,fontsize=fsz_title,y=1.08,x=-.02)# y = y=1.08
        ii+=1

cb = viz.hcbar(pcm,ax=axs.flatten())
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label(clab,fontsize=fsz_axis)
#fig.colorbar(pcm,ax=ax)

savename = figname
plt.savefig(savename,dpi=150,bbox_inches='tight')  


#%% Save output to plot in paper_figures_ginal
 # plotvars        = [lbdd_sst_conv,lbdd_sss_conv]
 # plotvars_corr   = [lbdd_sst,lbdd_sss]

da_lbdd_sst_conv = lbdd_sst_conv.rename("SST_taud")
da_lbdd_sss_conv = lbdd_sss_conv.rename("SSS_taud")
da_lbdd_sst = lbdd_sst.rename("SST_lbdd")
da_lbdd_sss = lbdd_sss.rename("SSS_lbdd")

da_lbdd_out = xr.merge([da_lbdd_sst_conv,da_lbdd_sss_conv,da_lbdd_sst,da_lbdd_sss])
revpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/revision_data/"
ncname_lbdd = "%sParameters_plot_subsurface_damping.nc" % revpath
edict       = proc.make_encoding_dict(da_lbdd_out)
da_lbdd_out.to_netcdf(ncname_lbdd,encoding=edict)

# <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0>
#%% Visualize Heat Flux Feedback and SST-Evaporation Feedback
# <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0>


fsz_title       = 32 #before
fsz_axis        = 26 #before
fsz_tick        = 22 #before

if plotver == "rev1":
    fsz_title       = 32 #before
    fsz_axis        = 32 #before
    fsz_tick        = 28 #before
    
    figsize = (30,14)
    dampvars_name   = ["Net Heat Flux Damping Timescale\n($\lambda^N$)","SST-Evaporation Feedback \n on SSS ($\lambda^e$)"]
    ylabs           = ["Net Heat Flux Damping ($\lambda^N$)","SST-Evaporation Feedback on SSS ($\lambda^e$)"]
    dampvars_units  = [r"[$\frac{W}{m^{-2} \,\, \degree C}]$",r'[$\frac{psu}{\degree C \,\, mon}$]' ] 
    
else:
    fsz_title       = 32 #before
    fsz_axis        = 26 #before
    fsz_tick        = 22 #before
    
    figsize = (30,10)
    dampvars_name   = ["Atmospheric Heat Flux Damping Timescale\n($\lambda^a$)","SST-Evaporation Feedback \n on SSS ($\lambda^e$)"]
    ylabs           = ["Net Heat Flux \n Feedback $(\lambda^a)$",dampvars_name[1]]
    #dampvars_units  = ['[Months]','[$psu \,\, (\degree C \,\, mon)^{-1}$]']
    dampvars_units  = ["[$W m^{-2} \degree C^{-1}$]",'[$psu \,\, (\degree C \,\, mon)^{-1}$]' ] 
    
plotdamp  = True


cints_hff       = np.arange(-40,44,4)
fig,axs,mdict = viz.init_orthomap(2,4,bboxplot=bboxplot,figsize=figsize)

lbd_expr = r'\rho c_p h \Delta t (\lambda^a)^{-1}'

dampvars        = [1/convda_byvar[0]['lbd_a'],lbd_emon,paramset_byvar[0][1]['lbd_a']]
dampvars_savg   = [proc.calc_savg(dv,ds=True,axis=0) for dv in dampvars]
#dampvars_units  = ['[Months]','[$psu \,\, (\degree C \,\, mon)^{-1}$]']
dampvars_cints  = [np.arange(0,48,3),np.arange(0,0.055,0.005)]
cints_taudamp   = np.arange(0,63,3)
dampvars_cmap   = ['cmo.amp_r','cmo.matter']
ii              = 0
for vv in range(2):
    
    cints  = dampvars_cints[vv]
    
    
    for ss in range(4):
        
        ax = axs[vv,ss]
        ax = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=fsz_tick,
                                fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
        
        # Plot the Variable
        if vv == 0 and plotdamp:
            plotvar = dampvars_savg[vv].isel(season=ss) * mask
            plothff = dampvars_savg[2].isel(season=ss) #* mask
            
            # Plot Damping as Values
            pcm       = ax.contourf(plothff.lon,plothff.lat,plothff,
                                  transform=proj,levels=cints_hff,extend='both',cmap='cmo.balance')
            
            # Plot Timescale as Lines
            cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                                  transform=proj,levels=cints_taudamp,extend='both',colors="navy",linewidths=0.75)
            cl_lab  = ax.clabel(cl,fontsize=fsz_tick)
            [tt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')]) for tt in cl_lab]
            
            dunits = "$W m^{-2} \degree C^{-1}$"
        else:
            plotvar = dampvars_savg[vv].isel(season=ss) * mask
            pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,
                                  transform=proj,levels=cints,extend='both',cmap=dampvars_cmap[vv])
            
            cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                                  transform=proj,levels=cints,extend='both',colors="lightgray",linewidths=0.75)
            cl_lab = ax.clabel(cl,fontsize=fsz_tick,)
            [tt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='dimgrey')]) for tt in cl_lab]
            
            
            dunits = dampvars_units[vv]

        
        if vv == 0:
            ax.set_title(plotvar.season.data.item(),fontsize=fsz_axis)
            
        if ss == 0:
            if plotver == "sub1":
                viz.add_ylabel(ylabs[vv],ax=ax,fontsize=fsz_axis)
                
        
        # Plot Ice Mask
        ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=1,
                   transform=mdict['noProj'],levels=[0,1],zorder=-1)
        
        # Plot Gulf Stream Position
        gss = ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=1.75,c='k',ls='dashdot')
        #gss[0].set_path_effects([PathEffects.withStroke(linewidth=4, foreground='lightgray')])
            
        # Label Subplot
        viz.label_sp(ii,alpha=0.75,ax=ax,fontsize=fsz_title,y=1.08,x=-.02)
        ii+=1
            
    if plotver == "sub1":        
        cb      = fig.colorbar(pcm,ax=axs[vv,:].flatten(),pad=0.01,fraction=0.025)
        cb.set_label(dunits,fontsize=fsz_axis)
        cb.ax.tick_params(labelsize=fsz_tick)
        
    elif plotver == "rev1":
        
        cb      = viz.hcbar(pcm,ax=axs[vv,:].flatten(),pad=0.05,fraction=0.055)
        cb.set_label("%s, %s" % (ylabs[vv],dunits),fontsize=fsz_axis)
        cb.ax.tick_params(labelsize=fsz_tick)
        
        
    
savename = "%sInputs_Damping_Feedback.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')            
        
#%% Save parameters above
    

#da_lbdd_out = xr.merge([da_lbdd_sst_conv,da_lbdd_sss_conv,da_lbdd_sst,da_lbdd_sss])
#revpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/revision_data/"
#ncname_lbdd = "%sParameters_plot_subsurface_damping.nc" % revpath

da_tau_a = dampvars[0].rename('tau_a')
da_lbde  = dampvars[1].rename('lbd_e')
da_lbd_a = dampvars[2].rename('lbd_a')

da_lbd_a['lon'] = da_lbde.lon.data
da_lbd_a['lat'] = da_lbde.lat.data

# = da_lbd_a.swap_dims(dict(lat=da_lbde.lat,lon=))

ds_damping_out = xr.merge([da_tau_a,da_lbde,da_lbd_a],join='left')

revpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/revision_data/"
ncname_damping  = "%sParameters_plot_damping_feedbacks.nc" % revpath
edict           = proc.make_encoding_dict(ds_damping_out)
ds_damping_out.to_netcdf(ncname_damping,encoding=edict)


#%%





    
#%%


    
    



#%% Other variables that I need to compute, which are not computed...
# lbd_e
# lbd_d ??



#%% Detrainment Damping (SST vs SSS)

#%% Forcings (Heat Flux Forcing, Evap, Precip, Qek, SST vs SSS)
# [EOF1, EOF2, Correction, Total]
# 4 x 5 plot?

fsz_axis       = 22
selmons        = [1,2] # Indices
monstr         = proc.mon2str(selmons)

Fprime         = convda_byvar[0]['Fprime']                  # [Mode x Mon x Lat x Lon]
Fprime_corr    = convda_byvar[0]['correction_factor']       # [Mon x Lat x Lon]
lhflx          = convda_byvar[1]['LHFLX']                   # [Mode x Mon x Lat x Lon]
lhflx_corr     = convda_byvar[1]['correction_factor_evap']  # [Mon x Lat x Lon]
prec           = convda_byvar[1]['PRECTOT']                 # [Mode x Mon x Lat x Lon]
prec_corr      = convda_byvar[1]['correction_factor_prec']  # [Mon x Lat x Lon]
qek_sst        = convda_byvar[0]['Qek']                     # [Mode x Mon x Lat x Lon]
qek_sss        = convda_byvar[1]['Qek']                     # [Mode x Mon x Lat x Lon]




# Take EOF1, EOF2, Conversion Factor, and Total
Fprime_in = [Fprime.isel(mode=0,mon=selmons).mean('mon'),
             Fprime.isel(mode=1,mon=selmons).mean('mon'),
             Fprime_corr.isel(mon=selmons).mean('mon'),
             stdsqsum_da(Fprime.isel(mon=selmons).mean('mon'),'mode'),
             ]

evap_in = [lhflx.isel(mode=0,mon=selmons).mean('mon'),
           lhflx.isel(mode=1,mon=selmons).mean('mon'),
           lhflx_corr.isel(mon=selmons).mean('mon'),
           stdsqsum_da(lhflx.isel(mon=selmons).mean('mon'),'mode'),
           ]

prec_in = [prec.isel(mode=0,mon=selmons).mean('mon'),
           prec.isel(mode=1,mon=selmons).mean('mon'),
           prec_corr.isel(mon=selmons).mean('mon'),
           stdsqsum_da(prec.isel(mon=selmons).mean('mon'),'mode'),
    ]

qek_sst_in = [qek_sst.isel(mode=0,mon=selmons).mean('mon'),
              qek_sst.isel(mode=1,mon=selmons).mean('mon'),
              None,
              stdsqsum_da(qek_sst.isel(mon=selmons).mean('mon'),'mode'),
              ]

qek_sss_in = [qek_sss.isel(mode=0,mon=selmons).mean('mon'),
              qek_sss.isel(mode=1,mon=selmons).mean('mon'),
              None,
              stdsqsum_da(qek_sss.isel(mon=selmons).mean('mon'),'mode'),
              ]

rownames       = ["EOF 1", "EOF 2", "Correction Factor", "Total"]
vnames_force   = ["Stochastic Heat Flux Forcing\n(SST)","Ekman Forcing\n(SST)","Evaporation\n(SSS)","Precipitation\n(SSS)","Ekman Forcing\n(SSS)"]
plotvars_force = [Fprime_in,qek_sst_in,evap_in,prec_in,qek_sss_in,]

sss_vlim        = [-.01,.01]
sss_vlim_var    = [0,.015]
sst_vlim        = [-.25,.25]
sst_vlim_var    = [0,.5]


#%% Plot the Variables (Old Forcing)

fig,axs,mdict = viz.init_orthomap(4,5,bboxplot=bboxplot,figsize=(30,20))
ii = 0
for rr in range(4):
    
    for vv in range(5):
        
        ax          = axs[rr,vv]
        
        # Get Variable and vlims and Clear Axis where needed
        plotvar     = plotvars_force[vv][rr]
        
        if plotvar is None:
            ax.clear()
            ax.axis('off')
            continue
        else:
            plotvar = plotvar * mask
        
        f_vname = vnames_force[vv]
        if "SST" in f_vname:
            vunit = rparams.vunits[0]
            if rr == 3: # Use variance
                clab = "Total Variance"
                vlim = sst_vlim_var
                cmap = 'cmo.thermal'
            else:
                clab = "SST Forcing"
                vlim = sst_vlim
                cmap = 'cmo.balance'
        
        elif "SSS" in f_vname:
            vunit = rparams.vunits[1]
            if rr == 3: # Use variance
                clab = "Total Variance"
                vlim = sss_vlim_var
                cmap = 'cmo.rain'
            else:
                clab = "SSS Forcing"
                vlim = sss_vlim
                cmap = 'cmo.delta'
        
        # Set Up Axes and Labeling ---
        blb = viz.init_blabels()
        if vv == 0:
            blb['left']  = True
            viz.add_ylabel(rownames[rr],ax=ax,fontsize=fsz_axis)
        if rr == 3:
            blb['lower'] = True
        
        ax          = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,blabels=blb,
                                          fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
        if rr == 0:
            ax.set_title(vnames_force[vv],fontsize=fsz_axis)
        # ------------
        
        pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                            transform=proj,vmin=vlim[0],vmax=vlim[1],cmap=cmap)
        
        
        # # Plot Regions (currently makes colorbars freak out, fix this later)
        # for ir in range(nregs):
        #     rr   = regplot[ir]
        #     rbbx = bboxes[rr]
            
        #     ls_in = rsty[rr]
        #     if ir == 2:
        #         ls_in = 'dashed'
        #     viz.plot_box(rbbx,color=rcols[rr],linestyle=ls_in,leglab=regions_long[rr],linewidth=2.5,return_line=False)

        # Make Colorbars
        makecb = False
        if (rr==2) and (vv==0): # SST Plots (EOF + Correction)
            axcb = axs[:3,:2]
            makecb = True
            frac=0.025
            pad =0.04
        if (rr==3) and (vv==1): # SST Plots (Total Variance)
            axcb = axs[3,:2]
            makecb = True
            frac=0.065
            pad =0.04
        if (rr==2) and (vv==3): # SSS Plots (EOF + Correction)
            axcb = axs[:3,2:]
            makecb = True
            frac=0.025
            pad =0.04
        if (rr==3) and (vv==4): # SST Plots (Total Variance)
            axcb = axs[3,2:]
            makecb = True
            frac=0.065
            pad =0.04
        
        if makecb:
            #cb = fig.colorbar(pcm,ax=axcb.flatten())
            cb  = viz.hcbar(pcm,ax=axcb.flatten(),fraction=frac,pad=pad)
            cb.ax.tick_params(labelsize=fsz_tick)
            cb.set_label(r"%s [$\frac{%s}{mon}$]" % (clab,vunit),fontsize=fsz_axis)
            
        
        # Plot Additional Stuff
        
        
        # Plot Ice Mask
        ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
                   transform=mdict['noProj'],levels=[0,1],zorder=-1)
        
        # Plot Gulf Stream Position
        ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=1.75,c='cornflowerblue',ls='dashdot')
        
        # Label Subplot
        viz.label_sp(ii,alpha=0.75,ax=ax,fontsize=fsz_title,y=1.08,x=-.02)
        ii+=1

    


savename = "%sForcing_SST_SSS_%s.png" % (figpath,monstr)
plt.savefig(savename,dpi=150,bbox_inches='tight')  
    

# ================================================================================
#%% Forcing Plot (Draft 2, Modified from Above)
# ================================================================================

viz_total_include_correction = False # Set to True to include correction in total forcing visualization

selmons        = [1,2] # Indices
monstr         = proc.mon2str(selmons)

Fprime         = convda_byvar[0]['Fprime']                  # [Mode x Mon x Lat x Lon]
Fprime_corr    = convda_byvar[0]['correction_factor']       # [Mon x Lat x Lon]
lhflx          = convda_byvar[1]['LHFLX']                   # [Mode x Mon x Lat x Lon]
lhflx_corr     = convda_byvar[1]['correction_factor_evap']  # [Mon x Lat x Lon]
prec           = convda_byvar[1]['PRECTOT']                 # [Mode x Mon x Lat x Lon]
prec_corr      = convda_byvar[1]['correction_factor_prec']  # [Mon x Lat x Lon]
qek_sst        = convda_byvar[0]['Qek']                     # [Mode x Mon x Lat x Lon]
qek_sst_corr   = convda_byvar[0]['correction_factor_Qek']   # [Mon x Lat x Lon]
qek_sss        = convda_byvar[1]['Qek']                     # [Mode x Mon x Lat x Lon]
qek_sss_corr   = convda_byvar[1]['correction_factor_Qek']   # [Mon x Lat x Lon]

#% Save output for paper figure visualization in another script
da_inputs_all = xr.merge([
    Fprime.rename('Fprime'),
    Fprime_corr.rename('Fprime_corr'),
    lhflx.rename('lhflx'),
    lhflx_corr.rename('lhflx_corr'),
    prec.rename('prec'),
    prec_corr.rename('prec_corr'),
    qek_sst.rename('qek_sst'),
    qek_sst_corr.rename('qek_sst_corr'),
    qek_sss.rename('qek_sss'),
    qek_sss_corr.rename('qek_sss_corr')
    ])

revpath         = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/revision_data/"
ncname_inputs   = "%sRevision01_SM_Input_Parameters_Forcing.nc" % revpath
edict           = proc.make_encoding_dict(da_inputs_all)
da_inputs_all.to_netcdf(ncname_inputs,encoding=edict) 

#%% Compute the Percentage of the correction (Corr% = Corr / (Corr + EOF))
Fprime_std_total = stdsqsum_da(Fprime.isel(mon=selmons).mean('mon'),'mode')
Fprime_corr_perc = (Fprime_corr.isel(mon=selmons).mean('mon')) / (Fprime_corr.isel(mon=selmons).mean('mon') + Fprime_std_total) *100

lhflx_std_total  = stdsqsum_da(lhflx.isel(mon=selmons).mean('mon'),'mode')
lhflx_corr_total = lhflx_corr.isel(mon=selmons).mean('mon')
lhflx_corr_perc  = (lhflx_corr.isel(mon=selmons).mean('mon')) / (lhflx_corr.isel(mon=selmons).mean('mon') + lhflx_std_total) *100

prec_std_total    = stdsqsum_da(prec.isel(mon=selmons).mean('mon'),'mode')
prec_corr_total   = prec_corr.isel(mon=selmons).mean('mon')
prec_corr_perc    = (prec_corr.isel(mon=selmons).mean('mon')) / (prec_corr.isel(mon=selmons).mean('mon') + prec_std_total) *100

qek_sst_std_total = stdsqsum_da(qek_sst.isel(mon=selmons).mean('mon'),'mode')
qek_sst_corr_perc = (qek_sst_corr.isel(mon=selmons).mean('mon')) / (qek_sst_corr.isel(mon=selmons).mean('mon') + qek_sst_std_total ) *100

qek_sss_std_total = stdsqsum_da(qek_sss.isel(mon=selmons).mean('mon'),'mode')
qek_sss_corr_perc = (qek_sss_corr.isel(mon=selmons).mean('mon')) / (qek_sss_corr.isel(mon=selmons).mean('mon') + qek_sss_std_total ) *100


# Try plotting the total forcing (eof + correction) for each case
if viz_total_include_correction:
    Fprime_std_total  = stdsqsum_da(Fprime.isel(mon=selmons).mean('mon'),'mode') + Fprime_corr.isel(mon=selmons).mean('mon')
    qek_sst_std_total = stdsqsum_da(qek_sst.isel(mon=selmons).mean('mon'),'mode') + qek_sst_corr.isel(mon=selmons).mean('mon')
    
    lhflx_std_total  = stdsqsum_da(lhflx.isel(mon=selmons).mean('mon'),'mode') + lhflx_corr.isel(mon=selmons).mean('mon')
    prec_std_total   = stdsqsum_da(prec.isel(mon=selmons).mean('mon'),'mode') + prec_corr.isel(mon=selmons).mean('mon')
    qek_sss_std_total = stdsqsum_da(qek_sss.isel(mon=selmons).mean('mon'),'mode') + qek_sss_corr.isel(mon=selmons).mean('mon')


# Take EOF1, EOF2, Conversion Factor, and Total
Fprime_in = [Fprime.isel(mode=0,mon=selmons).mean('mon'),
             Fprime.isel(mode=1,mon=selmons).mean('mon'),
             Fprime_std_total,
             Fprime_corr_perc,
             ]

evap_in = [lhflx.isel(mode=0,mon=selmons).mean('mon'),
           lhflx.isel(mode=1,mon=selmons).mean('mon'),
           lhflx_std_total,
           np.abs(lhflx_corr_perc), # Need to check why this is negative?
           ]

prec_in = [prec.isel(mode=0,mon=selmons).mean('mon'),
           prec.isel(mode=1,mon=selmons).mean('mon'),
           prec_std_total,
           prec_corr_perc,
           ]

qek_sst_in = [qek_sst.isel(mode=0,mon=selmons).mean('mon'),
              qek_sst.isel(mode=1,mon=selmons).mean('mon'),
              qek_sst_std_total,
              qek_sst_corr_perc,
              ]

qek_sss_in = [qek_sss.isel(mode=0,mon=selmons).mean('mon'),
              qek_sss.isel(mode=1,mon=selmons).mean('mon'),
              qek_sss_std_total,
              qek_sss_corr_perc,
              ]

rownames       = ["EOF 1", "EOF 2", "EOF Total", r"$\frac{Correction \,\, Factor}{EOF \,\, Total \,\, + Correction \,\, Factor }$"]
if plotver == "rev1":
    
    vnames_force   = ["Stochastic Heat Flux Forcing\n"+r"($\frac{1}{\rho C_p h} F_N'$, SST)",
                      "Ekman Forcing\n($Q_{ek,T},SST)$",
                      "Evaporation\n"+r"($\frac{\overline{S}}{\rho h L} F_L'$,SSS)",
                      "Precipitation\n"+r"($\frac{\overline{S}}{\rho h} P'$,SSS)",
                      "Ekman Forcing\n($Q_{ek,S},SSS)$"]
    
else:
    vnames_force   = ["Stochastic Heat Flux Forcing\n"+r"($\frac{F'}{\rho C_p h}$, SST)",
                      "Ekman Forcing\n($Q_{ek}'$, SST)",
                      "Evaporation\n"+r"($\frac{\overline{S} q_L'}{\rho h L}$,SSS)",
                      "Precipitation\n"+r"($\frac{\overline{S} P'}{\rho h}$,SSS)",
                      "Ekman Forcing\n($Q_{ek}'$, SSS)"]
plotvars_force = [Fprime_in,qek_sst_in,evap_in,prec_in,qek_sss_in,]

#%% Plot the Forcings (Draft 2 Onwards)

pubready = True

mult_SSS_factor = 1e3 #Default is 1

fsz_tick  = 26
fsz_title = 32
fsz_axis  = 28

sss_vlim        = np.array([-.01,.01]) * mult_SSS_factor
sss_vlim_var    = np.array([0,.015])   * mult_SSS_factor
sst_vlim        = [-.20,.20]
sst_vlim_var    = [0,.5]

plotover        = False
if plotover:
    cints_sst_lim = np.arange(0.5,1.6,0.25)
    cints_sss_lim = np.arange(0.015,1.5,0.015)  * mult_SSS_factor
else:
    cints_sst_lim = np.arange(0,0.55,0.05)
    cints_sss_lim = np.arange(0,0.028,0.004)  * mult_SSS_factor
    

fig,axs,mdict = viz.init_orthomap(4,5,bboxplot=bboxplot,figsize=(30,22))
ii = 0
for rr in range(4):
    
    for vv in range(5):
        
        ax          = axs[rr,vv]
        
        # Get Variable and vlims and Clear Axis where needed
        plotvar     = plotvars_force[vv][rr]
        
        if plotvar is None:
            ax.clear()
            ax.axis('off')
            continue
        else:
            plotvar = plotvar * mask
        
        f_vname = vnames_force[vv]
        if "SST" in f_vname:
            vunit = rparams.vunits[0]
            if rr == 2: # Use variance
                clab = "Total Standard Deviation"
                vlim = sst_vlim_var
                cmap = cm.lajolla_r
                cints_lim = cints_sst_lim
                if plotover:
                    ccol      = "k"
                else:
                    ccol      = 'dimgray'
                
            elif rr == 3: # % Correction
                clab = "% Correction"
                vlim = [0,100]
                cmap = 'cmo.amp'
                
            else:
                clab = "SST Forcing"
                vlim = sst_vlim
                cmap = 'cmo.balance'
                
        
        elif "SSS" in f_vname:
            if rr < 3:
                plotvar =  plotvar * mult_SSS_factor
            vunit = rparams.vunits[1]
            if rr == 2: # Use variance
                clab = "Total Standard Deviation"
                vlim = sss_vlim_var
                cmap = cm.acton_r#'cmo.rain'
                if plotover:
                    ccol = "lightgray"
                else:
                    ccol = 'lightgray'
                cints_lim = cints_sss_lim
            elif rr == 3: # % Correction
                clab = "% Correction"
                vlim = [0,100]
            else:
                clab = "SSS Forcing"
                vlim = sss_vlim
                cmap = 'cmo.delta'
        
        # Set Up Axes and Labeling ---
        blb = viz.init_blabels()
        if vv == 0:
            blb['left']  = True
            viz.add_ylabel(rownames[rr],ax=ax,fontsize=fsz_axis)
        if rr == 3:
            blb['lower'] = True
        
        ax          = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,blabels=blb,
                                          fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
        if rr == 0:
            ax.set_title(vnames_force[vv],fontsize=fsz_axis)
        # ------------
        
        pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                            transform=proj,vmin=vlim[0],vmax=vlim[1],cmap=cmap)
        pcm.set_rasterized(True) 
        
        # Plot additional contours
        if rr == 2:
            
            #ccol = "lightgray"
            cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                                transform=proj,levels=cints_lim,
                                colors=ccol,linewidths=0.75,)
            
            
            
            cl_lab = ax.clabel(cl,levels=cints_lim[::2],fontsize=fsz_tick)
            if ccol == "lightgray":
                fbcol = "k"
            else:
                fbcol = "w"
                
            viz.add_fontborder(cl_lab,w=4,c=fbcol)
        
        # Make Colorbars (and Adjust)
        makecb = False
        if (rr==1) and (vv==0): # SST Plots (EOF)
            axcb = axs[:2,:2]
            makecb = True
            frac=0.03
            pad =0.04
        if (rr==2) and (vv==1): # SST Plots (total Variance)
            axcb = axs[2,:2]
            makecb = True
            frac=0.06
            pad =0.04
        # if (rr==3) and (vv==1): # SST Plots (Correction)
        #     axcb = axs[3,:2]
        #     makecb = True
        #     frac=0.06
        #     pad =0.04
        if (rr==1) and (vv==3): # SSS Plots (EOF)
            axcb = axs[:2,2:]
            makecb = True
            frac=0.03
            pad =0.04
        if (rr==2) and (vv==3): # SSS Plots (total Variance)
            axcb = axs[2,2:]
            makecb = True
            frac=0.06
            pad =0.04
        if (rr==3) and (vv==4): # SST Plots (Total Variance)
            axcb = axs[3,:]
            makecb = True
            frac   = 0.065
            pad     =0.04
        
        if makecb:
            #cb = fig.colorbar(pcm,ax=axcb.flatten())
            cb  = viz.hcbar(pcm,ax=axcb.flatten(),fraction=frac,pad=pad)
            cb.ax.tick_params(labelsize=fsz_tick)
            if rr == 3: # Correction Factor
                
                cb.set_label(r"%s" % (clab),fontsize=fsz_axis)
            else:
                if vunit == "psu" and mult_SSS_factor > 1: # Add Multiplcation Factor
                    mult_factor = (np.log10(1/mult_SSS_factor))
                    
                    cb.set_label(r"%s [$\frac{%s}{mon}$ $\times$10$^{%i}$]" % (clab,vunit,mult_factor),fontsize=fsz_axis)
                else:
                    cb.set_label(r"%s [$\frac{%s}{mon}$]" % (clab,vunit),fontsize=fsz_axis)
            
        
        # Plot Additional Stuff
        
        
        # Plot Ice Mask
        ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
                   transform=mdict['noProj'],levels=[0,1],zorder=-1)
        
        # Plot Gulf Stream Position
        gss = ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=1.75,c='k',ls='dashdot')
        gss[0].set_path_effects([PathEffects.withStroke(linewidth=6, foreground='lightgray')])
        
        
        
        # Label Subplot
        viz.label_sp(ii,alpha=0.75,ax=ax,fontsize=fsz_title,y=1.08,x=-.02)
        ii+=1
        
savename = "%sForcing_SST_SSS_Draft02_%s.png" % (figpath,monstr)
if viz_total_include_correction:
    savename = proc.addstrtoext(savename,"_addCorrToTotal")
    
if pubready:
    savename = "%sFig03_Forcing.png" % (figpath)
    plt.savefig(savename,dpi=900,bbox_inches='tight')  
    
    savename = "%sFig03_Forcing.pdf" % (figpath)
    plt.savefig(savename,format="pdf",bbox_inches='tight')
    
else:
    plt.savefig(savename,dpi=150,bbox_inches='tight')  

# ====================================================================================
#%% Scrap Section Below
# =====================================================================================
# #%% Troubleshooting Precip


# cint = np.arange(0,0.15,0.002)
# fig,ax,_ = viz.init_orthomap(1,1,bboxplot,figsize=(22,4.5))

# ax          = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,blabels=blb,
#                                   fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")

# #plotvar = prec_corr_total + prec_std_total
# plotvar = lhflx_corr_total + lhflx_std_total
# cf = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
#                  cmap =cm.acton_r,vmin=0,vmax=0.015)

# cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
#                  colors="k",levels=cint)

# viz.hcbar(cf,ax=ax)
# ax.clabel(cl,fontsize=18)
# #prec_corr_total
# #prec_std_total.plot(vmin=0,vmax=0.015,cmap=cm.acton_r)


#%% What To Visualize First... Lets try the dampings














#%%
# get lat.lon


# # First, Check if there is EOF-based forcing (remove this if I eventually redo it)
# if expparams['eof_forcing']:
#     print("EOF Forcing Detected.")
#     eof_flag = True
# else:
#     eof_flag = False

# # Indicate the Parameter Names (sorry, it's all hard coded...)
# if expparams['varname']== "SSS": # Check for LHFLX, PRECTOT, Sbar
#     chk_params = ["h","LHFLX","PRECTOT","Sbar","lbd_d","beta","kprev","lbd_a","Qek"]
#     param_type = ["mld","forcing","forcing","forcing","damping","mld","mld","damping",'forcing']
# elif expparams['varname'] == "SST": # Check for Fprime
#     chk_params = ["h","Fprime","lbd_d","beta","kprev","lbd_a","Qek"]
#     param_type = ["mld","forcing","damping","mld","mld","damping",'forcing']

# # Check the params
# ninputs       = len(chk_params)
# inputs_ds     = {}
# inputs        = {}
# inputs_type   = {}
# missing_input = []
# for nn in range(ninputs):
#     # Get Parameter Name and Type
#     pname = chk_params[nn]
#     ptype = param_type[nn]
    
#     # Check for Exceptions (Can Fix this in the preprocessing stage)
#     if pname == 'lbd_a':
#         da_varname = 'damping'
#     else:
#         da_varname = pname
    
#     #print(pname)
#     if type(expparams[pname])==str: # If String, Load from input folder
        
#         # Load ds
#         ds = xr.open_dataset(input_path + ptype + "/" + expparams[pname])[da_varname]
        

#         # Crop to region
        
#         # Load dataarrays for debugging
#         dsreg            = proc.sel_region_xr(ds,expparams['bbox_sim']).load()
#         inputs_ds[pname] = dsreg.copy() 
        
#         # Load to numpy arrays 
#         varout           = dsreg.values
#         inputs[pname]    = dsreg.values.copy()
        
#         if ((da_varname == "Fprime") and (eof_flag)) or ("corrected" in expparams[pname]):
#             print("Loading %s correction factor for EOF forcing..." % pname)
#             ds_corr                          = xr.open_dataset(input_path + ptype + "/" + expparams[pname])['correction_factor']
#             ds_corr_reg                      = proc.sel_region_xr(ds_corr,expparams['bbox_sim']).load()
            
#             # set key based on variable type
#             if da_varname == "Fprime":
#                 keyname = "correction_factor"
#             elif da_varname == "LHFLX":
#                 keyname = "correction_factor_evap"
#             elif da_varname == "PRECTOT":
#                 keyname = "correction_factor_prec"
                
#             inputs_ds[keyname]   = ds_corr_reg.copy()
#             inputs[keyname]      = ds_corr_reg.values.copy()
#             inputs_type[keyname] = "forcing"
        
#     else:
#         print("Did not find %s" % pname)
#         missing_input.append(pname)
#     inputs_type[pname] = ptype



#%% Visualize some of the inputs

# Set up mapping template
# Plotting Params
mpl.rcParams['font.family'] = 'Avenir'
bboxplot                    = [-80,0,20,65]
proj                        = ccrs.PlateCarree()
lon                         = ds.lon.values
lat                         = ds.lat.values
mons3                       = proc.get_monstr()

plotmon                     = np.roll(np.arange(12),1)

fsz_title   = 26
fsz_axis    = 22
fsz_lbl     = 10

#%%

def init_monplot():
    plotmon       = np.roll(np.arange(12),1)
    fig,axs,mdict = viz.init_orthomap(4,3,bboxplot=bboxplot,figsize=(18,18))
    for a,ax in enumerate(axs.flatten()):
        im = plotmon[a]
        ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
        ax.set_title(mons3[im],fontsize=fsz_axis)
    return fig,axs

# def make_mask_xr(ds,var):
#     ds_masked = ds.where(ds != val)
#     return ds_masked
# # ds_mask = make_mask_xr(selvar,0.)
# # ds_mask = ds_mask.sum('mon')

# Make A Mask
ds_mask = ds.sum('mon')
ds_mask = xr.where( ds_mask!=0. , 1 , np.nan)
ds_mask.plot()

#%% Load additional variables for gulf stream, land ice mask

# Load Land Ice Mask
icemask         = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")

mask            = icemask.MASK.squeeze()
mask_plot       = xr.where(np.isnan(mask),0,mask)#mask.copy()


mask_reg_sub    = proc.sel_region_xr(mask,bboxplot)
mask_reg_ori    = xr.ones_like(mask) * 0
mask_reg        = mask_reg_ori + mask_reg_sub

mask_apply  = icemask.MASK.squeeze().values
#mask_plot[np.isnan(mask)] = 0

# Load Gulf Stream
ds_gs           = dl.load_gs()
ds_gs           = ds_gs.sel(lon=slice(-90,-50))
ds_gs2          = dl.load_gs(load_u2=True)


#%% Lets make the plot
print(inputs.keys())

# ---------------------------------
#%% Plot Mixed Layer Depth by Month
# ---------------------------------

# Set some parameters
vname       = 'h'
vname_long  = "Mixed-Layer Depth"
vlabel      = "HMXL (meters)"
plotcontour = False
vlms        = [0,200]# None
cints_sp    = np.arange(200,1500,100)# None
cmap        = 'cmo.dense'

# Get variable, lat, lon
selvar      = inputs_ds[vname]
lon         = selvar.lon
lat         = selvar.lat

# Special plot for HMXL, month of maximum
hmax        = selvar.argmax('mon').values
hmin        = selvar.argmin('mon').values


fig,axs     = init_monplot()

for aa in range(12):
    ax      = axs.flatten()[aa]
    im      = plotmon[aa]
    plotvar = selvar.isel(mon=im) * ds_mask

    
    # Just Plot the contour with a colorbar for each one
    if vlms is None:
        pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,cmap=cmap)
        fig.colorbar(pcm,ax=ax)
    else:
        pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                            cmap=cmap,vmin=vlms[0],vmax=vlms[1])
    
    # Do special contours
    if cints_sp is not None:
        cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                        levels=cints_sp,colors="w",linewidths=0.75)
        ax.clabel(cl,fontsize=fsz_lbl)
        
        
    # Special plot for MLD (mark month of maximum)
    hmask_max  = (hmax == im) * ds_mask # Note quite a fix, as 0. points will be rerouted to april
    hmask_min  = (hmin == im) * ds_mask
    
    smap = viz.plot_mask(hmask_max.lon,hmask_max.lat,hmask_max.T,reverse=True,
                         color="yellow",markersize=0.75,marker="x",
                         ax=ax,proj=proj,geoaxes=True)
    
    smap = viz.plot_mask(hmask_min.lon,hmask_min.lat,hmask_min.T,reverse=True,
                         color="red",markersize=0.75,marker="o",
                         ax=ax,proj=proj,geoaxes=True)
    
    
    
if vlms is not None:
    cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.025)
    cb.set_label(vlabel,fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)
    # cb = fig.colorbar(pcm,ax=axs.flatten(),
    #                   orientation='horizontal',pad=0.02,fraction=0.025)
    # cb.set_label(vlabel)

plt.suptitle("%s (CESM1 Ensemble Average)" % (vname_long),fontsize=fsz_title)
savename = "%s%s_Model_Inputs_%s_seasonrow.png" % (figpath,expname,vname)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# ------------------------------- -------------------
#%% Make plot of maximum wintertime mixed layer depth
# ------------------------------- -------------------


plot_dots = False
fsz_tick = 26
fsz_axis = 32


vlabel      = "Max Seasonal Mixed Layer Depth (meters)"

fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(24,14.5),)
ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)

# Plot maximum MLD
plotvar     = selvar.max('mon') * ds_mask

# Just Plot the contour with a colorbar for each one
if vlms is None:
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,cmap=cmap,zorder=-1)
    cb = fig.colorbar(pcm,ax=ax)
else:
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                        cmap=cmap,vmin=vlms[0],vmax=vlms[1],zorder=-4)
    
# Do special contours
if cints_sp is not None:
    cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                    levels=cints_sp,colors="w",linewidths=1.1,zorder=6)
    ax.clabel(cl,fontsize=fsz_tick,zorder=6)

if plot_dots:
    # Special plot for MLD (mark month of maximum)
    hmask_feb  = (hmax == 1) * ds_mask # Note quite a fix, as 0. points will be rerouted to april
    hmask_mar  = (hmax == 2) * ds_mask
    
    smap = viz.plot_mask(hmask_feb.lon,hmask_feb.lat,hmask_feb.T,reverse=True,
                         color="violet",markersize=8,marker="x",
                         ax=ax,proj=proj,geoaxes=True)
    
    smap = viz.plot_mask(hmask_mar.lon,hmask_mar.lat,hmask_mar.T,reverse=True,
                         color="palegoldenrod",markersize=4,marker="o",
                         ax=ax,proj=proj,geoaxes=True)

        
if vlms is not None:
    cb = viz.hcbar(pcm,ax=ax,fraction=0.025)
    cb.set_label(vlabel,fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)
    
# Add additional features
ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=4,c='firebrick',ls='dashdot')

# Plot Ice Edge
ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=4,
           transform=proj,levels=[0,1],zorder=-1)

savename = "%sWintertime_MLD_CESM1_%s.png" % (figpath,expname)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)

# ---------------------------------
#%% Plot Detrainment Damping
# ---------------------------------

# Set some parameters
vname          = 'lbd_d'
vname_long     = "Deep Damping"
plotcontour    = False
#corrmode      = True # Using correlations rather than detrainment damping values
#vlms          = [0,48]#[0,0.2]# None
plot_timescale = False

if "corr" in expparams['lbd_d']:
    corrmode = True
else:
    corrmode = False

if corrmode:
    vlabel         = "Corr(Detrain,Entrain)"
else:
    vlabel         = "$\lambda_d$ ($months^{-1}$)"
    
    if "SSS" in expname:
        cints_sp       = np.arange(0,66,12)#None#np.arange(200,1500,100)# None
    elif "SST" in expname:
        cints_sp       = np.arange(0,66,12)

# Get variable, lat, lon
selvar      = inputs_ds[vname]
lon         = selvar.lon
lat         = selvar.lat

# Preprocessing
ds_mask = xr.where( selvar != 0. , 1 , np.nan)
if corrmode is False:
    
    if plot_timescale:
        
        plotvar = 1/selvar
        vlabel = "$\lambda^d^{-1}$ ($months$)"
        if "SSS" in expname:
            vlms           = [0,48]
        elif "SST" in expname:
            vlms           = [0,24]
        cmap           = 'inferno'
    else:
        
        plotvar = selvar
        vlabel ="$\lambda^d$ ($months^{-1}$)"
        if "SSS" in expname:
            vlms           = [0,0.2]
        elif "SST" in expname:
            vlms           = [0,0.5]
        cmap           = 'inferno'
        
else:
    
    if plot_timescale:
        plotvar = -1/np.log(selvar)
        vlms    = [0,12]
        vlabel = "Deep Damping Timescale ($months$)"
    else:
        plotvar = selvar
        vlms    = [0,1]
        vlabel = "Corr(Detrain,Entrain)"
    
plotvar = plotvar * ds_mask

fig,axs = init_monplot()
for aa in range(12):
    ax      = axs.flatten()[aa]
    im      = plotmon[aa]
    pv = plotvar.isel(mon=im) 
    
    # Just Plot the contour with a colorbar for each one
    if vlms is None:
        pcm = ax.pcolormesh(lon,lat,pv,transform=proj,cmap=cmap)
        fig.colorbar(pcm,ax=ax)
    else:
        pcm = ax.pcolormesh(lon,lat,pv,transform=proj,
                            cmap=cmap,vmin=vlms[0],vmax=vlms[1])
    
    # Do special contours
    if cints_sp is not None:
        plotvar = 1/plotvar
        cl = ax.contour(lon,lat,pv,transform=proj,
                        levels=cints_sp,colors="lightgray",linewidths=0.75)
        ax.clabel(cl,fontsize=fsz_lbl)

if vlms is not None:
    
    cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.025)
    cb.set_label(vlabel,fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)
    # cb = fig.colorbar(pcm,ax=axs.flatten(),
    #                   orientation='horizontal',pad=0.02,fraction=0.025)
    # cb.set_label(vlabel)

plt.suptitle("%s (Colors) and Timescale (Contours), CESM1 Ensemble Average" % (vname_long),fontsize=fsz_title)
savename = "%s%s_Model_Inputs_%s_seasonrow_timescale%i.png" % (figpath,expname,vname,plot_timescale)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# ----------------------------------------------------------
#%% Plot Wintertime and Summertime Mean Detrainment Damping
# ----------------------------------------------------------

# Set some parameters
vname          = 'lbd_d'
vname_long     = "Detrainment Damping"
vlabel         = "$\lambda_d$ ($months^{-1}$)"
plotcontour    = False
#vlms          = [0,48]#[0,0.2]# None

if "SSS" in expname:
    cints_sp       = np.arange(0,66,12)#None#np.arange(200,1500,100)# None
elif "SST" in expname:
    cints_sp       = np.arange(0,66,12)

plot_timescale = False

# Get variable, lat, lon
selvar      = inputs_ds[vname]
lon         = selvar.lon
lat         = selvar.lat

# Preprocessing
ds_mask = xr.where( selvar != 0. , 1 , np.nan)
if plot_timescale:
    selvar = 1/selvar
    vlabel = "$\lambda^d^{-1}$ ($months$)"
    if "SSS" in expname:
        vlms           = [0,48]
    elif "SST" in expname:
        vlms           = [0,24]
    cmap           = 'inferno'
else:
    vlabel ="$\lambda^d$ ($months^{-1}$)"
    if "SSS" in expname:
        vlms           = [0,0.2]
    elif "SST" in expname:
        vlms           = [0,0.5]
    cmap           = 'inferno'

selvar = selvar * ds_mask

fig,axs,mdict = viz.init_orthomap(1,2,bboxplot=bboxplot,figsize=(12,6))
for a,ax in enumerate(axs.flatten()):
    im = plotmon[a]
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
    ax.set_title(mons3[im],fontsize=fsz_axis)
    
    if a == 0:
        selmon = [11,0,1,2]
        title  = "Winter (DJFM)"
        
        cints_sp       = np.arange(0,66,12)
    elif a == 1:
        selmon = [6,7,8,9]
        title  = "Summer (JJAS)"
        
        if "SST" in expname:
            cints_sp       = np.arange(0,25,1)
        elif "SSS" in expname:
            cints_sp       = np.arange(0,36,3)
    plotvar = selvar.isel(mon=selmon).mean('mon')
    
    if vlms is None:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap)
        fig.colorbar(pcm,ax=ax)
    else:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                            cmap=cmap,vmin=vlms[0],vmax=vlms[1])
    
    # Do special contours
    if cints_sp is not None:
        plotvar = 1/plotvar
        cl = ax.contour(lon,lat,plotvar,transform=proj,
                        levels=cints_sp,colors="lightgray",linewidths=0.75)
        ax.clabel(cl,fontsize=fsz_lbl)
    ax.set_title(title,fontsize=fsz_title-2)

# for aa in range(12):
#     ax      = axs.flatten()[aa]
#     im      = plotmon[aa]
#     plotvar = selvar.isel(mon=im) 
    
#     # Just Plot the contour with a colorbar for each one
#     if vlms is None:
#         pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap)
#         fig.colorbar(pcm,ax=ax)
#     else:
#         pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,
#                             cmap=cmap,vmin=vlms[0],vmax=vlms[1])
#     # Do special contours
#     if cints_sp is not None:
#         plotvar = 1/plotvar
#         cl = ax.contour(lon,lat,plotvar,transform=proj,
#                         levels=cints_sp,colors="lightgray",linewidths=0.75)
#         ax.clabel(cl,fontsize=fsz_lbl)

if vlms is not None:
    cb = fig.colorbar(pcm,ax=axs.flatten(),
                      orientation='horizontal',pad=0.02,fraction=0.05)
    cb.set_label(vlabel)

plt.suptitle("%s (Colors) and Timescale (Contours),\nCESM1 Ensemble Average" % (vname_long),fontsize=fsz_title)
savename = "%s%s_Model_Inputs_%s_wintersummermean_timescale%i.png" % (figpath,expname,vname,plot_timescale)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)

# -------------------
#%% Visualize Precip
# -------------------

# Set some parameters
vname          = 'PRECTOT'
vname_long     = "Total Precipitation"
cints_prec     = np.arange(0,0.022,0.002)
plotcontour    = True

cints_sp       = None# np.arange(0,66,12)#None#np.arange(200,1500,100)# None
cmap           = 'cmo.rain'
convert_precip = True

# For Precip, also get the correct factor

# Get variable, lat, lon
selvar      = inputs_ds[vname]
selvar      = np.sqrt((selvar**2).sum('mode'))
lon         = selvar.lon
lat         = selvar.lat

# Convert Precipitation, if option is set
if convert_precip: # Adapted from ~line 559 of run_SSS_basinwide

    print("Converting precip to psu/mon")
    conversion_factor   = ( dt*inputs['Sbar'] / inputs['h'] )
    selvar              =  selvar * conversion_factor
    
    vlabel              = "$P$ ($psu/mon$)"
    vlms                = np.array([0,0.02])#None#[0,0.05]#[0,0.2]# None
    
else:
    
    vlabel         = "$P$ ($m/s$)"
    vlms           = np.array([0,1.5])*1e-8#None#[0,0.05]#[0,0.2]# None
    


# Preprocessing
ds_mask     = xr.where( selvar != 0. , 1 , np.nan)
selvar      = selvar * ds_mask 

fig,axs     = init_monplot()
for aa in range(12):
    ax      = axs.flatten()[aa]
    im      = plotmon[aa]
    plotvar = selvar.isel(mon=im) 
    
    # Just Plot the contour with a colorbar for each one
    if vlms is None:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap,zorder=-3)
        fig.colorbar(pcm,ax=ax)
    else:
        if plotcontour:
            pcm = ax.contourf(lon,lat,plotvar,transform=proj,
                              cmap=cmap,levels=cints_prec,zorder=-3)
            
            cl = ax.contour(lon,lat,plotvar,transform=proj,colors="k",linewidths=0.75,
                              levels=cints_prec,zorder=-3)
            ax.clabel(cl,fontsize=fsz_tick-4)
        else:
            pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                                cmap=cmap,vmin=vlms[0],vmax=vlms[1],zorder=-3)
    
    # plotvar2 = selvar2.isel(mon=im)
    # cl = ax.contour(lon,lat,plotvar2,transform=proj,
    #                 colors="k",linewidths=0.75)
    # ax.clabel(cl,fontsize=fsz_lbl)
    
    # Add additional features
    ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=2.5,c='cornflowerblue',ls='dashdot')

    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2,
               transform=proj,levels=[0,1],zorder=-1)
    
    
if vlms is not None:
    cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.025)
    cb.set_label(vlabel,fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)
    
plt.suptitle("%s (CESM1 Ensemble Average)" % (vname_long),fontsize=fsz_title)
savename = "%s%s_Model_Inputs_%s_seasonrow_convert%i.png" % (figpath,expname,vname,convert_precip)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Plot the mean Precip
fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(24,14.5),)
ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)

plotvar = selvar.mean('mon')

# Just Plot the contour with a colorbar for each one
if vlms is None:
    pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap,zorder=-3)
    fig.colorbar(pcm,ax=ax)
else:
    if plotcontour:
        pcm = ax.contourf(lon,lat,plotvar,transform=proj,
                          cmap=cmap,levels=cints_prec,zorder=-3)
        
        cl = ax.contour(lon,lat,plotvar,transform=proj,colors="k",linewidths=0.75,
                          levels=cints_prec,zorder=-3)
        ax.clabel(cl,fontsize=fsz_tick-4)
    else:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                            cmap=cmap,vmin=vlms[0],vmax=vlms[1],zorder=-3)

# plotvar2 = selvar2.isel(mon=im)
# cl = ax.contour(lon,lat,plotvar2,transform=proj,
#                 colors="k",linewidths=0.75)
# ax.clabel(cl,fontsize=fsz_lbl)

# Add additional features
ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=2.5,c='cornflowerblue',ls='dashdot')

# Plot Ice Edge
ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2,
           transform=proj,levels=[0,1],zorder=-1)

    
if vlms is not None:
    cb = viz.hcbar(pcm,ax=ax)
    cb.set_label(vlabel,fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)
    
plt.suptitle("%s (CESM1 Ensemble Average)" % (vname_long),fontsize=fsz_title)
savename = "%s%s_Model_Inputs_%s_Mean_convert%i.png" % (figpath,expname,vname,convert_precip)
plt.savefig(savename,dpi=150,bbox_inches='tight')


# -------------------
#%% Visualize Evap
# -------------------

# Set some parameters
vname          = 'LHFLX'
vname_long     = "Latent Heat FLux"
vlabel         = "$LHFLX$ ($W/m^2$)"
plotcontour    = True

cints_sp       = None# np.arange(0,66,12)#None#np.arange(200,1500,100)# None
cmap           = 'cmo.amp'
convert_lhflx  = True


# Get variable, lat, lon
selvar      = inputs_ds[vname]
selvar      = np.sqrt((selvar**2).sum('mode'))
lon         = selvar.lon
lat         = selvar.lat

# Do Conversion of Evaporation if option is set
if convert_lhflx:
    conversion_factor = ( dt*inputs['Sbar'] / (rho*L*inputs['h']))
    selvar_in         = selvar * conversion_factor # [Mon x Lat x Lon] * -1
    
    vlms           = [0,0.02]#[0,0.2]# None
    cints_evap     = np.arange(0,0.022,0.002)
    
else:
    selvar_in = selvar.copy()
    vlms           = [0,35]#[0,0.2]# None
    cints_evap     = np.arange(0,39,3)
    

# Preprocessing
ds_mask        = xr.where( selvar != 0. , 1 , np.nan)
selvar_in      = selvar_in * ds_mask 

fig,axs = init_monplot()
for aa in range(12):
    ax      = axs.flatten()[aa]
    im      = plotmon[aa]
    plotvar = selvar_in.isel(mon=im) 
    
    # Just Plot the contour with a colorbar for each one
    if vlms is None:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap,zorder=-3)
        fig.colorbar(pcm,ax=ax)
    else:
        
        if plotcontour:
            
            pcm = ax.contourf(lon,lat,plotvar,transform=proj,
                              cmap=cmap,levels=cints_evap,zorder=-3)
            
            cl = ax.contour(lon,lat,plotvar,transform=proj,colors="k",linewidths=0.75,
                              levels=cints_prec,zorder=-3)
            ax.clabel(cl,fontsize=fsz_tick-4)
            
        else:
            
            pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                                cmap=cmap,vmin=vlms[0],vmax=vlms[1],zorder=-3)
        
    # # Do special contours
    # if cints_sp is not None:
    #     plotvar = 1/plotvar
    #     cl = ax.contour(lon,lat,plotvar,transform=proj,
    #                     levels=cints_sp,colors="lightgray",linewidths=0.75)
    #     ax.clabel(cl,fontsize=fsz_lbl)
        
    
    
if vlms is not None:
    cb = viz.hcbar(pcm,ax=axs.flatten())
    cb.set_label(vlabel,fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)

plt.suptitle("%s (CESM1 Ensemble Average)" % (vname_long),fontsize=fsz_title)
savename = "%s%s_Model_Inputs_%s_seasonrow_convert%i.png" % (figpath,expname,vname,convert_lhflx)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Plot the mean evaporation

fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(24,14.5),)
ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)

plotvar = selvar_in.mean('mon')


im      = plotmon[aa]
plotvar = selvar_in.isel(mon=im) 

# Just Plot the contour with a colorbar for each one
if vlms is None:
    pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap,zorder=-3)
    fig.colorbar(pcm,ax=ax)
else:
    
    if plotcontour:
        
        pcm = ax.contourf(lon,lat,plotvar,transform=proj,
                          cmap=cmap,levels=cints_evap,zorder=-3)
        
        cl = ax.contour(lon,lat,plotvar,transform=proj,colors="k",linewidths=0.75,
                          levels=cints_prec,zorder=-3)
        ax.clabel(cl,fontsize=fsz_tick-4)
        
    else:
        
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                            cmap=cmap,vmin=vlms[0],vmax=vlms[1],zorder=-3)
    

    
    
if vlms is not None:
    cb = viz.hcbar(pcm,ax=ax)
    cb.set_label(vlabel,fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)

plt.suptitle("%s (CESM1 Ensemble Average)" % (vname_long),fontsize=fsz_title)
savename = "%s%s_Model_Inputs_%s_Mean_convert%i.png" % (figpath,expname,vname,convert_lhflx)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# ----------------------------------------
#%% Visualize the correction factor (evap)
# ----------------------------------------

# Set some parameters
vname          = 'correction_factor_evap'
vname_long     = "Latent Heat FLux (Correction Factor)"
vlabel         = "$LHFLX$ ($W/m^2$)"
plotcontour    = True
vlms           = [0,15]#[0,0.2]# None
cints_sp       = None# np.arange(0,66,12)#None#np.arange(200,1500,100)# None
cmap           = 'cmo.amp'

convert_lhflx  = True


# Get variable, lat, lon
selvar      = inputs_ds[vname]
lon         = selvar.lon
lat         = selvar.lat

# Do Conversion of Evaporation if option is set
if convert_lhflx:
    conversion_factor = ( dt*inputs['Sbar'] / (rho*L*inputs['h']))
    selvar_in         = selvar * conversion_factor # [Mon x Lat x Lon] * -1
    
    vlms           = [0,0.02]#[0,0.2]# None
    cints_evap     = np.arange(0,0.022,0.002)
    
else:
    selvar_in = selvar.copy()
    vlms           = [0,35]#[0,0.2]# None
    cints_evap     = np.arange(0,39,3)

# Preprocessing
ds_mask     = xr.where( selvar != 0. , 1 , np.nan)
selvar_in      = selvar_in * ds_mask 

fig,axs = init_monplot()
for aa in range(12):
    ax      = axs.flatten()[aa]
    im      = plotmon[aa]
    plotvar = selvar_in.isel(mon=im) 
    
    # Just Plot the contour with a colorbar for each one
    if vlms is None:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap,zorder=-3)
        fig.colorbar(pcm,ax=ax)
    else:
        
        if plotcontour:
            
            pcm = ax.contourf(lon,lat,plotvar,transform=proj,
                              cmap=cmap,levels=cints_evap,zorder=-3)
            
            cl = ax.contour(lon,lat,plotvar,transform=proj,colors="k",linewidths=0.75,
                              levels=cints_prec,zorder=-3)
            ax.clabel(cl,fontsize=fsz_tick-4)
            
        else:
            
            pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                                cmap=cmap,vmin=vlms[0],vmax=vlms[1],zorder=-3)
        
        


if vlms is not None:
    cb = viz.hcbar(pcm,ax=axs.flatten())
    cb.set_label(vlabel,fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)

plt.suptitle("%s (CESM1 Ensemble Average)" % (vname_long),fontsize=fsz_title)
savename = "%s%s_Model_Inputs_%s_seasonrow_convert%i.png" % (figpath,expname,vname,convert_lhflx)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# ------------------------------------------
#%% Visualize the correction factor (precip)
# ------------------------------------------

# Set some parameters
vname          = 'correction_factor_prec'
vname_long     = "Precipitation (Correction Factor)"

plotcontour    = True
vlms           = np.array([0,3])*1e-9
cints_sp       = None# np.arange(0,66,12)#None#np.arange(200,1500,100)# None
cmap           = 'cmo.rain'
convert_precip = True

# Get variable, lat, lon
selvar      = inputs_ds[vname]
lon         = selvar.lon
lat         = selvar.lat

# Convert Precipitation, if option is set
if convert_precip: # Adapted from ~line 559 of run_SSS_basinwide
    print("Converting precip to psu/mon")
    conversion_factor   = ( dt*inputs['Sbar'] / inputs['h'] )
    selvar              =  selvar * conversion_factor
    
    vlms                = np.array([0,0.02])#None#[0,0.05]#[0,0.2]# None
    vlabel         = "$P$ ($psu/mon$)"
else:
    
    vlms           = np.array([0,1.5])*1e-8#None#[0,0.05]#[0,0.2]# None
    vlabel         = "$P$ ($m/s$)"

# Preprocessing
ds_mask     = xr.where( selvar != 0. , 1 , np.nan)
selvar      = selvar * ds_mask 

fig,axs     = init_monplot()
for aa in range(12):
    ax      = axs.flatten()[aa]
    im      = plotmon[aa]
    plotvar = selvar.isel(mon=im) 
    
    # Just Plot the contour with a colorbar for each one
    if vlms is None:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap,zorder=-3)
        fig.colorbar(pcm,ax=ax)
    else:
        
        if plotcontour:
            
            pcm = ax.contourf(lon,lat,plotvar,transform=proj,
                              cmap=cmap,levels=cints_evap,zorder=-3)
            
            cl = ax.contour(lon,lat,plotvar,transform=proj,colors="k",linewidths=0.75,
                              levels=cints_prec,zorder=-3)
            ax.clabel(cl,fontsize=fsz_tick-4)
        else:
            pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                                cmap=cmap,vmin=vlms[0],vmax=vlms[1],zorder=-3)

if vlms is not None:
    cb = fig.colorbar(pcm,ax=axs.flatten(),
                      orientation='horizontal',pad=0.02,fraction=0.025)
    cb.set_label(vlabel)

plt.suptitle("%s (CESM1 Ensemble Average)" % (vname_long),fontsize=fsz_title)
savename = "%s%s_Model_Inputs_%s_seasonrow_convert%i.png" % (figpath,expname,vname,convert_precip)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# ------------------------------------------
#%% Visualize Total Ekman Forcing
# ------------------------------------------

# Set some parameters
vname          = 'Qek'
vname_long     = "Ekman Forcing"
vlabel         = "$Q_{ek}$ (psu/mon)"
plotcontour    = False
vlms           = np.array([0,3]) * 1e-8
cints_sp       = None# np.arange(0,66,12)#None#np.arange(200,1500,100)# None
cmap           = 'cmo.amp'

# Get variable, lat, lon
selvar      = inputs_ds[vname].sum('mode')
lon         = selvar.lon
lat         = selvar.lat

# Preprocessing
ds_mask     = xr.where( selvar != 0. , 1 , np.nan)
selvar      = selvar * ds_mask 

fig,axs = init_monplot()
for aa in range(12):
    ax      = axs.flatten()[aa]
    im      = plotmon[aa]
    plotvar = selvar.isel(mon=im) 
    
    # Just Plot the contour with a colorbar for each one
    if vlms is None:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap,zorder=-3)
        fig.colorbar(pcm,ax=ax)
    else:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                            cmap=cmap,vmin=vlms[0],vmax=vlms[1],zorder=-3)

if vlms is not None:
    cb = fig.colorbar(pcm,ax=axs.flatten(),
                      orientation='horizontal',pad=0.02,fraction=0.025)
    cb.set_label(vlabel)

plt.suptitle("%s (CESM1 Ensemble Average)" % (vname_long),fontsize=fsz_title)
savename = "%s%s_Model_Inputs_%s_seasonrow.png" % (figpath,expname,vname)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Plot Monthly Values of each input
"""
For SSS

PRECIP
LHFL
Qek
Correction Factor(PRECIP)
Correction Factor(LHFLX)

For SST
 dict_keys(['correction_factor', 'Fprime', 'lbd_a', 'Qek', 'alpha'])
"""

vname_exp = expparams['varname']

if vname_exp == "SSS":
    
    prectot         = stdsqsum(convdict['PRECTOT'],0) # Precipitation  # Mon x Lat x Lon
    prectot_corr    = stdsqsum(convdict['correction_factor_prec'],0) # Precipitation 
    
    lhflx           = stdsqsum(convdict['LHFLX'],0)
    lhflx_corr      = stdsqsum(convdict['correction_factor_evap'],0)
    
    Qek             = stdsqsum(convdict['Qek'],0)
    
    alpha           = stdsqsum(convdict['alpha'],0)
    
    plotvars = [prectot,prectot_corr,lhflx,lhflx_corr,Qek,alpha]
    pvnames  = ["Precipitation" ,"Precipitation Correction" ,"Evaporation","Evaporation Correction","Ekman Forcing","Total Forcing"]
    pvcmaps  = ["cmo.rain"      ,"cmo.rain"                 ,"cmo.haline" ,"cmo.haline"            ,"cmo.amp"      ,"cmo.deep"]
    
elif vname_exp == "SST":
    
    
    fprime          = stdsqsum(convdict['Fprime'],0)
    corrfac         = stdsqsum(convdict['correction_factor'],0)
    
    Qek             = stdsqsum(convdict['Qek'],0)
    
    alpha           = stdsqsum(convdict['alpha'],0)
    
    plotvars = [fprime,corrfac,Qek,alpha]
    pvnames  = ["Stochastic Heat Flux"  ,"Heat Flux Correction" ,"Ekman Forcing"    ,"Total Forcing"]
    pvcmaps  = ["cmo.thermal"           ,"cmo.thermal"          ,"cmo.thermal"      ,"cmo.thermal"]

#% ['correction_factor_evap', 'LHFLX', 'correction_factor_prec', 'PRECTOT', 'lbd_a', 'Qek', 'Qfactor', 'alpha']

#%% First figure, visualize the total forcing for each season

plotfrac        = False
plotcontour     = True

if vname_exp == "SSS":
    nsp = 6
    figsize  = (28,4.75)
    vlim_in = [0,0.025]
    vstep = 0.002
    vunit = "$psu$"
elif vname_exp =="SST":
    figsize  = (18,4.75)
    nsp = 4
    vlim_in = [0,0.85]
    vstep = 0.05
    vunit = "$\degree C$"

fig,axs,mdict = viz.init_orthomap(1,nsp,bboxplot,figsize=figsize,constrained_layout=True,centlat=45)

for a,ax in enumerate(axs):
    
    # Plot Some Things
    ax = viz.add_coast_grid(ax,bbox=bboxplot,proj=proj,fill_color='lightgray')
    ax.set_title(pvnames[a],fontsize=fsz_title)
    
    plotvar = plotvars[a]
    if len(plotvar.shape)>2:
        plotvar = plotvar.mean(0)
    
    if plotfrac: # This is still giving funky answers
        alpha_denom = alpha.mean(0)
        alpha_denom[alpha_denom==0] = np.nan
        plotvar = plotvar/alpha_denom
        vlims = [0,1]
    else:
        vlims = vlim_in
        
        
    # Apply the mask
    coords  = dict(lat=lat,lon=lon)
    plotvar = xr.DataArray(plotvar,coords=coords)
    plotvar = plotvar * mask_reg
    
    if plotcontour:
        levels  = np.arange(vlims[0],vlims[1],vstep)
        pcm     = ax.contourf(lon,lat,plotvar,transform=proj,levels=levels,cmap=pvcmaps[a],extend="both")
        cl      = ax.contour(lon,lat,plotvar,transform=proj,levels=levels,colors="k",linewidths=0.75)#vmax=vlims[1],cmap=pvcmaps[a])
        ax.clabel(cl)
    else:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,vmin=vlims[0],vmax=vlims[1],cmap=pvcmaps[a])
    
    cb  = viz.hcbar(pcm,ax=ax,fraction=0.035)
    cb.ax.tick_params(labelsize=fsz_tick-3,rotation=45)
    
    # Plot Additional Features --- --- --- --- --- --- --- --- 
    # Plot Gulf Stream Position
    ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=1.5,c='red',ls='dashdot')
    
    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=1.5,
               transform=proj,levels=[0,1],zorder=2)
    
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    
plt.suptitle("Mean Std. Dev. for %s Forcing Terms [%s/mon]" % (vname_exp,vunit),fontsize=fsz_title)

savename = "%s%s_Model_Inputs_%s_MeanStdev.png" % (figpath,expname,vname_exp)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Look at regional averages of parameters

bbname    = "Sargasso Sea"
sel_box   = [-70,-55,35,40] # Sargasso Sea SSS CSU

# bbname     = "North Atlantic Current"
# sel_box    =  [-40,-30,40,50] # NAC

# bbname     = "Irminger Sea"
# sel_box    =  [-40,-25,50,60] # Irminger

varkeys = list(conv_da.keys())
nkeys   = len(varkeys)
da_reg  = []
for nk in range(nkeys):
    dar         = proc.sel_region_xr(conv_da[varkeys[nk]],sel_box)
    #dar_avg     = dar#.mean('lat').mean('lon')
    
    if len(dar.shape) > 3:
        dar = stdsqsum(dar,0)
    
    
    da_reg.append(dar)
    
da_reg.append(proc.sel_region_xr(inputs_ds['lbd_d'],sel_box))


# Set som consistent colors for plotting

if vname_exp == "SST":
    
    # varkeys: ['correction_factor', 'Fprime', 'lbd_a', 'Qek', 'alpha']
    vname_long_pt = ["Heat Flux Correction" ,"Stochastic Heat Flux"  ,"Atmospheric Heat Flux Feedback","Ekman Forcing" ,"Total Forcing"]
    vcolors       = ["plum"                 ,"darkviolet"            ,"red"                           ,"cornflowerblue","k"]
    vls           = ['dotted'               ,"dashed"                ,"solid"                         ,"dashed"        ,"solid"  ]
    
    
elif vname_exp == "SSS":
    
    skipvar       = ["lbd_a","Qfactor"]
    vname_long_pt = ["Evaporation Correction","Evaporation","Precipitation Correction" ,"Precipitation" ,"Atmospheric Heat Flux Feedback","Ekman Forcing","Correction Fator","Total Forcing"]
    vcolors       = ["khaki"                 ,"darkkhaki"  ,"darkturquoise"            ,"teal"          ,"gray"                          ,"cornflowerblue",'pink'           ,"k"]
    vls           = ["dotted"                ,"dashed"     ,"dotted"                   ,"dashed"        ,"solid"                         ,"dashed"        ,"dotted"         ,"solid"]
dcol = "navy"

    
#%% Plot the values

fig,ax = viz.init_monplot(1,1,figsize=(6,4))
for nk in range(nkeys):
    if varkeys[nk] in skipvar:
        print("skipping %s" % varkeys[nk] )
        continue
    plotvar = da_reg[nk]
    mu      = np.nanmean(plotvar,(1,2)) ##mean('lat').mean('lon')
    ax.plot(mons3,mu,label=vname_long_pt[nk],lw=2.5,c=vcolors[nk],ls=vls[nk])
    #ax.plot(mons3,mu,label=varkeys[nk],lw=2.5,)

# Plot deep damping
ax2 = ax.twinx()
ax2.plot(mons3,np.nanmean(da_reg[-1],(1,2)),c=dcol,lw=2.5)
ax2.set_ylabel("Detrainment Correlation")
ax2.spines['right'].set_color(dcol)
ax2.yaxis.label.set_color(dcol)
ax2.tick_params(axis='y', colors=dcol)

ax.legend(ncol=3,bbox_to_anchor=[1.1,1.15,0.1,0.1],fontsize=8)
ax.set_title("Monthly Variance for %s Forcing\nBounding Box: %s, %s" % (vname_exp,bbname,sel_box),y=1.25)
ax.set_ylabel("Forcing Amplitude (%s/month) \nor Damping (1/month)" % (vunit))
ax.set_xlabel("Month")


ax.tick_params(rotation=45)


savename = "%s%s_Model_Inputs_%s_region_%s.png" % (figpath,expname,vname,bbname)
print(savename)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# da_regss = [stdsqsum(da,0) for da in da_reg]

# ====================================================
#%% Visualize amplitude of the terms (at each location)
# ====================================================

vname   = "SSS"

combine_correction = True

if vname == "SST":
    vv = 0
    vunit = "\degree C"
elif vname == "SSS":
    vv = 1
    vunit = "psu"

fig,axs = viz.init_monplot(2,3,constrained_layout=True,figsize=(14.5,6.5))

for ip in range(3):
    
    pxy = ptcoords[ip]
    lonf,latf = pxy
    
    for ss in range(2):
        ax = axs[ss,ip]
        
        if ss == 0: # Forcings
            vlab = "Forcing [$%s/mon$]" % vunit
            if vv == 0:
                plotvars = [stdsqsum(convda_byvar[0]['Fprime'].sel(lon=lonf,lat=latf,method='nearest'),0),
                            stdsq(convda_byvar[0]['correction_factor'].sel(lon=lonf,lat=latf,method='nearest')),
                            stdsqsum(convda_byvar[0]['Qek'].sel(lon=lonf,lat=latf,method='nearest'),0),
                            stdsq(convda_byvar[0]['correction_factor_Qek'].sel(lon=lonf,lat=latf,method='nearest')),
                            stdsqsum(convda_byvar[0]['alpha'].sel(lon=lonf,lat=latf,method='nearest'),0),
                            stdsq(convda_byvar[0]['Qfactor'].sel(lon=lonf,lat=latf,method='nearest')),
                            ]

                    
                    
                    
                
                plotnames = ["F'",
                             "F' Correction",
                             "Qek",
                             "Qek Correction",
                             "Total Forcing",
                             "Total Correction"]
                
                if combine_correction:
                    plotvars[1] = plotvars[1] + plotvars[0] #F'
                    plotvars[3] = plotvars[3] + plotvars[2] #Qek
                    plotnames[1] = "F' Total"
                    plotnames[3] = "Qek Total"
                    
                
                plotcols  = ["firebrick",
                             "hotpink",
                             "navy",
                             "cornflowerblue",
                             "k",
                             "gray"]
            elif vv == 1:
                
                plotvars = [
                    stdsqsum(convda_byvar[1]['LHFLX'].sel(lon=lonf,lat=latf,method='nearest'),0),
                    stdsq(convda_byvar[1]['correction_factor_evap'].sel(lon=lonf,lat=latf,method='nearest')),
                    stdsqsum(convda_byvar[1]['PRECTOT'].sel(lon=lonf,lat=latf,method='nearest'),0),
                    stdsq(convda_byvar[1]['correction_factor_prec'].sel(lon=lonf,lat=latf,method='nearest')),
                    stdsqsum(convda_byvar[1]['Qek'].sel(lon=lonf,lat=latf,method='nearest'),0),
                    stdsq(convda_byvar[1]['correction_factor_Qek'].sel(lon=lonf,lat=latf,method='nearest')),
                    stdsqsum(convda_byvar[1]['alpha'].sel(lon=lonf,lat=latf,method='nearest'),0),
                    stdsq(convda_byvar[1]['Qfactor'].sel(lon=lonf,lat=latf,method='nearest')),
                    ]
                
                plotnames = ["E'",
                             "E' Correction",
                             "P'",  
                             "P' Correction",
                             "Qek",
                             "Qek Correction",
                             "Total Forcing",
                             "Total Correction"]
                
                plotcols  = ["firebrick",
                             "hotpink",
                             "purple",
                             "magenta",
                             "navy",
                             "cornflowerblue",
                             "k",
                             "gray"]
                
                
                if combine_correction:
                    plotvars[1]  = plotvars[1] + plotvars[0] #E'
                    plotvars[3]  = plotvars[3] + plotvars[2] #P'
                    plotvars[5]  = plotvars[5] + plotvars[4] #Qek
                    plotnames[1] = "E' Total"
                    plotnames[3] = "P' Total"
                    plotnames[5] = "Qek Total"
                    
                    plotvars.append(plotvars[3] + plotvars[1])
                    plotnames.append("E-P Total")
                    plotcols.append("limegreen")
                    
                    
                

                
            ax.set_title(ptnames[ip],fontsize=fsz_title)
            
        elif ss == 1:
            vlab = "Timescale [$mon$]"
            if vv == 0:
                
                plotvars  = [1/convda_byvar[0]['lbd_a'].sel(lon=lonf,lat=latf,method='nearest'),
                             lbdd_sst_conv.sel(lon=lonf,lat=latf,method='nearest')]
                plotnames = ["Atmospheric Damping Timescale","Deep Damping Timescale"]
                plotcols  = ["limegreen","navy"]
                
                
            elif vv == 1:
                plotvars  = [ lbdd_sss_conv.sel(lon=lonf,lat=latf,method='nearest'),]
                plotnames = ["Deep Damping Timescale"]
                plotcols  = ["navy"]
        else:
            continue
        # elif ss == 1: # Dampings
        #     if vv == 0:
        #         plotvars = []
        #plotvars = [pv.sel(lon=lonf,lat=latf,method='nearest') for pv in plotvars]
        
        if ip == 0: # Add Label
            lab = "%s %s" % (vname,vlab)
            viz.add_ylabel(lab,ax=ax)
            
            
        for zz in range(len(plotvars)):
            if combine_correction:
                if "Total" not in plotnames[zz]:
                    continue
            
            ax.plot(mons3,plotvars[zz],label=plotnames[zz],c=plotcols[zz],lw=2.5,marker="o")
        if ip == 0:
            ax.legend(framealpha=0.5)
        
        if ss == 1 and vv == 1: # Add SST_Evaporation Feedbacl
            lbde_plot = lbd_emon.sel(lon=lonf,lat=latf,method='nearest')
            ax2 = ax.twinx()
            ax2.plot(mons3,lbde_plot,c="violet",label="SST-evaporation feedback")
            ax2.set_ylabel("SST-evaporation feedback [psu/degC/mon]")
                
savename = "%sModel_Inputs_Point_%s_%s.png" % (figpath,pointset,vname)
print(savename)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# ----------------------------------------------------------------------------
#%% Section: Check Correction Factor
# ----------------------------------------------------------------------------

lonf           = -36
latf           = 50
dtmon          = 3600*24*30
locfn,loctitle = proc.make_locstring(lonf,latf)

eofvars = [
    paramset_byvar[0][1]['Fprime'], # Fprime
    paramset_byvar[1][1]['LHFLX'], # Eprime,
    paramset_byvar[1][1]['PRECTOT'], # Eprime,
    paramset_byvar[0][1]['Qek'], # Qek SST,
    paramset_byvar[1][1]['Qek'], # Qek SSS,
    ]

cfs = [
       paramset_byvar[0][1]['correction_factor'],       # Fprime
       paramset_byvar[1][1]['correction_factor_evap'],  # Eprime,
       paramset_byvar[1][1]['correction_factor_prec'],  # Precipitation
       paramset_byvar[0][1]['correction_factor_Qek'],   # Qek SST
       paramset_byvar[1][1]['correction_factor_Qek'],   # Qek SSS
       ]

eofvars_pt = [stdsqsum(ds.sel(lon=lonf,lat=latf,method='nearest'),0) for ds in eofvars]
cfs_pt     = [np.sqrt(ds.sel(lon=lonf,lat=latf,method='nearest')**2) for ds in cfs]
#cfs_pt    = [ds.sel(lon=lonf,lat=latf,method='nearest') for ds in cfs]
monvars_pt = [ds.sel(lon=lonf,lat=latf,method='nearest').mean('ens') for ds in dsmonvar_all]


# Load and compare some dummy vars
nceprime_old = rawpath + "CESM1_HTR_FULL_Eprime_nroll0_NAtl_EnsAvg.nc"
eprimeold    = xr.open_dataset(nceprime_old).load()['LHFLX']#.std('month')
eprimept     = eprimeold.sel(lon=lonf,lat=latf,method='nearest')

#%% Check the Forcings for the point


fig,axs = viz.init_monplot(1,5,figsize=(22,3.5))

for vv in range(5):
    
    ax      = axs[vv]
    
    eofin   = eofvars_pt[vv]
    cfin    = cfs_pt[vv]
    alpha   = eofin + cfin
    monvar  = np.sqrt(monvars_pt[vv])
    
    # Plot EOF
    ax.plot(mons3,monvar,label="Monthly Variance",c="k")
    ax.plot(mons3,eofin,label="EOFs Total",c='blue')
    ax.plot(mons3,cfin,label="Correction Factor",c='limegreen')
    ax.plot(mons3,alpha,label="Total Forcing",c='cyan',ls='dashed')
    
    
    
    ax.set_title(monvar.name)
    ax.legend()
    
#%% Visualize some of the other inputs (Damping, MLD, etc)


lonf    = -65
latf    = 36
hff_pt  = proc.selpt_ds(convda_byvar[0]['lbd_a'],lonf,latf)
lbda_pt = proc.selpt_ds(paramset_byvar[0][1]['lbd_a'],lonf,latf)
    
fig,axs  = viz.init_monplot(2,1)

# Plot Pointwise Heat Flux Feedback
ax = axs[0]
ax.plot(mons3,hff_pt,label="lbd_a",marker="o")
ax.legend()
ax.set_ylabel("lbd_a/(rho*cp*h)")

#
ax = axs[1]
ax.plot(mons3,lbda_pt,label="lbd_a",marker="o")
ax.legend()
ax.set_ylabel("lbd_a")

    
    






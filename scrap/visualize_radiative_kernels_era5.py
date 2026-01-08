#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Radiative Kernels ERA5

Created on Fri Nov  7 15:58:24 2025

@author: gliu

"""


import sys
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
import matplotlib.gridspec as gridspec
from scipy.io import loadmat
import matplotlib as mpl
import climlab

#%% Import Custom Modules

amvpath = "/home/niu4/gliu8/scripts/commons"

sys.path.append(amvpath)
from amv import proc,viz

ensopath = "/home/niu4/gliu8/scripts/ensobase"
sys.path.append(ensopath)
import utils as ut

#%%

regrid1x1   = True
expname     = "ERA5_1979_2024"
datpath     = "/home/niu4/gliu8/projects/common_data/ERA5/anom_detrend1/"
outpath     = "/home/niu4/gliu8/projects/ccfs/"

if regrid1x1:
    datpath = "/home/niu4/gliu8/projects/common_data/ERA5/regrid_1x1/anom_detrend1/"
    outpath = "/home/niu4/gliu8/projects/ccfs/regrid_1x1/"
    
    
figpath     = "/home/niu4/gliu8/figures/bydate/2025-11-18/"
proc.makedir(figpath)

flxnames     = ['cre',]#'allsky','clearsky']# ['cre',]
ccf_vars     = ["sst","eis","Tadv","r700","w700","ws10",]#"ucc"] 
ncstr        = datpath + "%s_1979_2024.nc"  
selmons_loop = [None,]#[[12,1,2],[3,4,5],[6,7,8],[9,10,11]] # Set to None to do 


tstart   = '1979-01-01'
tend     = '2024-12-31'
timename = 'valid_time'
latname  = 'latitude'
lonname  = 'longitude'

#% Load Land Mask
landnc   = datpath + "mask_1979_2024.nc"
landmask = xr.open_dataset(landnc).mask



# MLR Calculation Options
standardize = True # Set to True to standardize predictors before MLR
fill_value  = 0    # Replace NaN values with <fill_value>
add_ucc     = False # Set to True to include upper cloud concentration as a predictor


# Make sure they all have the right shape, dim names
def preprocess(ds,tstart,tend,timename,latname,lonname):
    print("\n now preprocessing")
    rename_dict = {
        timename : 'time',
        latname : 'lat',
        lonname : 'lon'
        }
    if 'time' not in ds.coords or timename not in ds.coords:
        del rename_dict[timename]
    if 'lat' in ds.coords:
        print("lat already found...")
        del rename_dict[latname]
        #rename_dict = rename_dict
    if 'lon' in ds.coords:
        print("lon already found...")
        del rename_dict[lonname]
    ds = ds.rename(rename_dict)
    ds = ut.standardize_names(ds)
    
    ds = ds.squeeze()
    
    
    if 'time' in ds.coords and len(ds.shape) >= 3:
        ds = ds.sel(time=slice(tstart,tend))
    return ds

#%% Load the calculation output


flxname     = 'cre'
selmons     = None


outname         = "%s%s_%s_CCFs_Regression_standardize%i_adducc%i.nc" % (outpath,expname,flxname,standardize,add_ucc)
if selmons is not None:
    selmonstr = proc.mon2str(np.array(selmons)-1)
    outname      = proc.addstrtoext(outname,"_"+selmonstr,adjust=-1)
if regrid1x1:
    outname      = proc.addstrtoext(outname,"_regrid1x1",adjust=-1)
dskernel          = xr.open_dataset(outname).load()


# Load the flux
ncsearch    = ncstr % (flxname)    
foundnc     = glob.glob(ncsearch)
print("Found the following for %s:" % flxname)
print("\t"+ str(foundnc))
try:
    dsflx = xr.open_dataset(foundnc[0])[flxname]
except:
    print("No flux found... (%s)" % flxname)
dsflx_anom = preprocess(dsflx,tstart,tend,timename,latname,lonname)


#%% Make Scott et al style plot
dtday = 3600*24 # ERA5 fluxes are accumulated over 1 day (https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Meanrates/fluxesandaccumulations)

proj    = ccrs.PlateCarree()

fig,axs = ut.init_globalmap(nrow=3,ncol=2,figsize=(12,12))

nccfs   = len(ccf_vars)
if add_ucc:
    nccfs = nccfs-1
for cc in range(nccfs):
    
    ax      = axs.flatten()[cc]
    
    plotvar = dskernel.coeffs.isel(ccf=cc)/dtday
    title  = r"d(%s) / d (%s)" % (flxname,plotvar.ccf.item())
    
    ax.set_title(title)
    pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                        vmin=-5,vmax=5,cmap='cmo.balance',
                        )

plt.suptitle("%s (%s) MLR Coefficients" % (flxname,expname),y=0.95)
cb = viz.hcbar(pcm,ax=axs.flatten(),pad=0.01,fraction=0.015)
cb.set_label(r"%s Feedback [W $m^{-2}$ $\sigma^{-1}$]" % flxname)
savename = "%s%s_%s_CCFs_Map_Combine_adducc%i.png" % (figpath,flxname,expname,add_ucc)
plt.savefig(savename,dpi=150,bbox_inches='tight')
#plt.show()


#%% Check R2

fig,ax      = ut.init_globalmap(1,1,figsize=(8,3.5))

vlims = [0,1]

plotvar     = dskernel.r2 #varsin[vv]#.mean('time')
    
# if plotvar.name == "sst" and :
#     plotvar = plotvar + 273.15

pcm         = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,cmap='cmo.deep',vmin=vlims[0],vmax=vlims[1])
cb              = viz.hcbar(pcm,ax=ax,fraction=0.045,pad=0.01)


ax.set_title(plotvar.name)
savename    = "%s%s_CCF_R2_TimeMean_check.png" % (figpath,"ERA5")
plt.savefig(savename,dpi=150,bbox_inches='tight')

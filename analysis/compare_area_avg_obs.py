#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Variance/Prsoperties of Area-Averaged SSTs for sets of observational SSTs

Created on Tue May 13 21:53:09 2025

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

#%% indicate cropping region and check for a folder

# Indicate Path to Area-Average Files
dpath_aavg      = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/region_average/"
regname         = "SPGNE"
bbsel           = [-40,-15,52,62]

regname         = "NNAT"
bbsel           = [-80,0,20,60]


# Set up Output Path
locfn,locstring = proc.make_locstring_bbox(bbsel,lon360=True)
bbfn            = "%s_%s" % (regname,locfn)
outpath         = "%s%s/" % (dpath_aavg,bbfn)
proc.makedir(outpath)
print("Region is        : %s" % bbfn)
print("Output Path is   : %s" % outpath)

# String Format is <bbfn>/<expname>_<vname>_<ystart>_<yend>_<procstring>.nc

#%% Indicate Which Experiments to Load

expnames = ["ERA5 (1979-2024)",
            "OISST (1982-2020)",
            "HadISST (1920-2017)",
            #"ERSST5",
            #"EN4"
            ]

ncnames = [
    'ERA5_sst_1979_2024_raw_IceMask5.nc',
    'OISST_sst_1982_2020_raw_IceMaskMax5.nc',
    'HadISST_SST_1920_2017_detrend_deseason.nc',
    #'ERSST5_sst_1854_2017_raw.nc',
    #'EN4_sst_1900_2021_raw.nc'
    ]

vname = "sst"

restrict_time = True
ystart = 1982
yend   = 2017

nexps = len(expnames)
#%% Load the experiments

dsall = []
for ex in range(nexps):
    #ncsearch = "%s%s*.nc" % (outpath,expnames[ex])
    #nclist   = glob.glob(ncsearch)
    ds = xr.open_dataset(outpath + ncnames[ex]).load()[vname]
    
    if np.any(np.isnan(ds)):
        print("NaN detected in %s" % (expnames[ex]))
        
    if restrict_time:
        ds = ds.sel(time=slice('%04i-01-01' % ystart,"%04i-12-31" % yend))
        
        
    dsall.append(ds)


    


#%% Preprocess Timeseries

def preprocess_ds(ds):
    dsa      = proc.xrdeseason(ds)
    #dsa      = xr.where(np.isnan(dsa),0,dsa)
    #dsa_dt  = sp.signal.detrend(dsa)
    dsa_dt  = proc.xrdetrend_1d(dsa,3)
    return dsa_dt

dsa = [preprocess_ds(ds) for ds in dsall]
stds = [ds.std('time') for ds in dsa]
dsa_lp = [proc.lp_butter(ds,120,6) for ds in dsa]
stds_lp = [np.nanstd(ts) for ts in dsa_lp]

#%% Check Spread

instd       = stds
instd_lp    = stds_lp

vratio      = np.array(instd_lp) / np.array(instd) * 100

xlabs       = ["%s\n%.2f" % (expnames[ii],vratio[ii])+"%" for ii in range(len(vratio))]
fig,ax      = plt.subplots(1,1,constrained_layout=True,figsize=(6,6))

braw        = ax.bar(np.arange(nexps),instd,color='gray')
blp         = ax.bar(np.arange(nexps),instd_lp,color='k')

ax.bar_label(braw,fmt="%.04f",c='gray')#color=expcols)#c='gray')
ax.bar_label(blp,fmt="%.04f",c='k')

ax.set_xticks(np.arange(nexps),labels=xlabs)
ax.set_ylabel("$\sigma$(SST) [$\degree$C]")
ax.set_ylim([0,1.0])
#ax.set_title("%s (%s)" % (bbname,bbstr))

ax.grid(True,ls='dotted',lw=0.55,c='gray')


#stds = [ds.std('tim')]



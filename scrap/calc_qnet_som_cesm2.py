#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate QNET in SOM CESm2 PiControl
Use output from [oreproc_cesm2]

Created on Wed Jul  2 14:54:59 2025

@author: gliu

"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

#%% Load Fluxes

vnames   = ["FSNS","FLNS","SHFLX","LHFLX"]
dpath    = "/stormtrack/data4/glliu/01_Data/CESM2_PiControl/proc/NAtl/"
ncsearch = dpath + "CESM2_SOM_%s_NAtl_0060to0360.nc"
dsall    = xr.merge([xr.open_dataset(ncsearch % vn).load() for vn in vnames])


#%% Check Sign

itime = 0 
fig,axs = plt.subplots(1,4,figsize=(12,4),constrained_layout=True)
for vv in range(4):
    ax = axs[vv]
    
    vname = vnames[vv]
    pcm = dsall[vname].isel(time=itime).plot(ax=ax,vmin=-100,vmax=100,cmap='RdBu_r')
    #plt.colorbar(pcm)
    
plt.show()

#%% Compute qnet

qnet    = dsall.FSNS - (dsall.FLNS + dsall.SHFLX + dsall.LHFLX)
qnet    = qnet.rename("SHF")

outname = ncsearch % "SHF"
edict   = {"SHF":{'zlib':True}}
qnet.to_netcdf(outname,encoding=edict)




    
    
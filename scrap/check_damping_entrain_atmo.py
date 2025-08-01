#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Look at the total damping between atmospheric and entrainment damping to
 identify where a simulation might blow up


Copied Upper Section of run SSS basinwide

Check the sign of damping

Created on Tue Jun  3 14:08:51 2025

@author: gliu
"""


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
sys.path.append("../")
cwd = os.getcwd()
sys.path.append(cwd+ "/..")
import reemergence_params as rparams

# Paths and Load Modules
pathdict   = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])
from amv import proc,viz
import amv.loaders as dl
import scm
import amv.loaders as dl
import yo_box as ybx

# Set needed paths
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
# procpath    = pathdict['procpath']
# figpath     = pathdict['figpath']
# proc.makedir(figpath)

#%% 

"""
Paste Experiment Parameters Below (see basinwide_experiment_params.py)

"""
figpath     = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20250606/"
proc.makedir(figpath)


expname     = "SST_Obs_Pilot_00_Tdcorr0_qnet_AConly_SPGNE"
expparams   = {
    'varname'           : "SST",
    'bbox_sim'          : [-40,-15,52,62],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : 'SST_Obs_Pilot_00_Tdcorr0_qnet', # If not None, load a runid from another directory
    'Fprime'            : "ERA5_Fprime_QNET_std_pilot.nc",
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "MIMOC_regridERA5_h_pilot.nc",
    'lbd_d'             : None,
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "MIMOC_regridERA5_kprev_pilot.nc",
    'lbd_a'             : "ERA5_qnet_damping_AConly.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
    'Qek'               : None, # Now in degC/sec
    'convert_Fprime'    : True,
    'convert_lbd_a'     : True, 
    'convert_PRECTOT'   : False,
    'convert_LHFLX'     : False,
    'froll'             : 0,
    'mroll'             : 0,
    'droll'             : 0,
    'halfmode'          : False,
    "entrain"           : True,
    "eof_forcing"       : False, # CHECK THIS
    "Td_corr"           : False, # Set to True if lbd_d is provided as a correlation, rather than 1/months
    "lbd_e"             : None, # Relevant for SSS
    "Tforce"            : None, # Relevant for SSS
    "correct_Qek"       : False, # Set to True if correction factor to Qek was calculated
    "convert_Qek"       : False, # Set to True if Qek is in W/m2 (True for old SST forcing...) False if in psu/sec or degC/sec (for new scripts)
    }



#%% Other Constants

# Constants
dt    = 3600*24*30 # Timestep [s]
cp    = 3850       # 
rho   = 1026       # Density [kg/m3]
B     = 0.2        # Bowen Ratio, from Frankignoul et al 1998
L     = 2.5e6      # Specific Heat of Evaporation [J/kg], from SSS model document

debug = False

print("==========================")
print("Now Running Experiment: %s" % expname)
print("==========================")

#%% Check and Load Params

print("\tLoading inputs for %s" % expname)

# Apply patch to expdict
expparams = scm.patch_expparams(expparams)

# First, Check if there is EOF-based forcing (remove this if I eventually redo it)
if expparams['eof_forcing']:
    print("\t\tEOF Forcing Detected.")
    eof_flag = True
else:
    print("\t\tEOF Forcing will not be used.")
    eof_flag = False

inputs,inputs_ds,inputs_type,params_vv = scm.load_params(expparams,input_path)

#%% Detect and Process Missing Inputs

_,nlat,nlon=inputs['h'].shape

# Get number of modes
if eof_flag:
    if expparams['varname'] == "SST":
        nmode = inputs['Fprime'].shape[0]
    elif expparams['varname'] == "SSS":
        nmode = inputs['LHFLX'].shape[0]

#%% For Debugging

lonf = -50
latf = 40
dsreg =inputs_ds['h']
latr = dsreg.lat.values
lonr = dsreg.lon.values
klon,klat=proc.find_latlon(lonf,latf,lonr,latr)

#%% Initialize An Experiment folder for output

expdir = output_path + expname + "/"
proc.makedir(expdir + "Input")
proc.makedir(expdir + "Output")
proc.makedir(expdir + "Metrics")
proc.makedir(expdir + "Figures")

# Save the parameter file
savename = "%sexpparams.npz" % (expdir+"Input/")
chk = proc.checkfile(savename)
if chk is False:
    print("Saving Parameter Dictionary...")
    np.savez(savename,**expparams,allow_pickle=True)

# Load out some parameters
runids = expparams['runids']
nruns  = len(runids)

froll = expparams['froll']
droll = expparams['droll']
mroll = expparams['mroll']

# Make a function to simplify rolling
def roll_input(invar,rollback,halfmode=False,axis=0):
    rollvar = np.roll(invar,rollback,axis=axis)
    if halfmode:
        rollvar = (rollvar + invar)/2
    return rollvar


#%% Do Conversions for Model Run ------------------------------------------


# Apply roll/shift to seasonal cycle
ninputs = len(inputs)
for ni in range(ninputs):
    
    pname = list(inputs.keys())[ni]
    ptype = inputs_type[pname]
    
    if ptype == "mld":
        rollback = mroll
    elif ptype == "forcing":
        rollback = froll
    elif ptype == "damping":
        rollback = droll
    else:
        print("Warning, Parameter Type not Identified. No roll performed.")
        rollback = 0
    
    if rollback != 0:
        print("Rolling %s back by %i" % (pname,rollback))
        
        if eof_flag and len(inputs[pname].shape) > 3:
            rollaxis=1 # Switch to 1st dim to avoid mode dimension
        else:
            rollaxis=0
        inputs[pname] = roll_input(inputs[pname],rollback,axis=rollaxis,halfmode=expparams['halfmode'])

# # Do Unit Conversions ---
inputs_convert  = scm.convert_inputs(expparams,inputs,dt=dt,rho=rho,L=L,cp=cp,return_sep=True)
alpha    = inputs_convert['alpha'] # Amplitude of the forcing
Dconvert = inputs_convert['lbd_a'] # Converted Damping
# # End Unit Conversion ---

# Tile Forcing (need to move time dimension to the back)
if eof_flag and expparams['qfactor_sep'] is False: # Append Qfactor as an extra mode (old approach)
    Qfactor = inputs_convert['Qfactor']
    alpha   = np.concatenate([alpha,Qfactor[None,...]],axis=0)

# Calculate beta and kprev
beta       = scm.calc_beta(inputs['h'].transpose(2,1,0)) # {lon x lat x time}
if expparams['kprev'] is None: # Compute Kprev if it is not supplied
    print("Recalculating Kprev")
    kprev = np.zeros((12,nlat,nlon))
    for o in range(nlon):
        for a in range(nlat):
            kprevpt,_=scm.find_kprev(inputs['h'][:,a,o])
            kprev[:,a,o] = kprevpt.copy()
    inputs['kprev'] = kprev


# Set parameters, and transpose to [lon x lat x mon] for old script
smconfig = {}

smconfig['h']       = inputs['h'].transpose(2,1,0)           # Mixed Layer Depth in Meters [Lon x Lat x Mon]
smconfig['lbd_a']   = Dconvert.transpose(2,1,0) # 
smconfig['beta']    = beta # Entrainment Damping [1/mon]
smconfig['kprev']   = inputs['kprev'].transpose(2,1,0)
smconfig['lbd_d']   = inputs['lbd_d'].transpose(2,1,0)
smconfig['Td_corr'] = expparams['Td_corr']



#%%

coords = dict(lon=lonr,lat=latr,mon=np.arange(1,13,1))


lbdain = xr.DataArray(smconfig['lbd_a'],coords=coords,dims=coords,name='lbda')
betain = xr.DataArray(smconfig['beta'],coords=coords,dims=coords,name='we')


#%% Plot for each mnonth
import cartopy.crs as ccrs
mons3 = proc.get_monstr()


lbd_full = lbdain + betain
proj = ccrs.PlateCarree()

for im in range(12):
    
    fig,ax,bb   = viz.init_regplot(regname="SPGE",fontsize=32)
    plotvar     = lbd_full.isel(mon=im).T
    pcm         = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                        vmin=-1,vmax=1,cmap='cmo.balance')
    cb          = viz.hcbar(pcm,ax=ax,fontsize=32)
        
    
    cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,levels=[0,])
    ax.clabel(cl)
    cb.set_label("%s Damping [degC/sec]" % mons3[im],fontsize=38)
    
#%% Plot Minimum Value

plot_total = False

if plot_total == True:
    

    minlbd = lbd_full.min('mon')
    minmon = lbd_full.argmin('mon') + 1
    outname = "Total_Damping"
    outname_long = r"Total Damping ($\lambda^a + \overline{\frac{w_e}{h}}$)"
else:
    minlbd = lbdain.min('mon')
    minmon = lbdain.argmin('mon') + 1
    outname = "AtmoDamping_Only"
    outname_long = r"Atmospheric Damping ($\lambda^a$)"

levels = np.arange(-.15,.16,0.01)

bboxSPGNE = [-40,-15,52,62]

bbplot   = [-45,-5,50,65]
fig,ax,_ = viz.init_orthomap(1,1,bbplot,centlon=-28,centlat=58,figsize=(12,10))

ax       = viz.add_coast_grid(ax,bbox=bbplot,proj=proj)


viz.plot_box(bboxSPGNE,ax=ax)

plotvar = minlbd.T
pcm = ax.contourf(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                    levels=levels,cmap='cmo.balance')
cb = viz.hcbar(pcm,ax=ax)
cb.set_label(outname_long,fontsize=24)

cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                    levels=[0,],colors='k')
ax.clabel(cl)

figname = "%sMinimum_%s.png" % (figpath,outname)
plt.savefig(figname,dpi=150)
#fig,ax,bb   = viz.init_regplot(regname="SPGE",fontsize=32)
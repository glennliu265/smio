#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Using output from [construct_enso_forcing], run stochastic model and see
how the resultant sst response is.

(1) Load Parameters
(2) Unit Conversions
(3) Integrate

Created on Wed Dec 17 11:53:59 2025

@author: gliu

"""

import sys
import time
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import xarray as xr
import sys
import glob 
import scipy as sp
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
from scipy.io import loadmat
import matplotlib as mpl

import importlib
from tqdm import tqdm

#%% Additional Modules

# local device (currently set to run on Astraeus, customize later)
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import cvd_utils as cvd

ensopath = "/Users/gliu/Downloads/02_Research/01_Projects/07_ENSO/03_Scripts/ensobase/"
sys.path.append(ensopath)
import utils as ut

# ========================================================
#%% User Edits
# ========================================================

# Load GMSST For ERA5 Detrending
detrend_obs_regression  = True
dpath_gmsst             = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
nc_gmsst                = "ERA5_GMSST_1979_2024.nc"

bbox_spgne              = [-40,-15,52,62]

# Set Figure Path
figpath  = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20251218/"
proc.makedir(figpath)

#%% Load ENSO Forcing

outpath  = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/enso/"
ensolag  = 1
lag_bymonth=True
flxnames = ['qnet','Fprime']

forcings = []
for ii in range(2):
    
    outname = "%sERA5_%s_ENSO_related_forcing_ensolag%i.nc" % (outpath,flxnames[ii],ensolag)
    if lag_bymonth:
        outname = proc.addstrtoext(outname,"_lagbymonth",adjust=-1)
    ds = xr.open_dataset(outname)[flxnames[ii]].load()
    forcings.append(ds)
    

#%% Load Parameters, let's use the ones from the Draft 2 Run
# Just need to extract h, lbd_a, and lbd_d...

expname     =  "SST_ORAS5_avg_GMSST_EOFmon_usevar"

expparams   = {
    
    'varname'           : "SST",
    'bbox_sim'          :  [-40,-15,52,62],
    'nyrs'              : 46,
    'runids'            : ["run00",],
    'runid_path'        : "SST_ENSO_qnet_pc1", # If not None, load a runid from another directory
    'Fprime'            : "ERA5_Fprime_QNET_timeseries_QNETgmsstMON_nroll0_NAtl_EOFFilt090_corrected_usevar.nc",
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "MIMOC_regridERA5_h_pilot.nc",
    'lbd_d'             : "ORAS5_avg_MIMOC_corr_d_TEMP_detrendGMSSTmon_lagmax3_interp1_ceil0_imshift1_dtdepth1_1979to2024_regridERA5.nc",
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "MIMOC_regridERA5_kprev_pilot.nc",
    'lbd_a'             : "ERA5_qnet_damping_AConly_detrendGMSSTmon.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
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
    "eof_forcing"       : True, # CHECK THIS
    "Td_corr"           : True, # Set to True if lbd_d is provided as a correlation, rather than 1/months
    "lbd_e"             : None, # Relevant for SSS
    "Tforce"            : None, # Relevant for SSS
    "correct_Qek"       : False, # Set to True if correction factor to Qek was calculated
    "convert_Qek"       : False, # Set to True if Qek is in W/m2 (True for old SST forcing...) False if in psu/sec or degC/sec (for new scripts)
    }
    
#%% Copied Below from Run SSS Basinwide

import os
rem_module_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/03_Scripts/reemergence/"
os.chdir(rem_module_path)

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

# Constants
dt    = 3600*24*30 # Timestep [s]
cp    = 3850       # 
rho   = 1026       # Density [kg/m3]
B     = 0.2        # Bowen Ratio, from Frankignoul et al 1998
L     = 2.5e6      # Specific Heat of Evaporation [J/kg], from SSS model document

debug = False



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
    
# For correction factor, check to see if "usevar" is in the forcing name
if eof_flag:
    if 'usevar' in expparams['Fprime']:
        usevar = True
        print("Using variance for white noise forcing...")
    else:
        usevar = False
inputs,inputs_ds,inputs_type,params_vv = scm.load_params(expparams,input_path)


if usevar:
    inputs['correction_factor']    = np.sqrt(inputs['correction_factor'])
    inputs_ds['correction_factor'] = np.sqrt(inputs_ds['correction_factor'])


#%%

# Load out some parameters
runids = expparams['runids']
nruns  = len(runids)

froll  = expparams['froll']
droll  = expparams['droll']
mroll  = expparams['mroll']

# Make a function to simplify rolling
def roll_input(invar,rollback,halfmode=False,axis=0):
    rollvar = np.roll(invar,rollback,axis=axis)
    if halfmode:
        rollvar = (rollvar + invar)/2
    return rollvar



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
alpha           = inputs_convert['alpha'] # Amplitude of the forcing
Dconvert        = inputs_convert['lbd_a'] # Converted Damping


#%%

# Calculate beta and kprev

beta       = scm.calc_beta(inputs['h'].transpose(2,1,0)) # {lon x lat x time}
nlon,nlat,_ = beta.shape
if expparams['kprev'] is None and expparams['entrain'] == True: # Compute Kprev if it is not supplied
    
    print("Recalculating Kprev")
    kprev = np.zeros((12,nlat,nlon))
    for o in range(nlon):
        for a in range(nlat):
            kprevpt,_=scm.find_kprev(inputs['h'][:,a,o])
            kprev[:,a,o] = kprevpt.copy()
    inputs['kprev'] = kprev


#%%


# Set parameters, and transpose to [lon x lat x mon] for old script
smconfig = {}

smconfig['h']       = inputs['h'].transpose(2,1,0)           # Mixed Layer Depth in Meters [Lon x Lat x Mon]
smconfig['lbd_a']   = Dconvert.transpose(2,1,0) # 
smconfig['beta']    = beta # Entrainment Damping [1/mon]
smconfig['kprev']   = inputs['kprev'].transpose(2,1,0)
smconfig['lbd_d']   = inputs['lbd_d'].transpose(2,1,0)
smconfig['Td_corr'] = expparams['Td_corr']

# 

# No SST-Evaporation Feedback
smconfig['add_F'] = None
    

#%% Set up the Forcing

# Convert the forcing
#dt=3600*24*30,rho=1026,L=2.5e6,cp=3850
forcings_input = []
forcing_names  = ["PC1","PC2","PCsum"]
vnames_out     = []
for ii in range(2):
    
    flxname        = flxnames[ii]
    #forcing_bytype = []
    
    
    for nn in tqdm(range(3)):
        
        # Read out Forcing
        if nn < 2: # Select Mode of the forcing
            forcing_sel = forcings[ii].isel(pc=nn).transpose('lon','lat','time')
        else: # Sum the PCs
            forcing_sel = forcings[ii].sum('pc').transpose('lon','lat','time')
        
        forcing_sel = proc.sel_region_xr(forcing_sel,expparams['bbox_sim'])
        
        # Reshape
        nlon,nlat,ntime = forcing_sel.shape
        nyr = int(ntime/12)
        forcing_sel = forcing_sel.data.reshape(nlon,nlat,nyr,12)
        
        # Convert Units (assuming it is W/m2)
        forcing_conv = forcing_sel / (rho*cp*inputs['h'].transpose(2,1,0)[:,:,None,:]) * dt
        
        forcing_conv = forcing_conv.reshape(nlon,nlat,ntime)
        
        forcings_input.append(forcing_conv)
        
        vnames_out.append("%s_%s" % (flxnames[ii],forcing_names[nn]))
        


#%% Run Simulation  

simulation_output = []
for jj in range(6):
    
    forcing_in          = forcings_input[jj]
    smconfig['forcing'] = forcing_in
    
    outdict = scm.integrate_entrain(smconfig['h'],smconfig['kprev'],smconfig['lbd_a'],smconfig['forcing'],
                                    Tdexp=smconfig['lbd_d'],beta=smconfig['beta'],add_F=smconfig['add_F'],
                                    return_dict=True,old_index=True,Td_corr=smconfig['Td_corr'])
    
    var_out  = outdict['T']
    latr     = inputs_ds['h'].lat
    lonr     = inputs_ds['h'].lon
    timedim  = forcings[0].time #xr.cftime_range(start="0001",periods=var_out.shape[-1],freq="MS",calendar="noleap")
    cdict    = {
        "time" : timedim,
        "lat" : latr,
        "lon" : lonr,
        }
    
    
    da       = xr.DataArray(var_out.transpose(2,1,0),coords=cdict,dims=cdict,name=expparams['varname'])
    edict    = {expparams['varname']:{"zlib":True}}
    
    savename = "%sERA5_SST_ENSO_Component_%s.nc" % (outpath,vnames_out[jj])
    
    #savename = "%sOutput/%s_runid%s.nc" % (expdir,expparams['varname'],runid)
    da.to_netcdf(savename,encoding=edict)
    
    

    
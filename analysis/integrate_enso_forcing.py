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
flxnames = ['qnet','Fprime']

forcings = []
for ii in range(2):
    
    outname = "%sERA5_%s_ENSO_related_forcing_ensolag%i.nc" % (outpath,flxnames[ii],ensolag)
    ds = xr.open_dataset(outname)[flxnames[ii]].load()
    forcings.append(ds)
    

#%% Load Parameters, let's use the ones from the Draft 2 Run
# Just need to extract h, lbd_a, and lbd_d...

expname     =  "SST_ORAS5_avg_GMSST_EOFmon_usevar_NATL"

expparams   = {
    
    'varname'           : "SST",
    'bbox_sim'          : [-80,0,30,65],
    'nyrs'              : 45,
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
    
#%% Copie Below from Run SSS Basinwide


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


def qfactor_noisemaker(expparams,expdir,expname,runid,share_noise=False):
    # Checks for separate wn timeseries for each correction factor
    # and loads the dictionary
    # If share noise is true, the same wn timeseries is used for:
    #     - Fprime and Evaporation
    #     - Qek forcing
    #     - Precipe
    
    # Makes (and checks for) additional white noise timeseries for the following (6) correction factors
    forcing_names = ("correction_factor",           # Fprime
                     "correction_factor_Qek_SST",   # Qek_SST
                     "correction_factor_evap",      # Evaporation
                     "correction_factor_prec",      # Precip
                     "correction_factor_Qek_SSS",   # Qek_SSS
                     )
    nforcings     = len(forcing_names)
    
    # Check for correction file
    noisefile_corr = "%sInput/whitenoise_%s_%s_corrections.npz" % (expdir,expname,runid)
    
    # Generate or reload white noise
    if len(glob.glob(noisefile_corr)) > 0:
        
        print("\t\tWhite Noise correction factor file has been found! Loading...")
        wn_corr = np.load(noisefile_corr)
        
        if share_noise:
            print("Checking for shared noise...")
            if wn_corr['correction_factor'] != wn_corr['correction_factor_evap']:
                print("\tSetting F' and E' white noise to be the same")
                wn_corr['correction_factor_evap'] = wn_corr['correction_factor']
            if wn_corr['correction_factor_Qek_SSS'] != wn_corr['correction_factor_Qek_SST']:
                print("\tSetting Qek white noise to be the same")
                wn_corr['correction_factor_Qek_SSS'] = wn_corr['correction_factor_Qek_SST']
                
        
    else:
        
        print("\t\tGenerating %i new white noise timeseries: %s" % (nforcings,noisefile_corr))
        noise_size  = [expparams['nyrs'],12,]
        
        wn_corr_out = {}
        for nn in range(nforcings):
            
            if (forcing_names[nn] == "correction_factor_evap") and share_noise:
                print("Copying same white noise timeseries as F' for E'")
                wn_corr_out[forcing_names[nn]] = wn_corr_out['correction_factor']
            elif (forcing_names[nn] == "correction_factor_Qek_SSS") and share_noise:
                print("Copying same white noise timeseries as Qek SST for Qek SSS")
                wn_corr_out[forcing_names[nn]] = wn_corr_out['correction_factor_Qek_SST']
            else: # Make a new noise timeseries
                wn_corr_out[forcing_names[nn]] = np.random.normal(0,1,noise_size) # [Yr x Mon x Mode]
        
        np.savez(noisefile_corr,**wn_corr_out,allow_pickle=True)
        wn_corr = wn_corr_out.copy()
        
    return wn_corr

#%%  
for nr in range(nruns):
    
    #%% Prepare White Noise timeseries ----------------------------------------
    runid = runids[nr]
    print("\tPreparing forcing...")
    
    # Check if specific path was indicated, and set filename accordingly
    if expparams['runid_path'] is None:
        noisefile = "%sInput/whitenoise_%s_%s.npy" % (expdir,expname,runid)
    else:
        expname_runid = expparams['runid_path'] # Name of experiment to take runid from
        print("\t\tSearching for runid path in specified experiment folder: %s" % expname_runid)
        noisefile     = "%sInput/whitenoise_%s_%s.npy" % (output_path + expname_runid + "/",expname_runid,runid)
    
    # Generate or reload white noise
    if len(glob.glob(noisefile)) > 0:
        print("\t\tWhite Noise file has been found! Loading...")
        wn = np.load(noisefile)
    else:
        print("\t\tGenerating new white noise file: %s" % noisefile)
        noise_size = [expparams['nyrs'],12,]
        if eof_flag: # Generate white noise for each mode
            if expparams['qfactor_sep'] is False:
                nmodes_plus1 = nmode + 1 # Directly include white noise timeseries
            else:
                nmodes_plus1 = nmode # Just use separate timeseries for each correction factor loaded through the helper
            print("\t\tDetected EOF Forcing. Generating %i white noise timeseries" % (nmodes_plus1))
            noise_size   = noise_size + [nmodes_plus1]
        
        wn = np.random.normal(0,1,noise_size) # [Yr x Mon x Mode]
        np.save(noisefile,wn)
    
    # Check if separate white noise timeseries should be loaded using the helper function
    if expparams['qfactor_sep']:
        wn_corr = qfactor_noisemaker(expparams,expdir,expname,runid,share_noise=expparams['share_noise'])
    
    #%% Do Conversions for Model Run ------------------------------------------
    
    if nr == 0: # Only perform this once
        
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
        if expparams['kprev'] is None and expparams['entrain'] == True: # Compute Kprev if it is not supplied
            
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
        
        
    # Use different white noise for each runid
    # wn_tile = wn.reshape()
    stfrc = time.time()
    if eof_flag:
        # if usevar: # Take the squareroot for white noise combining
        #     alpha = alpha#np.sqrt(np.abs(alpha)) * np.sign(alpha)
        
        if expparams['qfactor_sep']:
            nmode_final = alpha.shape[0]
            if wn.shape[2] != nmode_final:
                print("Dropping last mode, using separate correction timeseries...")
                wn = wn[:,:,:nmode_final]
            
            # Transpose and make the eof forcing
            forcing_eof = (wn.transpose(2,0,1)[:,:,:,None,None] * alpha[:,None,:,:,:]) # [mode x yr x mon x lat x lon]
            
            # Prepare the correction
            qfactors = [qfz for qfz in list(inputs_convert.keys()) if "correction_factor" in qfz]
            
            qftotal  = []
            for qfz in range(len(qfactors)): # Make timseries for each white noise correction
                qfname      = qfactors[qfz]
                qfactor     = inputs_convert[qfname] # [Mon x Lat x Lon]
                if "Qek" in qfname:
                    qfname = "%s_%s" % (qfname,expparams['varname'])
                wn_qf       = wn_corr[qfname] # [Year x Mon]
                
                # if usevar:
                #     qfactor = qfactor#np.sqrt(np.abs(qfactor)) * np.sign(qfactor)
                
                qf_combine  = wn_qf[:,:,None,None] * qfactor[None,:,:,:] # [Year x Mon x Lat x Lon]
                
                qftotal.append(qf_combine.copy())
            qftotal = np.array(qftotal) # [Mode x Year x Mon x Lat x Lon]
            
            print(forcing_eof.shape)
            print(qftotal.shape)
            forcing_in = np.concatenate([forcing_eof,qftotal],axis=0)
            
        else:
            forcing_in = (wn.transpose(2,0,1)[:,:,:,None,None] * alpha[:,None,:,:,:]) # [mode x yr x mon x lat x lon]
        forcing_in = np.nansum(forcing_in,0) # Sum over modes
        
    else:
        forcing_in  = wn[:,:,None,None] * alpha[None,:,:,:] # [Year x Mon]
    nyr,_,nlat,nlon = forcing_in.shape
    forcing_in      = forcing_in.reshape(nyr*12,nlat,nlon)
    smconfig['forcing'] = forcing_in.transpose(2,1,0) # Forcing in psu/mon [Lon x Lat x Mon]
    print("\tPrepared forcing in %.2fs" % (time.time()-stfrc))
    
    # New Section: Check for SST-Evaporation Feedback ------------------------
    smconfig['add_F'] = None
    if 'lbd_e' in expparams.keys() and expparams['varname'] == "SSS":
        if expparams['lbd_e'] is not None: 
            print("Adding SST-Evaporation Forcing on SSS!")
            # Load lbd_e
            lbd_e = xr.open_dataset(input_path + "forcing/" + expparams['lbd_e']).lbd_e.load() # [mon x lat x lon]
            lbd_e = proc.sel_region_xr(lbd_e,bbox=expparams['bbox_sim'])
            
            # Convert [sec --> mon]
            lbd_emon = lbd_e * dt
            lbd_emon = lbd_emon.transpose('lon','lat','mon').values
            
            # Load temperature timeseries
            assert expparams['Tforce'] is not None,"Experiment for SST timeseries [Tforce] must be specified"
            sst_nc = "%s%s/Output/SST_runid%s.nc" % (output_path,expparams['Tforce'],runid)
            sst_in = xr.open_dataset(sst_nc).SST.load()
            
            sst_in = sst_in.drop_duplicates('lon')
            
            sst_in = sst_in.transpose('lon','lat','time').values
            
            # Tile and combine
            lbd_emon_tile     = np.tile(lbd_emon,nyr) #
            lbdeT             = lbd_emon_tile * sst_in
            smconfig['add_F'] = lbdeT
    
    if debug: #Just run at a point
        ivnames = list(smconfig.keys())
        #[print(smconfig[iv].shape) for iv in ivnames]
        
        
        for iv in ivnames:
            vtype = type(smconfig[iv])
            if vtype == np.ndarray:
                smconfig[iv] = smconfig[iv][klon,klat,:].squeeze()[None,None,:]
                
        #[print(smconfig[iv].shape) for iv in ivnames]
    #  ------------------------------------------------------------------------
    
    #continue
    
    #%% Integrate the model
    if expparams['entrain'] is True:
        
        outdict = scm.integrate_entrain(smconfig['h'],smconfig['kprev'],smconfig['lbd_a'],smconfig['forcing'],
                                        Tdexp=smconfig['lbd_d'],beta=smconfig['beta'],add_F=smconfig['add_F'],
                                        return_dict=True,old_index=True,Td_corr=smconfig['Td_corr'])
        
    else:
        
        outdict = scm.integrate_noentrain(smconfig['lbd_a'],smconfig['forcing'],T0=0,multFAC=True,debug=True,old_index=True,return_dict=True)
    
    
    #% Save the output
    if debug:
        ts = outdict['T'].squeeze()
        plt.plot(ts),plt.show()
    else:
        var_out  = outdict['T']
        timedim  = xr.cftime_range(start="0001",periods=var_out.shape[-1],freq="MS",calendar="noleap")
        cdict    = {
            "time" : timedim,
            "lat" : latr,
            "lon" : lonr,
            }
        
        da       = xr.DataArray(var_out.transpose(2,1,0),coords=cdict,dims=cdict,name=expparams['varname'])
        edict    = {expparams['varname']:{"zlib":True}}
        savename = "%sOutput/%s_runid%s.nc" % (expdir,expparams['varname'],runid)
        da.to_netcdf(savename,encoding=edict)
        



    
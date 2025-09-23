#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

T

Copied run_SSS_basinwide on Wed Oct  2 14:02:31 2024

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

lonrun = -39.75
latrun = 59.75


expname     = "SST_ORAS5_avg_GMSSTmon"

expparams   = {
    'varname'           : "SST",
    'bbox_sim'          : [-40,-15,52,62],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : 'SST_Obs_Pilot_00_Tdcorr0_qnet', # If not None, load a runid from another directory
    'Fprime'            : "ERA5_Fprime_QNETgmsstMON_std_pilot.nc",
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
    "eof_forcing"       : False, # CHECK THIS
    "Td_corr"           : True, # Set to True if lbd_d is provided as a correlation, rather than 1/months
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

inputs,inputs_ds,inputs_type,params_vv = scm.load_params(expparams,input_path,pointmode=[lonrun,latrun])


#%% Convert to Point of choice



#%% Detect and Process Missing Inputs

_,nlat,nlon=inputs['h'].shape

# Get number of modes
if eof_flag:
    if expparams['varname'] == "SST":
        nmode = inputs['Fprime'].shape[0]
    elif expparams['varname'] == "SSS":
        nmode = inputs['LHFLX'].shape[0]


#%% Initialize An Experiment folder for output

# expdir = output_path + expname + "/"
# proc.makedir(expdir + "Input")
# proc.makedir(expdir + "Output")
# proc.makedir(expdir + "Metrics")
# proc.makedir(expdir + "Figures")

# # Save the parameter file
# savename = "%sexpparams.npz" % (expdir+"Input/")
# chk = proc.checkfile(savename)
# if chk is False:
#     print("Saving Parameter Dictionary...")
#     np.savez(savename,**expparams,allow_pickle=True)

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

dsoutput_all = []
for nr in range(nruns):
    
    #% Prepare White Noise timeseries ----------------------------------------
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
        print("\t\tWhite file has been found! Loading...")
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
    
    #% Do Conversions for Model Run ------------------------------------------
    
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
        
        
    # Use different white noise for each runid
    # wn_tile = wn.reshape()
    stfrc = time.time()
    if eof_flag:
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

    #  ------------------------------------------------------------------------
    #  ------------------------------------------------------------------------
    # Make some random modifications here
    smconfig['lbd_a'][:,:,4]= .1
    smconfig['lbd_a'][:,:,3]= .1
    smconfig['lbd_a'][:,:,2]= -0.19559057
    
    #  ------------------------------------------------------------------------
    #  ------------------------------------------------------------------------
    #  ------------------------------------------------------------------------
    

    #% Integrate the model
    if expparams['entrain'] is True:
        outdict = scm.integrate_entrain(smconfig['h'],smconfig['kprev'],smconfig['lbd_a'],smconfig['forcing'],
                                        Tdexp=smconfig['lbd_d'],beta=smconfig['beta'],add_F=smconfig['add_F'],
                                        return_dict=True,old_index=True,Td_corr=smconfig['Td_corr'])
    else:
        outdict = scm.integrate_noentrain(smconfig['lbd_a'],smconfig['forcing'],T0=0,multFAC=True,debug=True,old_index=True,return_dict=True)
    
    
    
    #%
    #timedim = xr.cftime_range(start='0000',periods=len(T),freq="MS",calendar="noleap")
    T       = outdict['T'].squeeze()
    timedim = xr.cftime_range(start='0000',periods=len(T),freq="MS",calendar="noleap")
    coords  = dict(time=timedim)
    sst             = xr.DataArray(T,coords=coords,dims=coords,name='sst')
    termnames       = ['damping_term', 'forcing_term', 'entrain_term', 'Td']
    term_ds         = [xr.DataArray(outdict[tname].squeeze(),coords=coords,dims=coords,name=tname) for tname in termnames]
    termnames_mon   = ['beta', 'FAC', 'lbd']
    coordsmon  = dict(mon=np.arange(1,13,1))
    param_ds        = [xr.DataArray(outdict[tname].squeeze(),coords=coordsmon,dims=coordsmon,name=tname) for tname in termnames_mon]
    ds_merge = [sst,]+term_ds+param_ds
    dsmerge  = xr.merge(ds_merge)
    
    dsoutput_all.append(dsmerge)

dsout = xr.concat(dsoutput_all,dim='run')

#%% Analysis

nr = 0


#for nr in range(10):
    
dsrun0 = dsout.isel(run=nr)


dsmonvar = dsrun0.groupby('time.month').mean('time')

mons3 = proc.get_monstr()

plotterms = ['sst','damping_term',"forcing_term","entrain_term"]



for ii in range(4):
    tname   = plotterms[ii]
    plotvar = dsmonvar[tname]
    fig,ax = viz.init_monplot(1,1)
    ax.bar(mons3,plotvar,label=tname)
    ax.legend()
    ax.set_xlim([-1,12])

    if ii <2:
        ax.set_ylim([0,0.75])
        #ax.set_ylim([0,10])
    

#%% PLot the timeseries


fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12.5,2))

lab = "stdev=%.4f" % dsout.isel(run=0)['sst'].std().item()
dsout.isel(run=0)['sst'].plot(ax=ax,lw=.5,label=lab)
ax.axhline([0],ls='dashed',c="k",lw=0.75)


ax.legend()


    
#%% PLot Lbd

yticks = np.arange(-.2,1.1,0.1)
fig,ax = viz.init_monplot(1,1)

ax.plot(mons3,dsout['lbd'].isel(run=0),label=r"Total Damping ($\frac{w_e}{h} + \lambda^a$)",c='gray')
ax.plot(mons3,smconfig['lbd_a'].squeeze(),color="r",label="$\lambda^a$",ls='dashed')
ax.plot(mons3,dsout['beta'].isel(run=0),label=r"$\frac{w_e}{h}$",c='blue',ls='dotted')

total_lbd = dsout['beta'].isel(run=0).data + smconfig['lbd_a'].squeeze()
#ax.plot(mons3,total_lbd,label=r"$\frac{w_e}{h} + \lambda^a$",c='yellow',ls='dashed')

ax.set_ylim([yticks[0],yticks[-1]])
ax.set_yticks(yticks)
ax.axhline([0],ls='dashed',lw=0.75,c='k')
ax.legend()

#print(outdict.keys())
    # #% Save the output
    # if debug:
    #     ts = outdict['T'].squeeze()
    #     plt.plot(ts),plt.show()
    # else:
    #     var_out  = outdict['T']
    #     timedim  = xr.cftime_range(start="0001",periods=var_out.shape[-1],freq="MS",calendar="noleap")
    #     cdict    = {
    #         "time" : timedim,
    #         "lat" : latr,
    #         "lon" : lonr,
    #         }
        
    #     da       = xr.DataArray(var_out.transpose(2,1,0),coords=cdict,dims=cdict,name=expparams['varname'])
    #     edict    = {expparams['varname']:{"zlib":True}}
    #     savename = "%sOutput/%s_runid%s.nc" % (expdir,expparams['varname'],runid)
    #     da.to_netcdf(savename,encoding=edict)

#%% Check Output


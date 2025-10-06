#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Local Edition of analyze_amoc_index

Created on Tue Sep 23 15:40:51 2025

@author: gliu

"""


#from amv import proc, viz
import scipy as sp
#import yo_box as yo
#import cvd_utils as cvd
#import amv.loaders as dl
#import scm
import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import os
import matplotlib.patheffects as PathEffects
import glob
import tqdm




# %%

amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/"  # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc, viz
import yo_box as yo
import cvd_utils as cvd
import amv.loaders as dl
import scm


# %% Load data

# Load the netCDF
amocpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/proc/"
amocnc = "MOC_merge_0to90.nc"
st = time.time()
dsmoc = xr.open_dataset(amocpath+amocnc).load()
proc.printtime(st, print_str="loaded in")


# Load SST
st = time.time()
sstnc = "CESM2_FCM_PiControl_TS_0000_2000_raw.nc"
sstpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/NAtl/Old/"
dsst = xr.open_dataset(sstpath+sstnc).load()
proc.printtime(st, print_str="loaded in")

# Declare Figure Path
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20251006/"
proc.makedir(figpath)

# Load N_HEAT
nheat_path  = "/Users/gliu/Globus_File_Transfer/CESM2_PiControl/N_HEAT/"
nclist      = glob.glob(nheat_path + "*.nc")
nclist.sort()

dsnheat  = xr.open_mfdataset(nclist,concat_dim='time',combine='nested')
dsnheat  = dsnheat.sel(lat_aux_grid=slice(0,90)).isel(transport_reg=1,transport_comp=1)
keepvars = ["N_HEAT","time",
            #"transport_regions","transport_components",
            "transport_reg",
            "moc_z","lat_aux_grid",
            "TLAT","z_t"]
dsnheat  = proc.ds_dropvars(dsnheat,keepvars).load()


# %% Some Functions

def preprocess_ds(ds):

    dsa = proc.xrdeseason(ds)

    dsa = proc.xrdetrend(dsa)
    dsa = proc.fix_febstart(dsa)

    return dsa


def pointwise_coherence(ts1, ts2, opt=1, nsmooth=100, pct=0.10, return_ds=False):

    if np.any(np.isnan(ts1)) or np.any(np.isnan(ts2)):
        return np.nan

    else:

        CP, QP, freq, dof, r1_x, r1_y, PX, PY = yo.yo_cospec(ts1,  # ugeo
                                                             ts2,  # sst
                                                             opt, nsmooth, pct,
                                                             debug=False, verbose=False, return_auto=True)

        # Compute Coherence
        coherence_sq = CP**2 / (PX * PY)

        # Make DataArray
        if return_ds:
            coords = dict(freq=freq)
            othervars = dict(dof=dof, r1_x=r1_x, r1_y=r1_y)
            spectra_x = xr.DataArray(
                PX, coords=coords, dims=coords, name="spectra_x")
            spectra_y = xr.DataArray(
                PY, coords=coords, dims=coords, name="spectra_y")
            quad_spec = xr.DataArray(
                QP, coords=coords, dims=coords, name="quad_spectrum")
            co_spec = xr.DataArray(
                CP, coords=coords, dims=coords, name="co_spectrum")
            coherence_sq = xr.DataArray(
                coherence_sq, coords=coords, dims=coords, name="coherence_sq")
            ds_out = xr.merge([othervars, spectra_x, spectra_y,
                              quad_spec, co_spec, coherence_sq])
            return ds_out
        return coherence_sq

# %% Preprocess

# Transport Components
comps = [b'Total', b'Eulerian-Mean Advection',
         b'Eddy-Induced Advection (bolus) + Diffusion',
         b'Eddy-Induced (bolus) Advection', b'Submeso Advection']

icomp   = 0
moca    = preprocess_ds(dsmoc.isel(moc_comp=icomp).MOC)
ssta    = preprocess_ds(dsst.TS)


nheata  = preprocess_ds(dsnheat.N_HEAT)

# %% Calculate AMOC Index by latitude (max depth)

# Basically taking the absolute maximum, keeping the sign... (prob smarter way to do this)
mocmax = moca.max('moc_z')
mocmin = moca.min('moc_z')
amocidx = xr.where(np.abs(mocmax) > np.abs(mocmin),mocmax,mocmin)


#amocidx     = moca.max('moc_z')  # [Time x Latitude]

maxdepth    = moca.idxmax('moc_z')/100


bbox_spgne  = [-40, -15, 52, 62]

# Calculate Area-averaged SPGNE Index
sstreg      = proc.sel_region_xr(ssta, bbox_spgne)
spgneid     = proc.area_avg_cosweight(sstreg)


# %% Calculate Correlation

#calc_r2 = True
calc_r2 = True
filts   = [None, 5*12, 10*12, 20*12]
nfilt   = len(filts)
border  = 6

ntime, nlat = amocidx.shape

corr_by_lat       = np.zeros((nfilt, nlat))
corr_by_lat_nheat = np.zeros((nfilt, nlat))

dofeff = corr_by_lat.copy()

for ff in range(nfilt):
    filtin = filts[ff]

    if filtin is None:
        spgin = spgneid.data
    else:
        spgin = proc.lp_butter(spgneid.data, filtin, border)
    
    for a in range(nlat):
        
        corr_by_lat[ff, a]      = np.corrcoef(spgin, amocidx[:, a])[0, 1]
        if calc_r2:
            corr_by_lat[ff, a] = (corr_by_lat[ff, a])** 2 

        
        dofeff[ff, a]           = proc.calc_dof(spgin, ts1=amocidx[:, a])
        
        corr_by_lat_nheat[ff,a] = np.corrcoef(spgin, nheata.data[:, a])[0, 1]
        if calc_r2:
            corr_by_lat_nheat[ff,a] = (corr_by_lat_nheat[ff,a]) ** 2

rhocrit    = proc.ttest_rho(0.05, 1, len(spgin))
rhocriteff = proc.ttest_rho(0.05, 1, dofeff)

# %% Plot Correlation vs Latitude

expcols = ["k", "magenta", "blue", "green"]


xtks    = np.arange(0, 95, 5)

lat     = moca.lat_aux_grid
z       = moca.moc_z/100

fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(12.5, 4))

for ff in range(nfilt):
    if ff == 0:
        lab = "No Filtering"
    else:
        lab = "LP Filter Cutoff %i-months" % (filts[ff])
    ax.plot(lat, corr_by_lat[ff, :], label=lab, c=expcols[ff])

    ax.plot(lat, rhocriteff[ff, :], ls='dashed', lw=.75, c=expcols[ff])
    
    ax.plot(lat, corr_by_lat_nheat[ff, :], label="", c=expcols[ff],ls='dotted')
    

ax.set_xticks(xtks)
if calc_r2:
    ax.set_ylabel("$R^2$")
else:
    ax.set_ylabel("Correlation")
ax.set_xlabel("Latitude")

ax.axvline([52], c='magenta', ls='dotted')
ax.axvline([62], c='magenta', ls='dotted')
ax.axhline([0], c='k', ls='dotted', lw=0.5)
ax.set_title(
    "Correlation between SPGNE SST and AMOC Maximum Transport @ each latitude")


ax.legend()


# %% Look at the patterns of MOC transport associated with each case

pats = []
sigs = []
for ff in tqdm.tqdm(range(nfilt)):

    filtin = filts[ff]

    if filtin is None:
        spgin = spgneid.data
    else:
        spgin = proc.lp_butter(spgneid.data, filtin, border)

    mocarr = moca.transpose('moc_z', 'lat_aux_grid', 'time').data

    st = time.time()
    output = proc.regress_ttest(mocarr, spgin)
    proc.printtime(st, print_str="Completed regression in")

    pats.append(output["regression_coeff"])
    sigs.append(output["sigmask"])


# %% Plot the patterns


levels = np.arange(-30, 32, 2)

for ff in range(nfilt):

    fig, axs = plt.subplots(2, 1, constrained_layout=True,
                            figsize=(12.5, 6), sharex=True)

    ax = axs[0]
    plotvar = pats[ff]
    mask = sigs[ff]
    pcm = ax.pcolormesh(lat, z, plotvar, cmap='cmo.balance',
                        vmin=-2.5, vmax=2.5)
    sigp = viz.plot_mask(lat, z, mask.T, ax=ax, color='gray')

    ax.invert_yaxis()

    cb = fig.colorbar(pcm, ax=ax, fraction=0.01, pad=0.01)
    cb.set_label("Sv MOC Transport per degC SPGNE Index")
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Depth (meters)")
    ax.axvline([52], c='magenta', ls='dotted')
    ax.axvline([62], c='magenta', ls='dotted')
    ax.grid(True, ls='dashed', lw=0.25, c='k')

    # Plot the line plots from above
    ax = axs[1]
    if ff == 0:
        lab = "No Filtering"
    else:
        lab = "LP Filter Cutoff %i-months" % (filts[ff])
    ax.plot(lat, corr_by_lat[ff, :], label=lab, c=expcols[ff])

    ax.plot(lat, rhocriteff[ff, :], ls='dashed', lw=.75, c=expcols[ff])

    ax.set_xticks(xtks)
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Latitude")

    ax.axvline([52], c='magenta', ls='dotted')
    ax.axvline([62], c='magenta', ls='dotted')
    ax.axhline([0], c='k', ls='dotted', lw=0.5)
    ax.set_title(
        "Correlation between SPGNE SST and AMOC Maximum Transport @ each latitude")
    
    axs[0].axvline([36.5],c='k')
    
    figname = "%sAMOC_SPGNE_Regression_fit%s.png" % (figpath,str(filts[ff]))
    plt.savefig(figname,dpi=150,bbox_inches='tight')

# =========================================================
# %% Try to calculate coherence looping across all latitudes
# =========================================================

# Indicate Smoothing and filter preferences
nsmooth = 250

# Select Filtering
filt    = None#10*12  # 20*12#None
filtstr = str(filt)




chsq = []
for a in tqdm.tqdm(range(nlat)):
    ts1 = spgneid.data
    if filt is not None:
        ts1 = proc.lp_butter(ts1, filt, 6)
    ts2 = amocidx.isel(lat_aux_grid=a)  # .data
    ts2 = proc.xrdeseason(ts2).data

    cs = pointwise_coherence(
        ts1, ts2, opt=1, nsmooth=nsmooth, pct=0.10, return_ds=True)
    chsq.append(cs)

# Merge and assign lat dimension
chsq2 = xr.concat(chsq, dim='lat')
chsq2 = chsq2.assign_coords(dict(lat=amocidx.lat_aux_grid.data))


# plotvar = chsq2.sel(lat=45,method='nearest') # Debug Plot with a slice

# %% Plot Latitude vs. Coherence

cints   = np.arange(0,0.48,0.03)
fig,ax  = plt.subplots(1, 1, figsize=(10, 4.5),constrained_layout=True)

plotvar = chsq2.coherence_sq


pcm     = ax.contourf(plotvar.freq,plotvar.lat,plotvar,cmap='cmo.thermal',levels=cints)
#cl      = ax.contour(plotvar.freq,plotvar.lat,plotvar,colors='w',levels=cints,linewidths=0.55)
#clbl = ax.clabel(pcm,levels=cints[::2])
#viz.add_fontborder(clbl)


cb      = viz.hcbar(pcm,ax=ax,pad=0.02,fraction=0.055)
if filt is None:
    cb.set_label("Magnitude Squared Coherence (SPGNE-SST vs. AMOC Transport)",fontsize=14)
else:
    cb.set_label("Magnitude Squared Coherence (%s-month filtered SPGNE-SST vs. AMOC Transport)" % filtstr,fontsize=14)
ax.set_xscale('log')

# ----- Axes Things --------

xper = np.array([40, 10, 5, 1, 0.5])
xper_ticks = 1 / (xper*12)

ax.set_xlim([xper_ticks[0], 0.5])
ax.axvline([1/(6)], label="", ls='dotted', c='gray')
ax.axvline([1/(12)], label="", ls='dotted', c='gray')
ax.axvline([1/(5*12)], label="", ls='dotted', c='gray')
ax.axvline([1/(10*12)], label="", ls='dotted', c='gray')
ax.axvline([1/(40*12)], label="", ls='dotted', c='gray')

ax.set_xlabel("Frequency (1/Month)",fontsize=14)


ax2 = ax.twiny()
ax2.set_xlim([xper_ticks[0], 0.5])
ax2.set_xscale('log')
ax2.set_xticks(xper_ticks, labels=xper)
ax2.set_xlabel("Period (Years)", fontsize=14)

# ----- Axes Things --------

ax.set_ylabel("Latitude ($\degree$N)",fontsize=14)
ax.axhline([62],c="magenta",ls='dashed',lw=0.75)
ax.axhline([52],c="magenta",ls='dashed',lw=0.75)


figname = "%sCoherenceSq_SPGNE_AMOC_Meridional_filt%s.png" % (
    figpath, filtstr)

plt.savefig(figname, dpi=150, bbox_inches='tight')



# %% Plot coherence at a given latitude along with the power spectra
latf =  36.5

# % Other PLotting Things
dtmon_fix = 60*60*24*30
xper = np.array([40, 10, 5, 1, 0.5])
xper_ticks = 1 / (xper*12)


fig, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True,figsize=(10,8))

for ii in range(3):

    ax = axs[ii]

    if ii == 0:
        plotvar = chsq2.sel(lat=latf, method='nearest').coherence_sq
        title = "Coherence Squared"
    elif ii == 1:
        plotvar = chsq2.sel(lat=latf, method='nearest').spectra_x
        title = "SPGNE-Average SST"
    elif ii == 2:
        plotvar = chsq2.sel(lat=latf, method='nearest').spectra_y
        title = "AMOC Transport at %.2f$\degree$N" % latf
    ax.loglog(plotvar.freq, plotvar)
    ax.set_title(title)

    # -----

    ax.set_xlim([xper_ticks[0], 0.5])
    ax.axvline([1/(6)], label="", ls='dotted', c='gray')
    ax.axvline([1/(12)], label="", ls='dotted', c='gray')
    ax.axvline([1/(5*12)], label="", ls='dotted', c='gray')
    ax.axvline([1/(10*12)], label="", ls='dotted', c='gray')
    ax.axvline([1/(40*12)], label="", ls='dotted', c='gray')

    # ax.set_xlabel("Frequency (1/Month)",fontsize=14)
    # ax.set_ylabel("Power [$\degree C ^2 cycle \, per \, mon$]")

    ax2 = ax.twiny()
    ax2.set_xlim([xper_ticks[0], 0.5])
    ax2.set_xscale('log')
    ax2.set_xticks(xper_ticks, labels=xper)
    #ax2.set_xlabel("Period (Years)", fontsize=14)

    # ----
# chsq2.coherence_sq.plot()

# %% Try a new function to compute coherence
# Values appear to be too low (maybe due to the large DOF values)

def calc_sig_coherence(alpha, dof, coh_sq=True, approx=False):
    """
    Source: https://dsp.stackexchange.com/questions/16558/statistical-significance-of-coherence-values

    Method (1): Coherence (coh_sq=False,approx=False)

        L1(a,q)        = 1 - alpha^(1/q)

    Method (2): Magnitude-Squared Coherence (coh_sq=True,approx=False)

        L2(a,q)        = F_(2,2q)(alpha) / (F_(2,2q)(alpha) + q)

    Method (3): Wunsch Approximation? (approx=True)

        L3(alpha=0.05) = 6/DOF (as q --> inf)

    """
    p = 1 - alpha
    n = dof/2  # Why?
    if approx:
        return 6/dof
    else:
        if coh_sq:
            fval = sp.stats.f.ppf(p, 2, dof-2)
            L2 = fval / (n - 1 + fval)
            return L2
        else:
            L1 = 1 - alpha**(1 / (n-1))
            return L1
    return None

# ======================================================
# %% Select a point and examine coherence + significance
# ======================================================


# Select Filtering
filt    = None #20*12#None#20*12  # None
filtstr = str(filt)

# Indicate Latitude
latf = 41.56
ts1 = spgneid.data

use_nheat=True


# Get Timeseries
if filt is not None:
    ts1 = proc.lp_butter(ts1, filt, 6)
    
if use_nheat:
    ts2 = nheata.sel(lat_aux_grid=latf, method='nearest').data
else:
    ts2 = proc.xrdeseason(amocidx.sel(lat_aux_grid=latf, method='nearest')).data

# Compute Coherence
cout_raw = pointwise_coherence(ts1, ts2, nsmooth=nsmooth, return_ds=True)
cohsq_raw = cout_raw.coherence_sq

# Calculate Effective DOF
dofeff = proc.calc_dof(ts1, ts1=ts2,)

# Try New Function to compute significance
L1 = calc_sig_coherence(0.05, dofeff, coh_sq=False)
L2 = calc_sig_coherence(0.05, dofeff, coh_sq=True)
L3 = calc_sig_coherence(0.05, dofeff, approx=True)

# Iteratively compute coherence
# note that the white noise generated still does not match the original timeseries
# need to troubleshoot this and look at how others estimate the noise term


def calcsig_both(ts1, ts2):
    return np.std(ts1), np.std(ts2)


mciter = 1000
def csig(x, y): return calcsig_both(x, y)


def ccoh(x, y): return pointwise_coherence(
    x, y, opt=1, nsmooth=nsmooth, pct=0.10, return_ds=True).coherence_sq.data


mcout = proc.montecarlo_ar1(ts1, ts2, mciter, [csig, ccoh])
mcout = [np.array(mc) for mc in mcout]

# %% Plot coherence at a point with significance


# % Other Plotting Things
dtmon_fix   = 60*60*24*30
xper        = np.array([40, 10, 5, 1, 0.5])
xper_ticks  = 1 / (xper*12)


# Initialize Plot
fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(10, 4.5))
plotvar = cohsq_raw  # chsq2.sel(lat=latf,method='nearest').coherence_sq
title = "Coherence Squared"
ax.loglog(plotvar.freq, plotvar)

# ax.plot(plotvar.freq,wncoh.coherence_sq,label=100)

# Plot Significance from montecarlo
coh = mcout[1]
lowerbnd, upperbnd = np.quantile(coh, [0.025, 0.975], axis=0)
ax.fill_between(plotvar.freq, lowerbnd, upperbnd,
                alpha=0.45, label="95% Significance")

ax.axhline([L1], color="k", label="L1")
ax.axhline([L2], color="k", label="L2", ls='dashed')
ax.axhline([L3], color="midnightblue", label="L3", ls='dotted')


# -----
# Axes Things
ax.set_xlim([xper_ticks[0], 0.5])
ax.axvline([1/(6)], label="", ls='dotted', c='gray')
ax.axvline([1/(12)], label="", ls='dotted', c='gray')
ax.axvline([1/(5*12)], label="", ls='dotted', c='gray')
ax.axvline([1/(10*12)], label="", ls='dotted', c='gray')
ax.axvline([1/(40*12)], label="", ls='dotted', c='gray')

# ax.set_xlabel("Frequency (1/Month)",fontsize=14)
# ax.set_ylabel("Power [$\degree C ^2 cycle \, per \, mon$]")

ax2 = ax.twiny()
ax2.set_xlim([xper_ticks[0], 0.5])
ax2.set_xscale('log')
ax2.set_xticks(xper_ticks, labels=xper)
ax2.set_xlabel("Period (Years)", fontsize=14)
# ---


ax.set_ylim([1e-9, 1e1])


if filt is not None:
    ax.set_title("Coherence Squared (%i-month Filtered SPGNE SST vs. AMOC @ %.2f$\degree$N)" %
                 (filt, latf), fontsize=18)
else:
    
    if use_nheat:
        ax.set_title(
            "Coherence Squared (SPGNE SST vs. N_HEAT @ %.2f$\degree$N)" % (latf), fontsize=18)
    else:
        ax.set_title(
            "Coherence Squared (SPGNE SST vs. AMOC @ %.2f$\degree$N)" % (latf), fontsize=18)

figname = "%sCoherenceSq_SPGNE_AMOC%02i_filt%s_mciter%i.png" % (
    figpath, latf, filtstr, mciter)

plt.savefig(figname, dpi=150, bbox_inches='tight')

# %% Visualize some basic features


fig,axs = plt.subplots(2,1)

ax1,ax2 = axs

proc.xrdeseason(amocidx.sel(lat_aux_grid=26,method='nearest')).plot(ax=ax1)

#%% Do simple covariance calculations

filt=None

latf                = 41.56
tsraw               = spgneid.data
tsraw_lp            = proc.lp_butter(tsraw,120,6)
amocin              = amocidx.sel(lat_aux_grid=latf,method='nearest').data

var_spgne           = np.var(tsraw)
cov_spgne_amoc      = np.cov(tsraw,amocin)[0,1]
cov_spgne_amoc_lp   = np.cov(tsraw_lp,amocin)[0,1]
pct                 = cov_spgne_amoc/var_spgne * 100
pct_lp              = cov_spgne_amoc_lp/var_spgne * 100


cov_bylat = []
pct_bylat = []
for a in tqdm.tqdm(range(nlat)):
    
    if filt is None:
        tsin = tsraw
    else:
        tsin = proc.lp_butter(tsraw,filt,6)
    var_spgne = np.var(tsin)
    
    
    amocin              = amocidx.isel(lat_aux_grid=a).data
    cov_spgne_amoc      = np.cov(tsin,amocin)[0,1]
    pct                 = cov_spgne_amoc/var_spgne * 100
    cov_bylat.append(cov_spgne_amoc)
    pct_bylat.append(np.abs(pct))
    
cov_bylat = np.array(cov_bylat)
pct_bylat = np.array(pct_bylat)


#%%

fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(12.5, 4))

ax.plot(lat,cov_bylat)
#ax.plot(lat,pct_bylat)

ax.set_ylabel("Cov(SST,AMOC) / Var(SST) * 100")
ax.set_xlabel("Latitude ($\degree$N)",fontsize=14)
ax.axvline([62],c="magenta",ls='dashed',lw=0.75)
ax.axvline([52],c="magenta",ls='dashed',lw=0.75)

ax.set_xticks(xtks)

#%%
    


np.cov(ts1,ts2)


#%% Try to compute R2 for different lags

both_filt = True
filtin   = 10*12
border   = 6
leadlags = np.arange(-60,61,1)


# abyfilt = []
# nbyfilt = []


# for ff in range(nfilt):
#     filtin = filts[ff]
    
if filtin is None:
    spgin = spgneid.data
else:
    spgin = proc.lp_butter(spgneid.data, filtin, border)

nleadlags   = len(leadlags)
amoc_corrs  = np.zeros((nlat,nleadlags))
nheat_corrs = amoc_corrs.copy()


for a in tqdm.tqdm(range(nlat)):
    
    amocin    = amocidx[:, a].data
    nheatin   = nheata.data[:, a]
    
    if both_filt and filtin is not None:
        amocin  = proc.lp_butter(amocin, filtin, border)
        nheatin = proc.lp_butter(nheatin, filtin, border)
        
        
        
    amoclags  = calc_lag_corr_1d(amocin,spgin,leadlags)
    nheatlags = calc_lag_corr_1d(nheatin,spgin,leadlags)
        
    # amoclags  = calc_lag_corr_1d(spgin,amocin,leadlags)
    # nheatlags = calc_lag_corr_1d(spgin,nheatin,leadlags)
    
    amoc_corrs[a,:] = amoclags.copy()
    nheat_corrs[a,:] = nheatlags.copy()
    

# abyfilt.append(amoc_corrs)
# nbyfilt.append(nheat_corrs)

    
# abyfilt = np.array(abyfilt)
# nbyfilt = np.array(nbyfilt)
        
# #% Save the Output


# coords = dict(filt=filts,lat=lat,lag=leadlags)

# daout_amoc  = xr.DataArray(abyfilt,coords=coords,dims=coords,name='amoc')
# daout_nheat = xr.DataArray(nbyfilt,coords=coords,dims=coords,name='n_heat')



    
    
#%% Try to Plot the Output

calc_r2    = True
plot_lag0  = False
plot_lagvalue = False
expcols    = ["k", "magenta", "blue", "green"]


xtks    = np.arange(0, 95, 5)

lat     = moca.lat_aux_grid
z       = moca.moc_z/100

if filtin is not None:
    lab     = "%i-month LP Filter" % filtin
else:
    lab = "Raw"

fig, ax= plt.subplots(1, 1, constrained_layout=True, figsize=(12.5, 4))

if plot_lag0:
    ax.plot(lat, amoc_corrs[:,60]**2, label="AMOC Transport " + lab, c='midnightblue')
    ax.plot(lat, nheat_corrs[:,60]**2, label="N_HEAT " + lab, c='salmon',ls='dotted')
else:
    
    if plot_lagvalue:
        ax.scatter(lat, leadlags[(amoc_corrs[:,:]**2).argmax(1)], label="AMOC Transport " + lab, c='midnightblue')
        ax.scatter(lat, leadlags[(nheat_corrs[:,:]**2).argmax(1)], label="N_HEAT " + lab, c='salmon',ls='dotted')
        
    else:
        ax.plot(lat, (amoc_corrs[:,:]**2).max(1), label="AMOC Transport " + lab, c='midnightblue')
        ax.plot(lat, (nheat_corrs[:,:]**2).max(1), label="N_HEAT " + lab, c='salmon',ls='dotted')
        
    # Also Plot index of maximum Lag
    # ax2 = ax.twinx()
    # ax2.scatter(lat, leadlags[(amoc_corrs[:,:]**2).argmax(1)], label="AMOC Transport " + lab, c='midnightblue')
    # ax2.scatter(lat, leadlags[(nheat_corrs[:,:]**2).argmax(1)], label="N_HEAT " + lab, c='salmon',ls='dotted')
    
#ax.plot(lat, corr_by_lat[ff, :], label="Original", c="k",ls='dashed')
print("Max Lat for AMOC R2 is %.2f" % (lat[(amoc_corrs[:,:]**2).max(1).argmax()].item()))
print("Max Lat for N_HEAT R2 is %.2f" % (lat[(nheat_corrs[:,:]**2).max(1).argmax()].item()))



#a#x.plot(lat, rhocriteff[ff, :], ls='dashed', lw=.75, c=expcols[ff])
#ax.plot(lat, corr_by_lat_nheat[ff, :], label="", c=expcols[ff],ls='dotted')

# for ff in range(nfilt):
#     if ff == 0:
#         lab = "No Filtering"
#     else:
#         lab = "LP Filter Cutoff %i-months" % (filts[ff])
#     ax.plot(lat, corr_by_lat[ff, :], label=lab, c=expcols[ff])

#     ax.plot(lat, rhocriteff[ff, :], ls='dashed', lw=.75, c=expcols[ff])
    
#     ax.plot(lat, corr_by_lat_nheat[ff, :], label="", c=expcols[ff],ls='dotted')
    

ax.set_xticks(xtks)
if plot_lagvalue:
    ax.set_ylabel("Lag (Months)")
    
    ax.set_title(
        "Lag of Max $R^2$ between SPGNE SST and AMOC Maximum Transport @ each latitude")
    
else:
    if calc_r2:
        ax.set_ylabel("$R^2$")
    else:
        ax.set_ylabel("Correlation")
    ax.set_title(
        "Max $R^2$ between SPGNE SST and AMOC Maximum Transport @ each latitude")
        
ax.set_xlabel("Latitude")

ax.axvline([52], c='magenta', ls='dotted')
ax.axvline([62], c='magenta', ls='dotted')
ax.axhline([0], c='k', ls='dotted', lw=0.5)
ax.set_xlim([xtks[0],xtks[-1]])


ax.legend()

figname = "%sAMOC_SPGNE_NHEAT_R2_fit%s_bothfilt%i.png" % (figpath,str(filtin),both_filt)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% For each Latitude, Plot the Correlation

for ii in range(2):
    
    if ii == 0:
        plotvar = ((amoc_corrs)**2).T
        iiname  = "AMOC"
    else:
        plotvar = ((nheat_corrs)**2).T
        iiname  = "N_HEAT"

    fig,ax = plt.subplots(1,1, constrained_layout=True, figsize=(12.5, 4))
    
    
    pcm = ax.pcolormesh(lat,leadlags,plotvar,cmap='cmo.thermal',vmin=0,vmax=0.75)
    
    ax.scatter(lat, leadlags[(amoc_corrs[:,:]**2).argmax(1)], label="AMOC Transport " + lab, c='cyan')
    ax.scatter(lat, leadlags[(nheat_corrs[:,:]**2).argmax(1)], label="N_HEAT " + lab, c='yellow',ls='dotted',marker="x")
    ax.legend()
    
    
    #fig.colorbar(pcm,ax=ax)
    ax.set_xticks(xtks)
    
    ax.axvline([52], c='magenta', ls='dotted')
    ax.axvline([62], c='magenta', ls='dotted')
    ax.axhline([0], c='k', ls='dotted', lw=0.5)
    ax.set_xlim([xtks[0],xtks[-1]])
    figname = "%s%s_SPGNE_MaxLag_fit%s_bothfilt%i.png" % (figpath,iiname,str(filtin),both_filt)
    plt.savefig(figname,dpi=150,bbox_inches='tight')

    
        
#%%
ts1 = 
ts2 = 


#%%


#calc_r2 = True
calc_r2 = True
filts   = [None, 5*12, 10*12, 20*12]
nfilt   = len(filts)
border  = 6

ntime, nlat = amocidx.shape

corr_by_lat       = np.zeros((nfilt, nlat))
corr_by_lat_nheat = np.zeros((nfilt, nlat))

dofeff = corr_by_lat.copy()

for ff in range(nfilt):
    filtin = filts[ff]

    if filtin is None:
        spgin = spgneid.data
    else:
        spgin = proc.lp_butter(spgneid.data, filtin, border)
    
    for a in range(nlat):
        
        corr_by_lat[ff, a]      = np.corrcoef(spgin, amocidx[:, a])[0, 1]
        if calc_r2:
            corr_by_lat[ff, a] = (corr_by_lat[ff, a])** 2 

        
        dofeff[ff, a]           = proc.calc_dof(spgin, ts1=amocidx[:, a])
        
        corr_by_lat_nheat[ff,a] = np.corrcoef(spgin, nheata.data[:, a])[0, 1]
        if calc_r2:
            corr_by_lat_nheat[ff,a] = (corr_by_lat_nheat[ff,a]) ** 2

rhocrit    = proc.ttest_rho(0.05, 1, len(spgin))
rhocriteff = proc.ttest_rho(0.05, 1, dofeff)


#%% Edit Function



def calc_lag_corr_1d(var1,var2,lags): # CAn make 2d by mirroring calc_lag_covar_annn
    # Calculate the regression where
    # (+) lags indicate var1 lags  var2 (var 2 leads)
    # (-) lags indicate var1 leads var2 (var 1 leads)
    
    ntime = len(var1)
    betalag = []
    poslags = lags[lags >= 0]
    for l,lag in enumerate(poslags):
        varlag   = var1[lag:]
        varbase  = var2[:(ntime-lag)]
        
        # Calculate correlation
        beta = sp.stats.linregress(varbase,varlag)[2] # #np.polyfit(varbase,varlag,1)[0]   
        betalag.append(beta.item())
    
    
    neglags      = lags[lags < 0]
    neglags_sort = np.sort(np.abs(neglags)) # Sort from least to greatest #.sort
    betalead     = []
    
    
    for l,lag in enumerate(neglags_sort):
        varlag   = var2[lag:] # Now Varlag is the base...
        varbase  = var1[:(ntime-lag)]
        # Calculate correlation
        #beta = np.polyfit(varlag,varbase,1)[0] 
        beta = sp.stats.linregress(varlag,varbase)[2]
        betalead.append(beta.item())
    
    
    # Append Together
    return np.concatenate([np.flip(np.array(betalead)),np.array(betalag)])



#%% Given 2 Time series, Looping

# Take the lag correlation across a set of lags (-12,12)



# plotvar = output['regression_coeff']
# mask    = output['sigmask']
# lat     = mocin.lat_aux_grid.data
# z       = mocin.moc_z.data/100

# pcm     = ax.pcolormesh(lat,z,plotvar,cmap='cmo.balance',vmin=-2.5,vmax=2.5)

# #cc  = ax.contour(lat,z,plotvar,colors='blue',levels=[0,])

# sigp    = viz.plot_mask(lat,z,mask.T,ax=ax,color='gray')

# cl      = ax.contour(lat,z,amocmean.isel(moc_comp=mc),colors="k",levels=levels,linewidths=.75)

# #ax.set_ylim([0,200])
# ax.invert_yaxis()

# cb      = viz.hcbar(pcm,ax=ax)
# cb.set_label("Sv MOC Transport per degC SPGNE Index")
# ax.set_title(comps[mc])

# ax.set_xlabel("Latitude")
# ax.set_ylabel("Depth (meters)")


# plt.show()

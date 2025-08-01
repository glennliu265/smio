#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test script to regress GMSST, but using monthly values

Created on Wed Jul 30 17:59:32 2025

@author: gliu

"""


from amv import proc, viz
import cvd_utils as cvd
import amv.loaders as dl
import scm
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

# %%

# local device (currently set to run on Astraeus, customize later)
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/"  # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)


# %% Load GMSST for Detrending (from common_load)

# Load GMSST
dpath_gmsst = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
nc_gmsst = "ERA5_GMSST_1979_2024.nc"
ds_gmsst = xr.open_dataset(
    dpath_gmsst + nc_gmsst).load()  # .GMSST_MeanIce.load()


def detrend_gm(ds_in): return proc.detrend_by_regression(
    ds_in, ds_gmsst.Mean_Ice)


# Load GMSST (Older)
nc_gmsst_pre = "ERA5_GMSST_1940_1978.nc"
ds_gmsst_pre = xr.open_dataset(
    dpath_gmsst + nc_gmsst_pre).load()  # .GMSST_MeanIce.load()

ds_gmsst_merge = xr.concat([ds_gmsst_pre, ds_gmsst], dim='time')

# Set Figure Path
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20250801/"
proc.makedir(figpath)


# %% Load ERA5 Datasets

dp = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
nc2 = "ERA5_sst_NAtl_1940to1978.nc"
nc1 = "ERA5_sst_NAtl_1979to2024.nc"
ncs = [dp+nc2, dp+nc1,]


dsall = xr.open_mfdataset(ncs, combine='nested', concat_dim='time').load()

# %% Restrict to Region/Time

ystart = '1979'
yend = '2024'


#bbox = [-40, -15, 52, 62]  # [-80,0,0,65]#
#regname = "SPGNE"  # NATL

bbox = [-75,-40,37,45]
regname = "GS"

dsreg = proc.sel_region_xr(dsall, bbox)
dsreg = dsreg.sel(time=slice("%s-01-01" % ystart, "%s-12-31" % yend))

ds_gmsst_sel = ds_gmsst_merge.GMSST_MaxIce.sel(
    time=slice("%s-01-01" % ystart, "%s-12-31" % yend))

# %% Initial Preprocessing (Deseasonalize)

dsanom = proc.xrdeseason(dsreg).sst


# %% Now Perform Monthly regression (based on function)


invar = dsanom
in_ts = ds_gmsst_sel
regress_monthly = True

mon1 = proc.detrend_by_regression(invar, in_ts, regress_monthly=True)
mon0 = proc.detrend_by_regression(invar, in_ts, regress_monthly=False)


# %% Lets Explore Some Differences
lags = np.arange(61)


sst_in = [dsanom, mon0.sst, mon1.sst]
aavgs  = [proc.area_avg_cosweight(sst).data for sst in sst_in]

tsm    = scm.compute_sm_metrics(aavgs, lags=lags, nsmooth=2, detrend_acf=False)

# Repeat without linear detrend before ACF (note, this only impacts ACF calculation)
tsm2   = scm.compute_sm_metrics(aavgs, lags=lags, nsmooth=2 ,detrend_acf=True)

# %% Check Impacts on ACF

kmonth  = 2
xtks    = lags[::3]

fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(8, 4.5))
ax, _   = viz.init_acplot(kmonth, xtks, lags, ax=ax)

mlabs   = ["Raw Anomaly",
         "GMSST Regression (All Months)", "GMSST Regression (Separately by Month)"]

mcols = ["midnightblue",'cornflowerblue','hotpink']
for ii in range(3):
    plotvar = tsm['acfs'][kmonth][ii]
    ax.plot(lags, plotvar, label=mlabs[ii], lw=2.5,c=mcols[ii])
    if ii == 0:
        plotvar = tsm2['acfs'][kmonth][ii]
        #lab = mlabs[ii] + "+ monthly linear detrend"
        ax.plot(lags, plotvar, label="Linear Detrend by Month", lw=2.5,c='r',ls='dashed')

ax.legend()

#%% Check Impacts On Monthly Variance
mons3 = proc.get_monstr()

fig,ax = viz.init_monplot(1,1)

for ii in range(3):
    plotvar = tsm['monvars'][ii]
    ax.plot(mons3, plotvar, label=mlabs[ii], lw=2.5,c=mcols[ii])
    
    # if ii == 0:
    #     plotvar = tsm2['monvars'][ii]
    #     ax.plot(mons3, plotvar, label="Linear Detrend by Month", lw=2.5,c='r',ls='dashed')
        

ax.legend()

ax.set_ylabel("Montly Variance (degC)")



# %% Check Regression Coefficients over the region

mons3 = proc.get_monstr()

fig, ax = viz.init_monplot(1, 1)
regpat = mon1['regression_pattern']


# Plot Monthly Values
pat_avg = proc.area_avg_cosweight(regpat)
_, nlat, nlon = regpat.shape
for a in tqdm.tqdm(range(nlat)):
    for o in range(nlon):
        plotvar = regpat.isel(lat=a, lon=o)
        ax.plot(mons3, plotvar, alpha=0.1)
ax.plot(mons3, pat_avg, color="hotpink", label="Region Avg., Separate Month")


# Now Get the Constants
regpat0 = mon0['regression_pattern']
avg0 = proc.area_avg_cosweight(regpat0)

std0 = regpat0.data.reshape(nlat*nlon).std()
ax.axhline([avg0], label="Regression Coeff (Region avg., All Months)",
           color='cornflowerblue', ls='dashed')

ax.axhline([avg0+std0], label="", color='cornflowerblue', ls='dotted')

ax.axhline([avg0-std0], label="Regression Coeff (Region std, All Months)",
           color='cornflowerblue', ls='dotted')


ax.legend()

#%% Look at spectra

vunit       = "degC"
dtmon_fix   = 60*60*24*30
xper        = np.array([40, 10, 5, 1, 0.5])
xper_ticks  = 1 / (xper*12)

fig, ax     = plt.subplots(1, 1, figsize=(8, 4.5), constrained_layout=True)

for ii in range(3):

    plotspec = tsm['specs'][ii] / dtmon_fix
    plotfreq = tsm['freqs'][ii] * dtmon_fix

    color_in = mcols[ii]

    ax.loglog(plotfreq, plotspec, lw=2.5, label=mlabs[ii], c=color_in,
              ls='solid', markersize=2.5)


# plotspec = tsm2['specs'][0] / dtmon_fix
# plotfreq = tsm2['freqs'][0] * dtmon_fix
# color_in = 'r'
# ax.loglog(plotfreq, plotspec, lw=2.5, label="Linear Detrend by Month", c=color_in,
#           markersize=2.5,ls='dashed')



ax.set_xlim([1/120, 0.5])
ax.axvline([1/(6)], label="", ls='dotted', c='gray')
ax.axvline([1/(12)], label="", ls='dotted', c='gray')
ax.axvline([1/(5*12)], label="", ls='dotted', c='gray')
ax.axvline([1/(10*12)], label="", ls='dotted', c='gray')
ax.axvline([1/(40*12)], label="", ls='dotted', c='gray')

ax.set_ylabel("Power [$%s ^2 cycle \, per \, mon$]" % vunit)

ax2 = ax.twiny()
ax2.set_xscale('log')
ax2.set_xticks(xper_ticks, labels=xper)
ax2.set_xlim([1/120, 0.5])

ax.legend(fontsize=12,ncol=2)


# ======================
#%% Looking at Patterns
# ======================
# NOTE: This was written to run with the region set to NATL




# %% Look At the Regression patterns
proj = ccrs.PlateCarree()

fsz_title = 18
fsz_ticks = 18

# regpat  = mon1['intercept']
# regpat0 = mon0['intercept']

vmax = 6
regpat = mon1['regression_pattern']

for im in range(12):

    # fig,ax,_ = viz.init_regplot(regname="SPGE")
    fig, ax, _ = viz.init_regplot()
    plotvar = regpat.isel(mon=im)

    pcm = ax.pcolormesh(plotvar.lon, plotvar.lat, plotvar, transform=proj,
                        vmin=-vmax, vmax=vmax, cmap='cmo.balance')

    plotvar = mon1.sigmask.isel(mon=im)
    xx = viz.plot_mask(plotvar.lon, plotvar.lat, plotvar.T, markersize=.1,
                       proj=proj, geoaxes=True, ax=ax)

    cb = viz.hcbar(pcm, ax=ax)
    cb.ax.tick_params(labelsize=fsz_ticks)

    cb.set_label("SST (deg C per deg C GMSST", fontsize=fsz_ticks)
    ax.set_title("%s Regression" % mons3[im], fontsize=fsz_title)

    figname = "%sGMSST_Regression_%s_Year_%sto%s_mon%02i.png" % (
        figpath, regname, ystart, yend, im+1)
    plt.savefig(figname, dpi=150, bbox_inches='tight')


# %% Plot Pattern for All Months

fig, ax, _ = viz.init_regplot()
regpat0 = mon0['regression_pattern']
plotvar = regpat0  # .isel(mon=im)

pcm = ax.pcolormesh(plotvar.lon, plotvar.lat, plotvar, transform=proj,
                    vmin=-vmax, vmax=vmax, cmap='cmo.balance')

plotvar = mon0.sigmask
xx = viz.plot_mask(plotvar.lon, plotvar.lat, plotvar.T, markersize=.1,
                   proj=proj, geoaxes=True, ax=ax)

cb = viz.hcbar(pcm, ax=ax)
cb.ax.tick_params(labelsize=fsz_ticks)
cb.set_label("SST (deg C per deg C GMSST", fontsize=fsz_ticks)
ax.set_title("All Months Regression", fontsize=fsz_title)

figname = "%sGMSST_Regression_%s_Year_%sto%s_ALL_Month.png" % (
    figpath, regname,ystart, yend)
plt.savefig(figname, dpi=150, bbox_inches='tight')


# %% Ideas:

"""
(1) Plot seasonal differences in regression coefficient
(2) Plot differences from the all-regression

"""


# %% Idea (1), Look at where there is the largest (mean) difference with all-months

regpat0 = mon0['regression_pattern']
regpat1 = mon1['regression_pattern']
rmse = np.sqrt(((regpat1 - regpat0)**2).mean('mon'))

vmax = 2

fig, ax, _ = viz.init_regplot()
plotvar = rmse
cints = np.arange(0, 2.4, .4)

pcm = ax.pcolormesh(plotvar.lon, plotvar.lat, plotvar, transform=proj,
                    vmin=0, vmax=vmax, cmap='cmo.matter')

cl = ax.contour(plotvar.lon, plotvar.lat, plotvar, transform=proj,
                levels=cints, colors="k", linewidths=0.55)
clbl = ax.clabel(cl, fontsize=12)
viz.add_fontborder(clbl, w=2)

cb = viz.hcbar(pcm, ax=ax)
cb.ax.tick_params(labelsize=fsz_ticks)
cb.set_label("Regression Slope RMSE (Seperate Month - All Month)",
             fontsize=fsz_ticks)
# ax.set_title("",fontsize=fsz_title)

figname = "%sGMSST_Regression_Year_%sto%s_RMSE_Comparison.png" % (
    figpath, ystart, yend)
plt.savefig(figname, dpi=150, bbox_inches='tight')

# %% Idea (2) Look at location with largest seasonal range
# rmse    = np.sqrt(((regpat1 - regpat0)**2).mean('mon'))

srange = regpat1.max('mon') - regpat1.min('mon')

vmax = 6

fig, ax, _ = viz.init_regplot()
plotvar = srange
cints = np.arange(0, 8, 1)

pcm = ax.pcolormesh(plotvar.lon, plotvar.lat, plotvar, transform=proj,
                    vmin=0, vmax=vmax, cmap='cmo.turbid')

cl = ax.contour(plotvar.lon, plotvar.lat, plotvar, transform=proj,
                levels=cints, colors="k", linewidths=0.55)
# clbl = ax.clabel(cl,fontsize=8)
# viz.add_fontborder(clbl,w=2)

cb = viz.hcbar(pcm, ax=ax)
cb.ax.tick_params(labelsize=fsz_ticks)
cb.set_label("Seasonal Range in Regression Slope", fontsize=fsz_ticks)
# ax.set_title("",fontsize=fsz_title)

figname = "%sGMSST_Regression_Year_%sto%s_Seasonal_Range_SepMon.png" % (
    figpath, ystart, yend)
plt.savefig(figname, dpi=150, bbox_inches='tight')


# =================
# %% Point Analysis
# =================
"""

In this section, we examine effects of regressing out GMSST at a selected point
and sensitivity to using All Months vs Separate months...

"""


# %% Check Timeseries Removal At a Point

lonf = -73
latf = 37
locfn, loctitle = proc.make_locstring(lonf, latf)

loopds = [mon0, mon1]
cols = ['cornflowerblue', 'hotpink']
loopnames = ["All Month", "Seperate Month"]

loopname_short = ["AllMon", "SepMon"]

ymax = 7.5

# PLot GMSST removal
for ii in range(2):

    fig, ax = plt.subplots(1, 1, figsize=(12.5, 4.5))

    # PLot Raw SST
    sst_raw = proc.selpt_ds(dsanom, lonf, latf)
    ax.plot(sst_raw.time, sst_raw, label="Raw SST Anomaly", c='midnightblue')

    loopin = loopds[ii]

    sstpt = proc.selpt_ds(loopin.sst, lonf, latf)
    fitpt = proc.selpt_ds(loopin.fit, lonf, latf)

    # xrange = np.arange(len(fitpt))
    lab = loopnames[ii]
    ax.plot(sstpt.time.data, sstpt,
            color=cols[ii], ls='solid', label="Detrended", lw=1)
    ax.plot(fitpt.time.data, fitpt, color="k", ls='dashed', label="Fit")

    # ax.plot(xrange,fitpt,color=cols[ii],ls='dashed')
    ax.legend(fontsize=14, ncol=2)
    ax.axhline([0], color='k', lw=0.44)
    ax.set_ylim([-ymax, ymax])
    ax.set_ylabel("SST (degC)", fontsize=14)
    ax.set_xlim([sstpt.time.isel(time=0), sstpt.time.isel(time=-1)])
    ax.tick_params(labelsize=14)
    ax.set_title("GMSST Regression using: " +
                 loopnames[ii] + "s\n %s" % loctitle, fontsize=16)

    figname = "%sGMSST_Regression_Timeseries_%s_%s.png" % (
        figpath, locfn, loopname_short[ii])
    plt.savefig(figname, dpi=150, bbox_inches='tight')


# %% Check Regression Coefficients

coeffs = [proc.selpt_ds(ds.regression_pattern, lonf, latf) for ds in loopds]

fig, ax = viz.init_monplot(1, 1, figsize=(6, 4))


ax.plot(mons3, coeffs[1].data, c=cols[1], label=loopnames[1], lw=2.5)


allmonval = coeffs[0].data.item()
lab = r"%s ($\beta=%.2f$)" % (loopnames[0], allmonval)
ax.axhline(allmonval, c=cols[0], label=lab, lw=2.5)
ax.set_ylabel("degC per degC GMSST")

ax.axhline([0], c="k", label="", lw=1)
ax.set_ylim([-10, 10])

ax.legend()

# %% Examine Impact on SST, calculate some metrics

sstraw_pt = proc.selpt_ds(dsanom, lonf, latf).data
nsmooth = 5
sstin = [proc.selpt_ds(mon0.sst, lonf, latf).data,
         proc.selpt_ds(mon1.sst, lonf, latf).data]
tsm = scm.compute_sm_metrics(
    sstin, lags=lags, nsmooth=nsmooth, detrend_acf=False)
tsmlin = scm.compute_sm_metrics(
    [sstraw_pt,], lags=lags, nsmooth=nsmooth, detrend_acf=True)
tsmraw = scm.compute_sm_metrics(
    [sstraw_pt,], lags=lags, nsmooth=nsmooth, detrend_acf=False)


# %% Plot the ACF

kmonth = 1
xtks = lags[::3]

fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(8, 4.5))
ax, _ = viz.init_acplot(kmonth, xtks, lags, ax=ax)

plotvar = tsmraw['acfs'][kmonth][0]
ax.plot(lags, plotvar, label="SST Anomaly Raw", lw=2.5, c="midnightblue")


plotvar = tsmlin['acfs'][kmonth][0]
ax.plot(lags, plotvar, label="Linear Detrend By Month", lw=2.5, c="red")

for ii in range(2):
    plotvar = tsm['acfs'][kmonth][ii]
    lab = "Regress GMSST (%s)" % loopnames[ii]
    ax.plot(lags, plotvar, label=lab, lw=2.5, c=cols[ii])

ax.legend()

# %% Take a look at the spectra


vunit = "degC"
dtmon_fix = 60*60*24*30
xper = np.array([40, 10, 5, 1, 0.5])
xper_ticks = 1 / (xper*12)


fig, ax = plt.subplots(1, 1, figsize=(8, 4.5), constrained_layout=True)

for ii in range(2):

    plotspec = tsm['specs'][ii] / dtmon_fix
    plotfreq = tsm['freqs'][ii] * dtmon_fix

    color_in = cols[ii]

    ax.loglog(plotfreq, plotspec, lw=2.5, label=loopnames[ii], c=color_in,
              ls='solid', markersize=2.5)


# Also Plot other cases
plotspec = tsmraw['specs'][0] / dtmon_fix
plotfreq = tsmraw['freqs'][0] * dtmon_fix
ax.loglog(plotfreq, plotspec, lw=2.5, label='Raw Anomaly', c='k',
          ls='dashed', markersize=2.5)


# Also Plot other cases
plotspec = tsmlin['specs'][0] / dtmon_fix
plotfreq = tsmlin['freqs'][0] * dtmon_fix
ax.loglog(plotfreq, plotspec, lw=2.5, label="Linear Detrend By Month", c='r',
          ls='dotted', markersize=2.5)


ax.set_xlim([1/120, 0.5])
ax.axvline([1/(6)], label="", ls='dotted', c='gray')
ax.axvline([1/(12)], label="", ls='dotted', c='gray')
ax.axvline([1/(5*12)], label="", ls='dotted', c='gray')
ax.axvline([1/(10*12)], label="", ls='dotted', c='gray')
ax.axvline([1/(40*12)], label="", ls='dotted', c='gray')

ax.set_ylabel("Power [$%s ^2 cycle \, per \, mon$]" % vunit)

ax2 = ax.twiny()
ax2.set_xscale('log')
ax2.set_xticks(xper_ticks, labels=xper)
ax2.set_xlim([1/120, 0.5])

ax.legend(fontsize=14)


# %%


# %%


# %%


# %%

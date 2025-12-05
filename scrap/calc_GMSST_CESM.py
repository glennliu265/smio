#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate Global Mean SST for CESM2 Hierarchy

(1) Load in TS
(2) Apply Ice Mask
(3) Take Area-weighted Global Mean



Created on Fri Oct 17 14:11:32 2025

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

#%% stormtrack modules

amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
scmpath = "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/" # scm module

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl

#%%
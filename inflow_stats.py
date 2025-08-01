# -*- coding: utf-8 -*-
'''
Plot inflow stats
'''

import os
cd=os.path.dirname(__file__)
import sys
import warnings
from datetime import datetime
import yaml
import pandas as pd
import numpy as np
import glob
import xarray as xr
from scipy import stats
import scipy as sp
from matplotlib import pyplot as plt
from windrose import WindroseAxes
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['savefig.dpi'] = 500
warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs
source_inflow=os.path.join(cd,'data/g3p3/roof.lidar.z01.c2/roof.lidar.z01.c2.20250314.20250720.csv')

#%% Initialization

#read inflow data
inflow_df=pd.read_csv(source_inflow).set_index('Time (UTC)')
inflow_df.index= pd.to_datetime(inflow_df.index)
inflow=xr.Dataset.from_dataframe(inflow_df).rename({'Time (UTC)':'time'})

ws=inflow.ws.values
wd=inflow.wd.values
tke=inflow.tke.values

#%% Main



#%% Plots
cmap=matplotlib.cm.get_cmap('viridis')
real=~np.isnan(ws+wd)
ax = WindroseAxes.from_ax()
ax.bar(wd[real], ws[real], normed=True,opening=0.8,cmap=cmap,edgecolor="white",bins=((0,2,4,6,8,10,12)))
ax.set_rgrids(np.arange(0,15.1,5), np.arange(0,15.1,5))
ax.set_xticks([0,np.pi/2,np.pi,np.pi/2*3], labels=['N','E','S','W'])
for label in ax.get_yticklabels():
    label.set_backgroundcolor('white')   # Set background color
    label.set_color('black')             # Set text color
    label.set_fontsize(18)               # Optional: tweak size
    label.set_bbox(dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3',alpha=0.5))

plt.legend(draggable=True)



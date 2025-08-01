# -*- coding: utf-8 -*-
'''
Plot inflow stats
'''

import os
cd=os.path.dirname(__file__)
import warnings
import pandas as pd
import numpy as np
import xarray as xr
from scipy import stats
from scipy.stats import weibull_min
from matplotlib import pyplot as plt
from windrose import WindroseAxes
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi'] = 500
warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs
source_inflow=os.path.join(cd,'data/g3p3/roof.lidar.z01.c2/roof.lidar.z01.c2.20250314.20250720.csv')

#stats
bin_hour=np.arange(25)#bins in hour
perc_lim=[5,95]#[%] percentile limits
p_value=0.05# p-value for c.i.

#%% Functions
def filt_stat(x,func,perc_lim=[5,95]):
    '''
    Statistic with percentile filter
    '''
    x_filt=x.copy()
    lb=np.nanpercentile(x_filt,perc_lim[0])
    ub=np.nanpercentile(x_filt,perc_lim[1])
    x_filt=x_filt[(x_filt>=lb)*(x_filt<=ub)]
       
    return func(x_filt)

def filt_BS_stat(x,func,p_value=5,M_BS=100,min_N=10,perc_lim=[5,95]):
    '''
    Statstics with percentile filter and bootstrap
    '''
    x_filt=x.copy()
    lb=np.nanpercentile(x_filt,perc_lim[0])
    ub=np.nanpercentile(x_filt,perc_lim[1])
    x_filt=x_filt[(x_filt>=lb)*(x_filt<=ub)]
    
    if len(x_filt)>=min_N:
        x_BS=bootstrap(x_filt,M_BS)
        stat=func(x_BS,axis=1)
        BS=np.nanpercentile(stat,p_value)
    else:
        BS=np.nan
    return BS

def bootstrap(x,M):
    '''
    Bootstrap sample drawer
    '''
    i=np.random.randint(0,len(x),size=(M,len(x)))
    x_BS=x[i]
    return x_BS


#%% Initialization

#read inflow data
inflow_df=pd.read_csv(source_inflow).set_index('Time (UTC)')
inflow_df.index= pd.to_datetime(inflow_df.index)
inflow=xr.Dataset.from_dataframe(inflow_df).rename({'Time (UTC)':'time'})

#extract variables
ws=inflow.ws.values
wd=inflow.wd.values
tke=inflow.tke.values

#%% Main

#hour vector
hour=np.array([(t-np.datetime64(str(t)[:10]))/np.timedelta64(1,'h') for t in inflow.time.values])

#daily cycles
hour_avg=(bin_hour[:-1]+bin_hour[1:])/2

f_sel=ws
real=~np.isnan(f_sel)
ws_avg= stats.binned_statistic(hour[real], f_sel[real],statistic=lambda x: filt_stat(x,   np.nanmean,perc_lim=perc_lim),                          bins=bin_hour)[0]
ws_low= stats.binned_statistic(hour[real], f_sel[real],statistic=lambda x: filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=p_value/2*100),    bins=bin_hour)[0]
ws_top= stats.binned_statistic(hour[real], f_sel[real],statistic=lambda x: filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=(1-p_value/2)*100),bins=bin_hour)[0]

f_sel=tke
real=~np.isnan(f_sel)
tke_avg= stats.binned_statistic(hour[real], f_sel[real],statistic=lambda x: filt_stat(x,   np.nanmean,perc_lim=perc_lim),                          bins=bin_hour)[0]
tke_low= stats.binned_statistic(hour[real], f_sel[real],statistic=lambda x: filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=p_value/2*100),    bins=bin_hour)[0]
tke_top= stats.binned_statistic(hour[real], f_sel[real],statistic=lambda x: filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=(1-p_value/2)*100),bins=bin_hour)[0]

#weibull fit
counts, bins, _ = plt.hist(ws, bins=100, density=True, alpha=0.5, label='Histogram')
c, loc, scale  = weibull_min.fit(ws[~np.isnan(ws)], floc=0) 

pdf = weibull_min.pdf((bins[1:]+bins[:-1])/2, c, loc=loc, scale=scale)

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

plt.figure(figsize=(14,7))
plt.subplot(1,2,1)
plt.plot(hour_avg,ws_avg,'.-k',markersize=10)
plt.fill_between(hour_avg,ws_low,ws_top,color='k',alpha=0.25)
plt.xlabel('Hour (UTC)')
plt.ylabel(r'Wind speed (140-160 m.a.g.l.) [m s$^{-1}$]')
plt.grid()

plt.subplot(1,2,2)
plt.plot(hour_avg,tke_avg,'.-k',markersize=10)
plt.fill_between(hour_avg,tke_low,tke_top,color='k',alpha=0.25)
plt.xlabel('Hour (UTC)')
plt.ylabel(r'TKE (140-160 m.a.g.l.) [m$^2$ s$^{-2}$]')
plt.grid()

plt.figure()
plt.bar((bins[1:]+bins[:-1])/2,counts,width=np.diff(bins)[0],color='k',label='Data')
plt.plot((bins[1:]+bins[:-1])/2,pdf,'r',label=f'Weibull fit: shape = {c:.2f}, scale = {scale:.2f}')
plt.xlabel(r'Wind speed (140-160 m.a.g.l.) [m s$^{-1}$]')
plt.ylabel('P.d.f.')
plt.grid()
plt.legend()




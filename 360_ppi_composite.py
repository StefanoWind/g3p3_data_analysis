# -*- coding: utf-8 -*-
'''
Calculate composite statistics from 360 PPI scans
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
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi'] = 500
warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs

#users inputs
if len(sys.argv)==1:
    sdate='2025-05-01' #start date
    edate='2025-07-02' #end date
    path_config=os.path.join(cd,'configs/config.yaml') #config path
    path_inflow='roof.lidar.z01.c2.20250528.20250531.csv'
    ws_range=[5,12]
    wd_range=[170,190]
    tke_range=[0,1]
    
else:
    sdate=sys.argv[1]
    edate=sys.argv[2]  
    path_config=sys.argv[3]
    path_inflow=sys.argv[4]
    ws_range=[sys.argv[5],sys.argv[6]]
    wd_range=[sys.argv[7],sys.argv[8]]
    tke_range=[sys.argv[9],sys.argv[10]]
 
#stats
min_range=90#[m] minimum range
max_range=500#[m] maximum range
min_azi=150 #[deg] minimum azimuth
max_azi=210 #[deg] maximum azimuth
dx=5 #[m] x-grid spacing
dy=5 #[m] y-grid spacing
perc_lim_u=[5,95]
perc_lim_ti=[1,99]
azi_thickness=4#[deg] blind cone thickness
u_lim=[0.1,2]#limits of u 

#grid limits [m]
xmin=-200 
xmax=200
ymin=-500
ymax=130

interp_limit=5#inpainting range

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

def inpaint(variable,limit=2):
    valid_mask1 = ~np.isnan(variable)
    distance1 = sp.ndimage.distance_transform_edt(~valid_mask1)
    interp_mask1 = (np.isnan(variable)) & (distance1 <= limit)
    yy1, xx1 = np.indices(variable.shape)
    points1 = np.column_stack((yy1[valid_mask1], xx1[valid_mask1]))
    values1 = variable[valid_mask1]
    interp_points1 = np.column_stack((yy1[interp_mask1], xx1[interp_mask1]))
    inpainted = sp.interpolate.griddata(points1, values1, interp_points1, method='linear')
    variable_int=variable.copy()
    variable_int[interp_mask1]=inpainted
    
    return variable_int

def cos_fit(azi,U,wd):
    return U*np.cos(np.radians(azi-wd))
    
#%% Initalization

#configs
with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
print(f"Started processing with inputs: \n start date = {sdate} \n end data = {edate} \n ws_range = {ws_range} \n wd_range = {wd_range} \n tke_range = {tke_range}")
    
#read inflow data
inflow_df=pd.read_csv(os.path.join(config['path_data'],'g3p3/roof.lidar.z01.c2',path_inflow)).set_index('Time (UTC)')
inflow_df.index= pd.to_datetime(inflow_df.index)
inflow=xr.Dataset.from_dataframe(inflow_df).rename({'Time (UTC)':'time'})

#find and load PPI data
source=os.path.join(config['path_data'],'g3p3/roof.lidar.z01.b0/*360.ppi*nc')
files=np.array(sorted(glob.glob(source)))
dates=np.array([datetime.strptime(os.path.basename(f).split('.')[4],'%Y%m%d') for f in files])
sel=(dates>=datetime.strptime(sdate,'%Y-%m-%d'))*(dates<=datetime.strptime(edate,'%Y-%m-%d'))
files_sel=files[sel]

#%% Main

#find inflow conditions associated with each file
time=np.array([np.datetime64(datetime.strptime('.'.join(os.path.basename(f).split('.')[4:6]),"%Y%m%d.%H%M%S")) for f in files_sel])

inflow_int=inflow.interp(time=time)

sel_ws=(inflow_int.ws>=ws_range[0])*(inflow_int.ws<ws_range[1])
sel_wd=(inflow_int.wd>=wd_range[0])*(inflow_int.wd<wd_range[1])
sel_tke=(inflow_int.tke>=tke_range[0])*(inflow_int.tke<tke_range[1])

print(f'{np.sum((sel_ws*sel_ws*sel_tke).values)} files meet conditions')
x_all=[]
y_all=[]
u_all=[]
i_f=0
for f in files_sel[(sel_ws*sel_ws*sel_tke).values]:
    data=xr.open_dataset(f)
    data=data.where((data.range>=min_range)*(data.range<=max_range),drop=True)
    
    #cos fit
    azi=data.azimuth.transpose('range','beamID','scanID').values.ravel()
    rws=data.wind_speed.where(data.qc_wind_speed==0).values.ravel()
    real=~np.isnan(azi+rws)
    popt = sp.optimize.curve_fit(cos_fit, azi[real], -rws[real],bounds=([0,0], [30,360]),
                                 p0=(inflow_int.ws.isel(time=i_f),inflow_int.wd.isel(time=i_f)))[0]
    
    # select azimuth
    data=data.where((data.azimuth>=min_azi)*(data.azimuth<=max_azi),drop=True)
    
    #exclude blind cone
    excl_g3p3=(data.azimuth>config['g3p3_azi']-azi_thickness/2)*(data.azimuth<config['g3p3_azi']+azi_thickness/2)*(data.range>config['g3p3_range'])
    data=data.where(~excl_g3p3)
    excl_hotsstar=(data.azimuth>config['hotsstar_azi']-azi_thickness/2)*(data.azimuth<config['hotsstar_azi']+azi_thickness/2)*(data.range>config['hotsstar_range'])
    data=data.where(~excl_hotsstar)
    
    #deproject velocity
    angle=np.radians(data.azimuth-popt[1])
    u=-data.wind_speed.where(data.qc_wind_speed==0)/np.cos(angle)
    u=u.where((u/popt[0]>u_lim[0])*(u/popt[0]<u_lim[1]))
    
    #stack data
    real=~np.isnan(u.values.ravel())
    x_all=np.append(x_all,data.x.values.ravel()[real])
    y_all=np.append(y_all,data.y.values.ravel()[real])
    u_all=np.append(u_all,u.values.ravel()[real])
    
    i_f+=1
    
#bin average
bin_x=np.arange(xmin-dx/2,xmax+dx,dx)
bin_y=np.arange(ymin-dy/2,ymax+dy,dy)
x=(bin_x[:-1]+bin_x[1:])/2
y=(bin_y[:-1]+bin_y[1:])/2
u_avg=stats.binned_statistic_2d(x_all, y_all, u_all,statistic=lambda x: filt_stat(x,   np.nanmean,perc_lim=perc_lim_u),bins=[bin_x,bin_y])[0]

#inpainting
u_avg_int=inpaint(u_avg,interp_limit)

#bin std
bin_x=np.arange(xmin-dx/2,xmax+dx,dx)
bin_y=np.arange(ymin-dy/2,ymax+dy,dy)
x=(bin_x[:-1]+bin_x[1:])/2
y=(bin_y[:-1]+bin_y[1:])/2
u_std=stats.binned_statistic_2d(x_all, y_all, u_all,statistic=lambda x: filt_stat(x,   np.nanstd,perc_lim=perc_lim_ti),bins=[bin_x,bin_y])[0]

#inpainting
u_std_int=inpaint(u_std,interp_limit)
ti_int=u_std_int/u_avg_int*100

#%% Plots
plt.figure(figsize=(18,8.5))
ax=plt.subplot(1,2,1)
vmin=np.floor(np.nanpercentile(u_avg_int,0.1)/0.25)*0.25
vmax=np.ceil(np.nanpercentile(u_avg_int,95)/0.25)*0.25
cf=plt.contourf(x,y,u_avg_int.T,np.arange(vmin,vmax+0.25,0.25),cmap='coolwarm',extend='both')
plt.contour(x,y,u_avg_int.T,np.arange(vmin,vmax+0.25,0.25),colors='k',linewidth=.1,alpha=0.1,extend='both')
ax=plt.gca()
ax.set_aspect('equal')
plt.grid()
plt.plot(np.cos(np.radians(90-config['g3p3_azi']))*config['g3p3_range'],
         np.sin(np.radians(90-config['g3p3_azi']))*config['g3p3_range'],'ks',markersize=15)
plt.plot(np.cos(np.radians(90-config['hotsstar_azi']))*config['hotsstar_range'],
         np.sin(np.radians(90-config['hotsstar_azi']))*config['hotsstar_range'],'ks',markersize=15)
plt.plot(0,0,'ks',markersize=15)
plt.xlabel('W-E [m]')
plt.ylabel('S-N [m]')
plt.text(xmin+10,ymax-100,r'Wind speed limits [m s$^{-1}$]: '+str(ws_range[0])+'-'+str(ws_range[1])+'\n'+\
                     r'Wind direction limits [$^\circ$]: '+ str(wd_range[0])+'-'+str(wd_range[1])+'\n'+\
                     r'TKE limits [m$^2$/s$^2$]: '+str(tke_range[0])+'-'+str(tke_range[1])+'\n'+\
                     f'File count: {np.sum((sel_ws*sel_ws*sel_tke).values)}',
                     bbox={'edgecolor':'k','facecolor':'w'})
plt.xticks(np.arange(xmin,xmax+1,100))
plt.yticks(np.arange(ymin,ymax+1,100))
plt.colorbar(cf,label='Wind speed [m/s]')

ax=plt.subplot(1,2,2)
vmin=np.floor(np.nanpercentile(ti_int,5)/2)*2
vmax=np.ceil(np.nanpercentile(ti_int,99.5)/2)*2
cf=plt.contourf(x,y,ti_int.T,np.arange(vmin,vmax+2,2),cmap='hot',extend='both')
plt.contour(x,y,ti_int.T,np.arange(vmin,vmax+2,2),colors='k',linewidth=.1,alpha=0.1,extend='both')
ax=plt.gca()
ax.set_aspect('equal')
plt.grid()
plt.plot(np.cos(np.radians(90-config['g3p3_azi']))*config['g3p3_range'],
         np.sin(np.radians(90-config['g3p3_azi']))*config['g3p3_range'],'ks',markersize=15)
plt.plot(np.cos(np.radians(90-config['hotsstar_azi']))*config['hotsstar_range'],
         np.sin(np.radians(90-config['hotsstar_azi']))*config['hotsstar_range'],'ks',markersize=15)
plt.plot(0,0,'ks',markersize=15)
plt.xlabel('W-E [m]')
plt.ylabel('S-N [m]')
plt.text(xmin+10,ymax-100,r'Wind speed limits [m s$^{-1}$]: '+str(ws_range[0])+'-'+str(ws_range[1])+'\n'+\
                     r'Wind direction limits [$^\circ$]: '+ str(wd_range[0])+'-'+str(wd_range[1])+'\n'+\
                     r'TKE limits [m$^2$/s$^2$]: '+str(tke_range[0])+'-'+str(tke_range[1])+'\n'+\
                     f'File count: {np.sum((sel_ws*sel_ws*sel_tke).values)}',
                     bbox={'edgecolor':'k','facecolor':'w'})
plt.xticks(np.arange(xmin,xmax+1,100))
plt.yticks(np.arange(ymin,ymax+1,100))
plt.colorbar(cf,label='TI [%]')

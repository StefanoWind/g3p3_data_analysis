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
from matplotlib import pyplot as plt
warnings.filterwarnings('ignore')

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
    sdate=sys.argv[1] #start date
    edate=sys.argv[2]  #end date
    path_config=sys.argv[3]#config path
    ws_range=sys.argv[4]
    wd_range=sys.argv[5]
    tke_range=sys.argv[6]
 
min_range=90
max_range=500
min_azi=150
max_azi=210
    
#%% Initalization

#configs
with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
inflow_df=pd.read_csv(os.path.join(config['path_data'],'g3p3/roof.lidar.z01.c2',path_inflow)).set_index('Time (UTC)')
inflow_df.index= pd.to_datetime(inflow_df.index)
inflow=xr.Dataset.from_dataframe(inflow_df).rename({'Time (UTC)':'time'})

#find and load data
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
    data=data.where((data.range>min_range)*(data.range<max_range)*(data.azimuth>min_azi)*(data.azimuth<max_azi),drop=True)
    
    angle=np.radians(data.azimuth-inflow_int.wd.isel(time=i_f))
    u=-data.wind_speed.where(data.qc_wind_speed==0)/np.cos(angle)
    
    real=~np.isnan(u.values.ravel())
    x_all=np.append(x_all,data.x.values.ravel()[real])
    y_all=np.append(y_all,data.y.values.ravel()[real])
    u_all=np.append(u_all,u.values.ravel()[real])
    
    i_f+=1
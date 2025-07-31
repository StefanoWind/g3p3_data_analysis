# -*- coding: utf-8 -*-
'''
Create inflow table for G3P3
'''

import os
cd=os.path.dirname(__file__)
import sys
import warnings
from datetime import datetime
import xarray as xr
import yaml
import numpy as np
import glob
import pandas as pd

warnings.filterwarnings('ignore')

#%% Inputs

#users inputs
if len(sys.argv)==1:
    sdate='2025-05-01' #start date
    edate='2025-07-02' #end date
    path_config=os.path.join(cd,'configs/config.yaml') #config path
else:
    sdate=sys.argv[1] #start date
    edate=sys.argv[2]  #end date
    path_config=sys.argv[3]#config path
    
#%% Initalization

#configs
with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#finad and load data
source=os.path.join(config['path_data'],'g3p3/roof.lidar.z01.c1/*six.beam*nc')
files=np.array(sorted(glob.glob(source)))
dates=np.array([datetime.strptime(os.path.basename(f).split('.')[4],'%Y%m%d') for f in files])
sel=(dates>=datetime.strptime(sdate,'%Y-%m-%d'))*(dates<=datetime.strptime(edate,'%Y-%m-%d'))
data=xr.open_mfdataset(files[sel])

#system
os.makedirs(os.path.join('data','roof.lidar.z01.c2'),exist_ok=True)

#%% Main

#extract variables
WS=data.WS.compute()
WD=data.WD.compute()
TKE=data.tke.compute()

#find minumum available height
height_min=data.height.values[np.where((~np.isnan(WS)).sum(dim='time').values>0)[0][0]]

#extract heights
ws=WS.sel(height=slice(height_min-0.1,height_min+config['thickness']+0.2)).median(dim='height')

cos=np.cos(np.radians(WD.sel(height=slice(height_min-0.1,height_min+config['thickness']+0.2)).median(dim='height')))
sin=np.sin(np.radians(WD.sel(height=slice(height_min-0.1,height_min+config['thickness']+0.2)).median(dim='height')))
wd=np.degrees(np.arctan2(sin,cos))%360

tke=TKE.sel(height=slice(height_min-0.1,height_min+config['thickness']+0.2)).median(dim='height')

df={'ws':ws,'wd':wd,'tke':tke}

#%% Output
output=pd.DataFrame(data=df,index=data.time.values)

t1=str(data.time.values[0])[:10].replace('-','')
t2=str(data.time.values[-1])[:10].replace('-','')
output.to_csv(os.path.join('data','roof.lidar.z01.c2',f'roof.lidar.z01.c2.{t1}.{t2}.csv'))
    
    
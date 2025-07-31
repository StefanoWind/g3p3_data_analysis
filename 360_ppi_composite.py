# -*- coding: utf-8 -*-
'''
Calculate composite statistics from 360 PPI scans
'''

import os
cd=os.path.dirname(__file__)
import sys
import traceback
import warnings
import lidargo as lg
from datetime import datetime
import yaml
from multiprocessing import Pool
import logging
import re
import panda as pd
import glob

warnings.filterwarnings('ignore')

#%% Inputs

#users inputs
if len(sys.argv)==1:
    sdate='2025-07-01' #start date
    edate='2025-07-02' #end date
    path_config=os.path.join(cd,'configs/config.yaml') #config path
    path_inflow='roof.lidar.z01.c2.20250528.20250530.csv'
else:
    sdate=sys.argv[1] #start date
    edate=sys.argv[2]  #end date
    path_config=sys.argv[3]#config path
    
#%% Initalization

#configs
with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
inflow=pd.read_csv(os.path.join(config['path_data'],path_inflow))
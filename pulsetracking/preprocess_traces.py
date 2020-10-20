''' load rate traces and bin data in seconds. Then save the mean and variance of EODr for each fish.
'''
import pickle
import numpy as np
from datetime import datetime, timedelta
import os

load_path = '../data/results/'
save_path = '../data/results/'

N = 60*60*24*5*10

dates = ['2019-10-17-12_36','2019-10-17-19_48','2019-10-18-09_52','2019-10-19-08_39','2019-10-20-08_30']
datis = [datetime(2019,10,17,13,36),datetime(2019,10,17,19,48),datetime(2019,10,18,9,52),datetime(2019,10,19,8,39),datetime(2019,10,20,8,30)]

firing_rates_storage = np.zeros((N))
var_storage = np.zeros((N))
position_storage = np.zeros((N,2))
firing_rates_storage[:] = np.nan
var_storage[:] = np.nan
timestamps = np.zeros((N))
clusters = np.zeros((N))
pk_max = 0
n=0

for date,dati in zip(dates,datis):
    
    dpk = pk_max
    
    path = load_path+'/%s/'%date
    
    file = [i for i in os.listdir(path) if '.pkl' in i] 
    
    if len(file) == 0:
        continue
    
    pd_ob = pickle.load(open(path+file[0],'rb'))

    positions = pd_ob['positions']
    sts = pd_ob['sts']
    isis = pd_ob['isis']
    fsts = pd_ob['fsts']
    fisis = pd_ob['fisis']
    et = pd_ob['et']
    ep = pd_ob['ep']
    
    max_time = 0
    for k,v in sts.items():
        max_time = max(max_time,np.max(v))
            
    for t in np.arange(0,np.floor(max_time)):

        for i,pk in enumerate(sts.keys()):
            if np.count_nonzero((sts[pk]>t) & (sts[pk]<(t+1))) > 1:

                p = positions[pk][:-1]

                x = sts[pk]
                y = isis[pk]
                fx = fsts[pk]
                fy = fisis[pk]

                firing_rates_storage[n] = np.nanmean(y[(x>t) & (x<t+1)])
                var_storage[n] = np.nanvar(y[(x>t) & (x<t+1)])
                position_storage[n] = np.nanmean(p[(x>t) & (x<t+1)],axis=0)
                timestamps[n] = (dati + timedelta(0,int(t))).timestamp()
                clusters[n] = pk + dpk
                n=n+1
                pk_max = max(pk_max,pk+dpk)

firing_rates_storage = firing_rates_storage[:n]
var_storage = var_storage[:n]
position_storage = position_storage[:n]
timestamps = timestamps[:n]
clusters = clusters[:n]

np.savez(save_path+'processed_traces.npz',frs=firing_rates_storage,vs=var_storage,pos=position_storage,ts=timestamps,cl=clusters)
import thunderfish.dataloader as dl
import thunderfish.eventdetection as ed
import pulse_tracker_helper_new as pth
import thunderfish.pulses as lp

import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.interpolate import interp1d
from scipy.signal import correlate
from scipy.signal import argrelextrema
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LinearRegression
from scipy.optimize import lsq_linear
from scipy.ndimage.measurements import center_of_mass

from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from thunderfish.eventdetection import detect_peaks


import pickle
import re

import copy

import argparse

from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.metrics import pairwise_distances

import warnings

from numba import jit, int64

'''
This is the version that is running rn on orca.
Results are saved in: t3
'''

cmap = plt.get_cmap("tab10")

def fxn():
    warnings.warn("runtime", RuntimeWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

#@jit(nopython=True)
def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def get_position(spatial_pattern,grid_shape=(4,8),n_elec=2):
    spatial_pattern[~np.isfinite(spatial_pattern)] = 0
    mask_idx = spatial_pattern.argsort()[-n_elec:][::-1]
    mask = np.zeros(len(spatial_pattern))
    mask[mask_idx] = 1

    return np.asarray(center_of_mass((spatial_pattern*mask).reshape(grid_shape)))

def append_spikes(ts,tmin,tmax):
    
    ISI = np.median(np.diff(ts))

    new_ts = np.concatenate([[tmin-ISI+(np.min(ts)-tmin)%ISI],ts,[tmax+ISI-(tmax-np.max(ts))%ISI]])
    '''
    plt.figure()
    plt.plot(new_ts,np.ones(len(new_ts)),'o')
    plt.plot(ts,np.ones(len(ts)),'o')
    plt.show()
    '''

    return new_ts


def merge_clusters(clusters_1, clusters_2, ts, tmin, tmax, verbose=0):
    """ Merge clusters resulting from two clustering methods.

    This method only works  if clustering is performed on the same EODs
    with the same ordering, where there  is a one to one mapping from
    clusters_1 to clusters_2. 

    Parameters
    ----------
    clusters_1: list of ints
        EOD cluster labels for cluster method 1.
    clusters_2: list of ints
        EOD cluster labels for cluster method 2.
    ts
        Indices of EODs ?.
    verbose : int (optional)
        Verbosity level.

    Returns
    -------
    clusters : list of ints
        Merged clusters.
    x_merged : list of ints
        Merged cluster indices.
    mask : 2d numpy array of ints (N,2)
        Mask for clusters that are selected from clusters_1 (mask[:,0]) and from clusters_2 (mask[:,1]).
    """

    # plot originals
    '''
    plt.figure()
    for i,c in enumerate(np.unique(clusters_1[clusters_1!=-1])):
        plt.plot(ts[clusters_1==c],i*np.ones(len(ts[clusters_1==c])),'o',label = np.var(np.diff(ts[clusters_1==c])))
    plt.title('C1')
    plt.legend()

    plt.figure()
    for i,c in enumerate(np.unique(clusters_2[clusters_2!=-1])):
        plt.plot(ts[clusters_2==c],i*np.ones(len(ts[clusters_2==c])),'o',label = np.var(np.diff(ts[clusters_2==c])))
    plt.title('C2')
    plt.legend()
    '''

    if verbose > 0:
        print('\nMerge cluster:')

    # these arrays become 1 for each EOD that is chosen from that array
    c1_keep = np.zeros(len(clusters_1))
    c2_keep = np.zeros(len(clusters_2))

    # add n to one of the cluster lists to avoid overlap
    ovl = np.max(clusters_1) + 1
    clusters_2[clusters_2!=-1] = clusters_2[clusters_2!=-1] + ovl

    remove_clusters = [[]]
    keep_clusters = []
    og_clusters = [np.copy(clusters_1),np.copy(clusters_2)]
    
    # loop untill done
    while True:

        # compute unique clusters and cluster sizes
        # of cluster that have not been iterated over
        c1_labels = np.unique(clusters_1[(clusters_1!=-1) & (c1_keep == 0)])
        c2_labels = np.unique(clusters_2[(clusters_2!=-1) & (c2_keep == 0)])

        cov_1 = np.zeros(len(c1_labels))
        cov_2 = np.zeros(len(c2_labels))

        for i,c in enumerate(c1_labels):
            cov_1[i] = np.std(np.diff(append_spikes(ts[clusters_1==c],tmin,tmax)))

        for i,c in enumerate(c2_labels):
            cov_2[i] = np.std(np.diff(append_spikes(ts[clusters_2==c],tmin,tmax)))

        # if all clusters are done, break from loop
        if len(c1_labels) == 0 and len(c2_labels) == 0:
            break

        # if the biggest cluster is in c_p, keep this one and discard all clusters on the same indices in c_t
        # elif np.argmax([np.max(np.append(c1_size,0)), np.max(np.append(c2_size,0))]) == 0:
        # check cluster with smallest cov.
        elif np.argmin([np.min(np.append(cov_1,1e100)), np.min(np.append(cov_2,1e100))]) == 0:
            
            # remove all the mappings from the other indices
            cluster_mappings = np.unique(clusters_2[clusters_1==c1_labels[np.argmin(cov_1)]])
            
            clusters_2[np.isin(clusters_2, cluster_mappings)] = -1
            
            c1_keep[clusters_1==c1_labels[np.argmin(cov_1)]] = 1

            remove_clusters.append(cluster_mappings)
            keep_clusters.append(c1_labels[np.argmin(cov_1)])

        # if the biggest cluster is in c_t, keep this one and discard all mappings in c_p
        elif np.argmin([np.min(np.append(cov_1,1e100)), np.min(np.append(cov_2,1e100))]) == 1:
            
            # remove all the mappings from the other indices
            cluster_mappings = np.unique(clusters_1[clusters_2==c2_labels[np.argmin(cov_2)]])
            
            clusters_1[np.isin(clusters_1, cluster_mappings)] = -1
            
            c2_keep[clusters_2==c2_labels[np.argmin(cov_2)]] = 1

            remove_clusters.append(cluster_mappings)
            keep_clusters.append(c2_labels[np.argmin(cov_2)])

    # combine results    
    clusters = (clusters_1+1)*c1_keep + (clusters_2+1)*c2_keep - 1               
                    
    '''
    plt.figure()
    for i,c in enumerate(np.unique(clusters[clusters!=-1])):
        plt.plot(ts[clusters==c],i*np.ones(len(ts[clusters==c])),'o')
    plt.title('merged')
    plt.show()
    '''
      
    return clusters



#### functions for data loading / handling data
def load_channels(master_filepath,slave_filepath,starttime,endtime):
    print('loading master data')
    with dl.open_data(master_filepath, -1, 1.0,10) as data:
        dt = 1/data.samplerate

        # do something with the content of the file:        
        if starttime == None:
            starttime = 0
        if endtime == None:
            endtime = len(data)*dt

        x = np.arange(starttime,endtime,dt)
        y = data[starttime*data.samplerate:endtime*data.samplerate]

    print('loading slave data')
    with dl.open_data(slave_filepath, -1, 1.0,10) as data:
        dt = 1/data.samplerate
      
        # do something with the content of the file:
        
        if starttime == None:
            starttime = 0
        if endtime == None:
            endtime = len(data)*dt

        x = np.arange(starttime,endtime,dt)
        y2 = data[starttime*data.samplerate:endtime*data.samplerate]

    if len(y)==0 or len(y2)==0 or len(y)!=len(y2):
        return [],[],0

    data = np.concatenate([y,y2],axis=1)
    print('data loaded')
    return x,data,dt

def extract_eod_times(data, thresh, peakwidth, samplerate, interp_freq, win_size = 0.0005, n_stds = 1000, threshold_factor=6):

    print('extracting times')

    all_int_data = np.array([])

    channels = []
    x_peaks = []
    x_troughs = []
    eod_hights = []
    eod_widths = []

    for i in range(data.shape[1]):
        y = data[:,i]
        
        x_peak, x_trough, eod_hight, eod_width, int_samplerate, int_data, interp_f, _ = lp.extract_eod_times(y,samplerate,3,interp_freq=interp_freq,min_peakwidth=2.5/samplerate,max_peakwidth=5/samplerate)

        if len(all_int_data)>0:
            all_int_data = np.vstack([all_int_data,int_data])
            x_peaks = np.concatenate((x_peaks,x_peak))
            x_troughs = np.concatenate((x_troughs,x_trough))
            eod_hights = np.concatenate((eod_hights,eod_hight))
            eod_widths = np.concatenate((eod_widths,eod_width))
            channels = np.concatenate((channels,np.ones(len(eod_width))*i))
        else:
            all_int_data = int_data
            x_peaks = x_peak
            x_troughs = x_trough
            eod_hights = eod_hight
            eod_widths = eod_width
            channels = np.ones(len(eod_width))*i

    # sort by peaktimes.n
    a = np.argsort(x_peaks)
    print('times extracted')

    return x_peaks[a], x_troughs[a], eod_hights[a], eod_widths[a], channels[a], all_int_data, int_samplerate, interp_f


### functions for extracting features


### functions for discarding artefacts and wavefish

def detect_artefacts(eods,dt,threshold=8000, cutoff=0.75):

    xf = np.linspace(0.0, 1.0/(2.0*dt), int(len(eods)/2))
    
    fft = np.abs(np.fft.fft(eods.T))[:,:len(xf)]    
    
    LFP = fft[:,xf<threshold]
    LFP_ratio = LFP.sum(axis=1)/fft.sum(axis=1)
   
    output = np.ones(eods.shape[1])
    output[LFP_ratio<cutoff] = 0

    return output

def detect_wavefish(eods,lb=0.3,ub=0.75):
    max_abs_slopes = np.max(np.abs(np.diff(eods,axis=0)),axis=0)
    normalization = np.max(eods,axis=0)-np.min(eods,axis=0)

    max_slope = max_abs_slopes/normalization
    
    output = np.ones(eods.shape[1])
    output[max_slope<lb] = 0
    output[max_slope>ub] = 0
    
    return output


## functions for extracting eel times

#@jit(nopython=True)
def extract_eel_times(data):
    print('extracting eeltimes')
    all_eel_times = []
    for i in range(data.shape[1]):
        y = data[:,i]
        et = zero_runs(np.diff(y))
        aplist = np.mean(et[et[:,1] - et[:,0] > 5], axis=1)

        if len(aplist) > 0:
            all_eel_times = np.append(all_eel_times,aplist)

    if len(all_eel_times) > 0:
        all_eel_times = np.unique(all_eel_times)

    return np.round(all_eel_times)


def plot_snippet(eods,x=None,ct=[],cp=[],cc=[],color='k'):
    
    for i in range(eods.shape[0]):
        plt.subplot(4,8,i+1)
        if x is None:
            x = np.arange(len(eods[i]))
        plt.plot(x,eods[i],color=color,alpha=0.25)
        
        if i in cc:
            try:
                plt.plot(ct[cc==i],eods[i,ct[cc==i]-x[0]],'x',color='b')
            except:
                pass
            try:
                plt.plot(cp[cc==i],eods[i,cp[cc==i]-x[0]],'x',color='r')
            except:
                pass

        plt.ylim([np.min(eods),np.max(eods)])
        plt.axis('off')

def to_coordinates(i,shape):
    return [np.floor(i/shape[1]),i%shape[1]]

def percent_change(a,b):
    return np.abs((a-b)/((a+b)/2))

def argmax_2d(a):
    return np.unravel_index(np.argmax(a, axis=None), a.shape)

def analyse_window(ap,at,ap_n,at_n,tt,maxchans,starttime,endtime,samplerate,int_f,mean_peak_eods,times_p,stimes_p,freqs,counter_p,o_data,data,x_peaks,x_troughs,channels,eod_hights,eod_width,f_width):                  
    
    time_res = 0.25
    window_size=1

    ap = np.vstack(ap)
    at = np.vstack(at)
    ap_n = np.vstack(ap_n)
    at_n = np.vstack(at_n)

    for t in np.arange(starttime,endtime-window_size+time_res,time_res):

        sl = (np.where((tt/samplerate+starttime>=t)&(tt/samplerate+starttime<t+window_size))[0]).astype('int')

        pf_raw = ap[sl]
        tf_raw = at[sl] 

        pf = ap_n[sl]
        tf = at_n[sl]

        ctt = tt[sl]

        if len(pf)<10:
            continue

        #take log of spatial patterns
        s_patterns = np.std(pf.reshape(pf.shape[0],32,-1),axis=2)**(1/3)
        s_patterns[np.isnan(s_patterns)] = 0

        pc = OPTICS(min_samples=10).fit(s_patterns).labels_

        prev_counter_p = counter_p

        for cl in np.unique(pc[pc!=-1]):
            
            # get the maxchan of this cluster
            # not by the average but the most prominent maxchan??
            all_mc, counts = np.unique(np.array(maxchans)[sl][pc==cl],return_counts=True)
            maxchan = all_mc[np.argmax(counts)]#np.argmax(np.mean(np.std(pf[pc==cl].reshape(pf[pc==cl].shape[0],32,-1),axis=2),axis=0))

            mean_snip_p = []
            mean_snip_t = []

            for ct in ctt[pc==cl]:
                # check out surrounding peaks, troughs, and hights.
                # but I should somehow center around the highest peak?
                eod_time = ct                
                snip = (((x_peaks>(eod_time-eod_width*2)) & (x_peaks<(eod_time+eod_width*2)) & (x_troughs>(eod_time-eod_width*6)) & (x_troughs<(eod_time+eod_width*6))) | ((x_troughs>(eod_time-eod_width*2)) & (x_troughs<(eod_time+eod_width*2)) & (x_peaks>(eod_time-eod_width*6)) & (x_peaks<(eod_time+eod_width*6))))
  
                cp = x_peaks[snip]
                ct = x_troughs[snip]
                cc = channels[snip]
                ch = eod_hights[snip]

                if maxchan in cc:
                    width = f_width/2
                    p_eods = np.zeros((32,f_width))
                    t_eods = np.zeros((32,f_width))
                    # center around biggest pt pair.
                    p_eods = data[:, int(cp[cc==maxchan][np.argmax(ch[cc==maxchan])]-width):int(cp[cc==maxchan][np.argmax(ch[cc==maxchan])]+width)]
                    p_eods[np.isnan(p_eods)] = 0 # set nan to zero as broken elctrodes give NaN
                    t_eods = data[:, int(ct[cc==maxchan][np.argmax(ch[cc==maxchan])]-width):int(ct[cc==maxchan][np.argmax(ch[cc==maxchan])]+width)]  
                    t_eods[np.isnan(t_eods)] = 0
                    mean_snip_p.append(p_eods.flatten())
                    mean_snip_t.append(t_eods.flatten())

            mean_snip_p = np.vstack(mean_snip_p)
            mean_snip_t = np.vstack(mean_snip_t)

            # pick the mean snip with the smallest error
            # then check if this is a real EOD
            if np.mean(np.std(mean_snip_p,axis=0))<np.mean(np.std(mean_snip_t,axis=0)):
                mean_snip = np.mean(mean_snip_p,axis=0).reshape(32,-1)
            else:
                mean_snip = np.mean(mean_snip_t,axis=0).reshape(32,-1)

            mean_eod = mean_snip[np.argmax(np.var(mean_snip,axis=1))]
            mean_eod = mean_eod-np.mean(mean_eod)
            snip_peaks, snip_troughs = detect_peaks(mean_eod,np.std(mean_eod))

            cut_fft = int(len(np.fft.fft(mean_eod))/2)
            low_frequency_ratio = np.sum(np.abs(np.fft.fft(mean_eod))[:int(cut_fft/(2*int_f))])/np.sum(np.abs(np.fft.fft(mean_eod))[:int(cut_fft)])           

            if ((len(snip_peaks)+len(snip_troughs)) < 5) and (low_frequency_ratio>0.75):
                mean_peak_eods[counter_p] = mean_snip.flatten()
                times_p[counter_p] = t
                stimes_p.append(tt[sl][pc==cl]/samplerate+starttime)
                freqs[counter_p] = np.median(1/np.diff(tt[sl][pc==cl]/samplerate))
                counter_p = counter_p + 1
                #plt.plot(tt[sl][pc==cl]/samplerate+starttime,cl*np.ones(len(tt[sl][pc==cl])),'o',label=np.std(np.diff(tt[sl][pc==cl]/samplerate))/np.mean(np.diff(tt[sl][pc==cl]/samplerate)))
            #else:
                #plt.plot(tt[sl][pc==cl]/samplerate+starttime,cl*np.ones(len(tt[sl][pc==cl])),'x',label=np.std(np.diff(tt[sl][pc==cl]/samplerate))/np.mean(np.diff(tt[sl][pc==cl]/samplerate)))
        
        #plt.legend()
        #plt.title(int(np.mean(tt[sl])))
        #plt.show()

        # TODO
        # for all that have similar location and frequency,
        # check how many fish there actually are by sliding window.
        # then use the frequency as a factor for sliding window size

        # look at the clusters > 50Hz
        # see if I can/should connect them.
        cur_freqs = freqs[prev_counter_p:counter_p]
        cur_mpe = mean_peak_eods[prev_counter_p:counter_p]
        cur_tt = np.array(stimes_p[prev_counter_p:counter_p])


        if len(cur_mpe)>1:
            # start by the most similar clusters, and only connect them if the tt is improved (less variation)
            s_patterns = np.std(cur_mpe.reshape(cur_mpe.shape[0],32,-1),axis=2)**(1/3)

            # make three matrices.
            pdm = cdist(s_patterns,s_patterns,'euclidean') #distance_matrix(s_patterns,s_patterns).flatten()
            print('CDIST')
            print(pdm)

            # get location of closest nonzero.
            i = np.argmin(pdm[pdm>0])
            idxs = np.argsort(pdm)
            pdm_flat = pdm[idxs][len(cur_mpe):][::2]
            idxs = idxs[len(cur_mpe):][::2]
            pops = []


            for i in idxs:
                x = int(np.mod(i,len(cur_mpe)))
                y = int(np.floor(i/len(cur_mpe)))

                x1=np.std(cur_mpe[x].reshape(32,-1),axis=1)**(1/3)
                x2=np.std(cur_mpe[y].reshape(32,-1),axis=1)**(1/3)

                score = np.linalg.norm(x1-x2)/(0.5*(np.linalg.norm(x1)+np.linalg.norm(x2))) #min(np.sum((x1-x2)/(0.5*(x1+x2))),np.sum((x2-x1)/(0.5*(x1+x2))))
                #print(score)
                #score = np.linalg.norm(x1-x2)/np.sqrt(np.linalg.norm(x1)*np.linalg.norm(x2)) #geometric mean


                # take into account harmonics.
                if np.abs(max(cur_freqs[x],cur_freqs[y])/min(cur_freqs[x],cur_freqs[y]) - np.round(max(cur_freqs[x],cur_freqs[y])/min(cur_freqs[x],cur_freqs[y]))) > 0.2:
                    continue


                if (score<0.5 and cur_freqs[x]==cur_freqs[y]) or (score<0.2 and np.std(np.diff(append_spikes(np.sort(np.concatenate([cur_tt[x],cur_tt[y]])),t,t+window_size)))<min(np.std(np.diff(append_spikes(cur_tt[x],t,t+window_size))),np.std(np.diff(append_spikes(cur_tt[y],t,t+window_size))))) or (score<0.3 and ((np.min(cur_tt[x])>np.max(cur_tt[y]) or np.min(cur_tt[y])>np.max(cur_tt[x])))):
                    #print('yes')
                    if np.arange(prev_counter_p,counter_p)[y] not in pops and np.arange(prev_counter_p,counter_p)[x] not in pops:
                        # append to whichever is more centered in the current timeblock
                        center_d = t+window_size/2

                        if np.abs(center_d - np.mean(cur_tt[x])/samplerate+starttime) < np.abs(center_d - np.mean(cur_tt[y])/samplerate+starttime):
                            stimes_p[np.arange(prev_counter_p,counter_p)[x]] = np.sort(np.concatenate([cur_tt[x],cur_tt[y]])).tolist()
                            pops.append(np.arange(prev_counter_p,counter_p)[y])
                        else:
                            stimes_p[np.arange(prev_counter_p,counter_p)[y]] = np.sort(np.concatenate([cur_tt[x],cur_tt[y]])).tolist()
                            pops.append(np.arange(prev_counter_p,counter_p)[x])
                

            for p in sorted(pops, reverse=True):
                stimes_p.pop(p)
                times_p = np.delete(times_p,p,0)
                mean_peak_eods = np.delete(mean_peak_eods,p,0)
                freqs = np.delete(freqs,p,0)
                counter_p = counter_p-1

        #plt.figure()
        #for i,t in enumerate(stimes_p[prev_counter_p:counter_p]):
        #    plt.plot(t,i*np.ones(len(t)),'o')
        #plt.show()

        '''
        for i,cl in enumerate(np.unique(pc[pc!=-1])):
            if np.mean(np.std(pf_raw[pc==cl],axis=0))<np.mean(np.std(tf_raw[pc==cl],axis=0)):
                mean_snip = np.mean(pf_raw[pc==cl],axis=0).reshape(32,-1)
            else:
                mean_snip = np.mean(tf_raw[pc==cl],axis=0).reshape(32,-1)
            plt.figure()
            plot_snippet(mean_snip,color=cmap(i))
            plt.title(cl)
        
        plt.show()
        '''


    # check for spiketrains which are near/double??


    return mean_peak_eods, times_p, stimes_p, freqs, counter_p


def get_clusters(master_filepath, slave_filepath, save_path, load_buffer=15,save_buffer=60,grid_shape=(4,8), 
    starttime=0, endtime=60*60*48, peak_detection_threshold=0.001, peakwidth=0.0005, track_length=10, 
    spatial_feature_error_threshold=0.25, temporal_feature_error_threshold=0.025, 
    track_length_t = 10, debug_vars=[], save_artefacts=False, save_wavefish=False,mode=''):
    
    # make the cutwidth fixed based on the eod of the fish of interest.
    
    
    '''
    Get EOD clusters

    PARAMETERS:
        master_filepath : string
            (relative) path to grid data of the master DAQ
        slave_filepath  : string
            (relative) path to grid data of the slave DAQ
        save_path        : string
            (relative) path for saving output

        starttime       : int (seconds)
            Define at which point in the grid recording you want 
            to start clustering. None = start at beginning
        endtime         : int (seconds)
            Define at which point in the grid recording you want
            to end the clustering. None = stop at end of recording.

        load_buffer     : int (seconds)
            Amount of seconds of data to load per iteration. 
            Choose a high number if you have a lot of memory
        save_buffer     : int (seconds)
            Amount of seconds between saving checkpoints

        grid_shape      : tuple shape=(2,)
            Shape of electrode grid e.g. (x,y)
        peakwidth       : float (s)
            Define peakwidth of species of interest

        peak_detection_threshold        : float (V)
            Define threshold for peak detection
        
        spatial_feature_error_threshold     : float
            error (percentage wise) of the regression fit.            
        temporal_feature_error_threshold    : float
            error (percentage or absolute??) of the temporal features

        track_length    : int
            Define number of points to 'remember' when clustering
            high number for robustness, low number for flexibility
        track_lenth_t   : float (seconds)
            Define how long the algorithm should wait before removing a cluster

        debug_vars      : list of strings
            Define which variables you would like to save for debugging purposes
            list to choose from: 'spatial_features', 'spatial_patterns', 'temporal_features', 
            'temporal_patterns', 'regression_fit' #TODO: fill list with more options
        save_artefacts  : boolean
            set to TRUE to save the peak detections that are likely artefacts. 
            for debugging purposes only.
        save_wavefish   : boolean
            set to TRUE to save the peak detections that are likely wavefish. 
            for debugging purposes only.
    
    '''
    
    maxlabel = 0
    dd={}
    interp_freq = 200000
    f_width = int(0.00075*interp_freq)
    
    skipplot = []
    # check if path for previous path exists, if it does, use those clusters and empty them.
    print("looking for this:")
    print('%scache_%i.pkl'%(save_path,int(starttime/60-1)))
    try: 
        loaded_data = np.load('%scache_%i.npz'%(save_path,int(starttime/60-1)), allow_pickle=True)
        print('loaded old clusters from %s%i.npz'%(save_path,int(starttime/60-1)))
        prev_clusters = loaded_data['c']
        prev_times_p = loaded_data['t']
        prev_stimes_p = loaded_data['st'].tolist()
        prev_mean_peak_eods = loaded_data['eods']   
        cluster_counter = loaded_data['cc'] 
        prev_freqs = loaded_data['pf']
        prev_eel_times = loaded_data['pet'].tolist()

        starttimes = np.arange(starttime+10,endtime,5)
        endtimes = np.append(starttimes[1:],[endtime])+0.75
        nt = 10
        data_loaded = True
   
    except:   
        print('initiating new clusters')
        # load data in steps of n seconds
        # first ever should be 15 s, after that only load 5s at a time.
        starttimes = np.append(np.array([starttime,starttime+15]),np.arange(starttime+20,endtime,5))
        endtimes = np.append(starttimes[1:],[endtime])+0.75
        prev_clusters = []
        prev_times_p = []
        prev_stimes_p = []
        prev_mean_peak_eods = []
        prev_freqs = []
        prev_eel_times = []
        nt=0
        cluster_counter = 0
        data_loaded = False

    # max estimate for nr of eods in 1 minute?
    # max 10 fish, max eodf of 120
    all_clusters = np.ones((10*120*60))*-1
    all_ts = np.zeros((10*120*60))
    all_positions = np.ones((10*120*60,2))*-1
    a_counter = 0

    eel_times = prev_eel_times

    # save data in steps of n minutes
    for i_block, (starttime, endtime) in enumerate(zip(starttimes,endtimes)):
        
        # free memory??
        # I think if I do this, I should have enough memory to load data again?
        o_data = []
        data = []

        # load data and find potential eod times
        x, o_data, dt = load_channels(master_filepath,slave_filepath,starttime,endtime)
        
        if len(x)==0:
            break
        
        x_peaks, x_troughs, eod_hights, eod_widths, channels, data, samplerate, int_f = extract_eod_times(np.copy(o_data),peak_detection_threshold,peakwidth/dt,1/dt,interp_freq)
        dt = 1/samplerate

        # extract eel times
        eel_times.extend(extract_eel_times(o_data)*int_f+starttime*samplerate)

        mean_peak_eods = np.zeros((15*4*10,32*f_width))
        if len(prev_mean_peak_eods)>0:
            mean_peak_eods[:len(prev_mean_peak_eods)] = prev_mean_peak_eods
        
        mean_trough_eods = np.zeros((15*4*10,32*f_width))

        times_p = np.zeros((15*4*10)) # max nr of expected clusters is 10.
        freqs = np.zeros((15*4*10))
        if len(prev_times_p)>0:
            times_p[:len(prev_times_p)] = prev_times_p
            freqs[:len(prev_freqs)] = prev_freqs

        times_t = np.zeros((15*4*10))

        stimes_p = prev_stimes_p
        stimes_t = []

        ap = []
        at = []
        ap_n = []
        at_n = []
        tt = []
        maxchans = []

        skip = 0
        
        counter_p = len(prev_mean_peak_eods)
        counter_t = 0
        b_count = 1

        for i_peak, (eod_time, eod_width, eod_peak, eod_trough) in enumerate(zip((x_peaks+x_troughs)/2, eod_widths, x_peaks, x_troughs)): 

            if skip > 0:
                skip -= 1
                continue

            h_pattern = np.zeros(32)
            w_pattern = np.zeros(32)
            
            # check out surrounding peaks, troughs, and hights.
            # but I should somehow center around the highest peak?
            sl = (((x_peaks>(eod_time-eod_width)) & (x_peaks<(eod_time+eod_width)) & (x_troughs>(eod_time-eod_width*3)) & (x_troughs<(eod_time+eod_width*3))) | ((x_troughs>(eod_time-eod_width)) & (x_troughs<(eod_time+eod_width)) & (x_peaks>(eod_time-eod_width*3)) & (x_peaks<(eod_time+eod_width*3))))
  
            cp = x_peaks[sl]
            ct = x_troughs[sl]
            cc = channels[sl]
            ch = eod_hights[sl]
            cw = eod_widths[sl]

            skip = len(cc)
            
            #if len(ch)>1 and np.min(ch)/np.max(ch) > 0.25:
            #    continue
                
            h_pattern[cc.astype('int')] = ch
            w_pattern[cc.astype('int')] = cw

            maxchan = np.argmax(h_pattern)
            width = f_width/2

            p_eods = np.zeros((32,f_width))
            t_eods = np.zeros((32,f_width))

            # center around biggest pt pair.
            p_eods = data[:, int(cp[np.argmax(ch)]-width):int(cp[np.argmax(ch)]+width)]
            p_eods[np.isnan(p_eods)] = 0 # set nan to zero as broken elctrodes give NaN
            t_eods = data[:, int(ct[np.argmax(ch)]-width):int(ct[np.argmax(ch)]+width)]  
            t_eods[np.isnan(t_eods)] = 0

            # check if it is centered around the highest peak in the snip.
            # find channel with highest ch?
            if np.max(ch)<np.max(np.abs(p_eods)) or np.sum(p_eods)==0:
                skip=0
                continue

            #artefact_mask = detect_artefacts(p_eods.T,dt)        
            #wavefish_mask = detect_wavefish(p_eods.T)
            #p_eods[np.invert((artefact_mask).astype('bool'))] = 0

            if np.sum(p_eods) == 0:
                continue

            mean_eod = p_eods[np.argmax(np.var(p_eods,axis=1))]
            mean_eod = mean_eod-np.mean(mean_eod)
            snip_peaks, snip_troughs = detect_peaks(mean_eod,np.std(mean_eod))

            cut_fft = int(len(np.fft.fft(mean_eod))/2)
            low_frequency_ratio = np.sum(np.abs(np.fft.fft(mean_eod))[:int(cut_fft/(2*int_f))])/np.sum(np.abs(np.fft.fft(mean_eod))[:int(cut_fft)])           

            if ((len(snip_peaks)+len(snip_troughs)) >= 5) or (low_frequency_ratio<0.7):
                continue

            ap.append(p_eods.flatten())
            at.append(t_eods.flatten())
            tt.append(eod_time)
            maxchans.append(cc[np.argmax(ch)])

            # normalize
            p_n, _ = lp.subtract_slope(p_eods,h_pattern.T)
            p_n = (p_n.T - np.mean(p_n,axis=1)).T
            t_n, _ = lp.subtract_slope(t_eods,h_pattern.T)
            t_n = (t_n.T - np.mean(t_n,axis=1)).T
            ap_n.append(p_n.flatten())
            at_n.append(t_n.flatten())
            
        # add last block
        if len(ap)<10:
            print('NO FISH')
        
        else:
            mean_peak_eods, times_p, stimes_p, freqs, counter_p = analyse_window(ap,at,ap_n,at_n,np.array(tt),maxchans,starttime,endtime,samplerate,int_f,mean_peak_eods,times_p,stimes_p,freqs,counter_p,o_data,data,x_peaks,x_troughs,channels,eod_hights,eod_width,f_width)
            # do the merge step here...?

        # delete zeros
        times_p = times_p[np.sum(np.abs(mean_peak_eods),axis=1)>0]
        freqs = freqs[np.sum(np.abs(mean_peak_eods),axis=1)>0]
        mean_peak_eods = mean_peak_eods[np.sum(np.abs(mean_peak_eods),axis=1)>0]

        # load previous clusters
        clusters = np.ones(len(mean_peak_eods))*-1
        clusters[:len(prev_clusters)] = prev_clusters

        if len(mean_peak_eods)>0:

            #pdm = distance_matrix(np.var(mean_peak_eods.reshape(mean_peak_eods.shape[0],32,-1),axis=2),np.var(mean_peak_eods.reshape(mean_peak_eods.shape[0],32,-1),axis=2)).flatten()
            
            # modify patterns so that smaller values are more prominent (inverse relation of r**3)
            s_patterns = np.std(mean_peak_eods.reshape(mean_peak_eods.shape[0],32,-1),axis=2)**(1/3)
            
            # set small values to min values (ReLu). determine these by peak thresh?
            #for i in range(32):
            #    s_patterns[s_patterns[:,i]<np.log(np.std(o_data[:,i])),i] = np.log(np.std(o_data[:,i]))

            # spatial distance matrix
            dm_s = distance_matrix(s_patterns,s_patterns).flatten()
            temp = dm_s.argsort()
            raw_score_s = np.empty_like(temp)
            raw_score_s[temp] = np.arange(len(dm_s))
            
            # compute frequency component
            dm_f = distance_matrix((freqs).reshape(-1,1),(freqs).reshape(-1,1)).flatten()
            temp = dm_f.argsort()
            raw_score_f = np.empty_like(temp)
            raw_score_f[temp] = np.arange(len(dm_f))

            # set all f_dists above 5 to the same value
            #raw_score_f[dm_f>5] = np.max(raw_score_f)
            
            # add distance matrices before sorting
            pdm = (2/3)*raw_score_s + (1/3)*raw_score_f

            idxs = np.argsort(pdm)
            pdm_flat = pdm[idxs][len(mean_peak_eods):][::2]
            idxs = idxs[len(mean_peak_eods):][::2]

            for i in idxs:

                x = int(np.mod(i,len(mean_peak_eods)))
                y = int(np.floor(i/len(mean_peak_eods)))

                t1 = times_p[x]
                t2 = times_p[y]

                if (t1 != t2) and (x is not y):
                    # is either already in a cluster? if no, connect.

                    if clusters[x] == -1 and clusters[y] == -1:
                        # check if t for either of the connecting eods already exists?
                        clusters[x] = cluster_counter
                        clusters[y] = cluster_counter
                        cluster_counter = cluster_counter + 1

                    elif clusters[x] == clusters[y]:
                        continue

                    elif (clusters[x] >=0) and (clusters[y] >= 0):

                        t_existing_x = times_p[clusters==clusters[x]]
                        t_existing_y = times_p[clusters==clusters[y]]

                        if len(times_p[(clusters==clusters[x])|(clusters==clusters[y])]) == len(np.unique(times_p[(clusters==clusters[x])|(clusters==clusters[y])])):                                                    
                        
                            clusters[clusters==clusters[x]] = min(clusters[x],clusters[y])
                            clusters[clusters==clusters[y]] = min(clusters[x],clusters[y])
                        
                    elif clusters[y] >=0:
                        t_existing = times_p[clusters==clusters[y]]
                        
                        if times_p[x] not in t_existing:
                            clusters[x] = clusters[y]

                    elif clusters[x] >=0:
                        t_existing = times_p[clusters==clusters[x]]        
     
                        # does the existing cluster already have a connection at this timepoint?
                        if times_p[y] not in t_existing:
                            clusters[y] = clusters[x]
                    '''
                    plt.figure()
                    plt.subplot(121)
                    plt.imshow(s_patterns[x].reshape(4,8))
                    plt.subplot(122)
                    plt.imshow(s_patterns[y].reshape(4,8))
                    plt.show()

                    if not clusters[x]==clusters[y]:
                        plt.figure()
                        plt.plot(times_p,freqs,'o',c='grey')
                        print(times_p[[x,y]])
                        print(freqs[[x,y]])
                        plt.plot(times_p[[x,y]],freqs[[x,y]],linestyle='--',c='k',linewidth=2)
                        for i,c in enumerate(np.unique(clusters[clusters!=-1])):
                            plt.plot(times_p[clusters==c],freqs[clusters==c],c=cmap(i))
                        plt.plot(times_p[[x,y]],freqs[[x,y]],linestyle='--',c='k',linewidth=2)

                        plt.title(clusters[y]==clusters[x])

                        plt.figure()
                        plt.subplot(121)
                        plt.imshow(s_patterns[x].reshape(4,8))
                        plt.subplot(122)
                        plt.imshow(s_patterns[y].reshape(4,8))
                        plt.show()
                    '''
                    
        '''
        # check if there are overlapping EOD times
        for i,c in enumerate(np.unique(clusters[clusters!=-1])):
            st1 = np.concatenate(([stimes_p[i] for i in (np.where(clusters==c)[0]).astype('int')]))
            for i,c2 in enumerate(np.unique(clusters[clusters!=-1])):
                st2 = np.concatenate(([stimes_p[i] for i in (np.where(clusters==c2)[0]).astype('int')]))
                if c!=c2 and any(np.isin(st1,st2)):
                    print('merge by spiketimes')
                    print(c)
                    print(c2)
        
        plt.figure()
        plt.plot(times_p,freqs,'o',c='k',alpha=0.5)
        plt.plot(times_p[[x,y]],freqs[[x,y]],linestyle='--',c='k')

        for i,c in enumerate(np.unique(clusters[clusters!=-1])):
            plt.plot(times_p[clusters==c],freqs[clusters==c],c=cmap(i))
        plt.show()
        
        plt.figure()
        for i,c in enumerate(np.unique(clusters[clusters!=-1])):
            st = np.concatenate(([stimes_p[i] for i in (np.where(clusters==c)[0]).astype('int')]))
            plt.plot(st,c*np.ones(len(st)),'o')
        plt.show()
        plt.figure()
        for i,c in enumerate(np.unique(clusters[clusters!=-1])):
            st = np.concatenate(([stimes_p[i] for i in (np.where(clusters==c)[0]).astype('int')]))
            plt.plot(np.unique(st)[1:],1/np.diff(np.unique(st)),'o')
        plt.show()
        '''
  
        # save middle
        # I would save clusters, timepoints, and location?

        for c in np.unique(clusters[clusters!=-1]):
            times = []
            if not data_loaded:
                if len(np.where((clusters==c)&(times_p<10+starttime))[0])>0:
                    #-> this is a list of lists
                    # save the first 10 seconds as there is no previous data
                    # stack all time vectors for this cluster
                    times = np.concatenate(([stimes_p[i] for i in (np.where((clusters==c))[0]).astype('int')]))
                    times = times[times<starttime+10]
            else:
                if len(np.where((clusters==c)&(times_p<starttime)&(times_p>=starttime-5))[0])>0:
                    times = np.concatenate(([stimes_p[i] for i in (np.where((clusters==c))[0]).astype('int')]))
                    times = times[(times>starttime-5)&(times<starttime)]

            all_clusters[a_counter:a_counter+len(np.unique(times))] = np.ones(len(np.unique(times)))*c
            all_ts[a_counter:a_counter+len(np.unique(times))] = np.sort(np.unique(times))

            ccounter=0
            for i,t in zip((np.where((clusters==c))[0]).astype('int'),times_p[clusters==c]):
                all_positions[a_counter+ccounter:a_counter+ccounter+len(stimes_p[i])] = get_position(np.var(mean_peak_eods[(clusters==c)&(times_p==t)].reshape(32,-1),axis=1))
                ccounter = ccounter+len(stimes_p[i])
            a_counter = a_counter+len(np.unique(times))


        if not data_loaded:
            # keep clusters of 5-9 and map times to 0-4
            prev_clusters = clusters[(times_p>=5+starttime) & (times_p<10+starttime)]

            # keep features+spiketimes of 5-14
            prev_mean_peak_eods = mean_peak_eods[(times_p>=5+starttime)]
            prev_stimes_p = [stimes_p[i] for i in np.where(times_p>=5+starttime)[0].astype('int')]
            prev_times_p = times_p[(times_p>=5+starttime)] # map times to 0-9
            prev_freqs = freqs[(times_p>=5+starttime)]

        else:
            # keep clusters of 5-9 and map times to 0-4
            prev_clusters = clusters[(times_p>=starttime-5) & (times_p<starttime)]

            # keep features+spiketimes of 5-14
            prev_mean_peak_eods = mean_peak_eods[(times_p>=starttime-5)]
            prev_stimes_p = [stimes_p[i] for i in np.where(times_p>=starttime-5)[0].astype('int')]
            prev_times_p = times_p[(times_p>=starttime-5)] # map times to 0-9
            prev_freqs = freqs[(times_p>=starttime-5)]

        data_loaded = True

        # move nt
        nt = 10

        '''
        print(all_clusters)
        for i,c in enumerate(np.unique(all_clusters[all_clusters!=-1])):
            plt.plot(all_ts[all_clusters==c][1:],1/np.diff(all_ts[all_clusters==c]),'o-')
        plt.show()
        '''

        if (endtime == endtimes[-1]) or ((endtime>starttimes[0]+5.75) and (endtime%60 == 5.75)):
            print('save :), t=%f'%endtime)
            # save numpy arrays:
            # clusters, positions and times.

            cur_et = np.array(eel_times)
            prev_eel_times = cur_et[cur_et>((endtime-5.75)*samplerate)]
            save_eel_times = cur_et[cur_et<=((endtime-5.75)*samplerate)]

            np.savez('%s%i.npz'%(save_path,int(endtime/60)-1), t=all_ts[:a_counter], c=all_clusters[:a_counter], p=all_positions[:a_counter],et=save_eel_times/samplerate)
            np.savez('%scache_%i.npz'%(save_path,int(endtime/60)-1), c=prev_clusters, cc=cluster_counter, eods=prev_mean_peak_eods, st=prev_stimes_p, t=prev_times_p,pf=prev_freqs,pet=prev_eel_times)

            all_clusters = np.ones((10*120*60))*-1
            all_ts = np.zeros((10*120*60))
            all_positions = np.ones((10*120*60,2))*-1
            a_counter = 0
            eel_times = []

    cur_et = np.array(eel_times)
    prev_eel_times = cur_et[cur_et>((endtime-5)*samplerate)]
    save_eel_times = cur_et[cur_et<=((endtime-5)*samplerate)]

    np.savez('%s%i.npz'%(save_path,int(endtime/60)-1), t=all_ts[:a_counter]/samplerate, c=all_clusters[:a_counter], p=all_positions[:a_counter],et=save_eel_times/samplerate)
    np.savez('%scache_%i.npz'%(save_path,int(endtime/60)-1), c=prev_clusters, cc=cluster_counter, eods=prev_mean_peak_eods, st=prev_stimes_p, t=prev_times_p,pf=prev_freqs,pet=prev_eel_times)

    return 0

if __name__ == '__main__':

    print('NEW VERSION :) !')

    m_files = ['data/master/2019-10-17-12_36/', 'data/master/2019-10-17-19_48',  'data/master/2019-10-18-09_52',  'data/master/2019-10-19-08_39',  'data/master/2019-10-20-08_30']
    s_files = ['data/slave/2019-10-17-13_35/', 'data/slave/2019-10-17-19_48',  'data/slave/2019-10-18-09_44',  'data/slave/2019-10-19-08_21',  'data/slave/2019-10-20-08_30']
    
    # command line arguments:
    #parser = argparse.ArgumentParser(description='Analyze EOD waveforms of weakly electric fish.')    
    #                    help='name of a file with time series data of an EOD recording')
    #parser.add_argument('slave_files', nargs=1, default='', type=str,
    #                    help='name of a file with time series data of an EOD recording')
    #args = parser.parse_args()

    for master_file,slave_file in zip(m_files,s_files):
        if master_file[-1] == '/':
            save_folder = 'data/t3/' + master_file.split('/')[-2] + '/'
        else:
            save_folder = 'data/t3/' + master_file.split('/')[-1] + '/'

        starttime = 0
        
        if os.path.exists(save_folder):
            # check the last file that was saved and continue analysis there.
            starttime = len([name for name in os.listdir(save_folder) if 'cache' in name])

        else:
            # make dir.
            os.mkdir(save_folder)


        #starttime = 74#71 #74

        # maybe first check the last file that was output to the save folder and continue analysis from there?
        # 
        get_clusters(master_file,slave_file,save_folder,starttime=starttime*60,debug_vars=['c_sf','h_pattern','m_eod','s_err','t_err'])

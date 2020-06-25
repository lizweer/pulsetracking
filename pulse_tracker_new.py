import thunderfish.dataloader as dl
import thunderfish.eventdetection as ed
import pulse_tracker_helper_new as pth
import thunderfish.lizpulses as lp

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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pickle
import re

import copy

import argparse


# save all eods. then compare across the same channel.


class pd(object):
    """
    Object generated to store all data extracted from the recording.
    
    Input:
        master_filepath :   string
            path to input data of master DAQ
        slave_filepath  :   string
            path to input data of slave DAQ
        dt              :   float, 1/seconds
            1/sampling_rate of the raw data
        starttime       :   float, seconds
            starttime of analysis wrt beginning of recording
        endtime         :   float, seconds
            endtime of analysis wrt beginning of recording


        clusters         :   list of cluster_objects
            each instance is one cluster with corresponding features and spiketimes
        deleted_clusters :   list of cluster_objects
            list of clusters that were deleted from the analysis as they have been idle for too long
        potential_eod    :   cluster_object
            all features and recording times of potential eods that were not included in the 
            clustering yet because they are too sparse
        artefacts        :   cluster_object
            all features and recording times of artefacts in one cluster object
        eel              :   list of floats
            specifying the spiketimes of an eel
    """

    def __init__(self,starttime,endtime,dt,clusters,master_filepath,slave_filepath,eel):
        self.master_filepath = master_filepath
        self.slave_filepath = slave_filepath
        self.dt = dt
        self.starttime = starttime
        self.endtime = endtime

        self.clusters = clusters
        self.eel = eel

    # return so I can save as a dict? or save within this?


class cluster_object(object):
    """
    Object generated to store and edit clusters
    """
    
    def __init__(self,label,track_length,grid_shape=(4,8),temporal_featnum=6,max_peak_count=60*120,debug_vals=[]):
        
        """
        Initialize cluster object. create empty lists for feature tracking.
        
        Input:
            label            : int
                cluster label, should be unique for each cluster
            track_length     : int
                track_length of features. Use long lenght for robustness, short length for flexibility
            temporal_featnum : int
                number of temporal features, dependent on the chosen feature selection
            spatial_featnum  : int
                number of spatial features, dependent on the chosen feature selection
            peaknum          : int
                space to allocate for the peak instances. Use an estimate of the upper 
                bound of expected peaks within the analysis time window 
                (e.g. for 60 seconds and an upper bound of a 1kHz firing rate, peaknum = 60000)
                preferred over dynamical allocation to improve speed.

            *args         : list of strings
                identify what you would like to track (for debugging only, is slow.)
        """

        # cluster label
        self.label = label
        self.track_length = track_length
        self.grid_shape = grid_shape

        # this is updated to keep track of the peak_idx
        self.peak_count = 0

        # storage buffers for features. Each feature is only tracked for track_length.        
        self.f_spatial = np.ones((track_length, grid_shape[0]*grid_shape[1]))*99
        self.f_temporal = np.ones((track_length, 2, 32, temporal_featnum))*99
        self.f_ts = np.zeros(track_length)
        self.f_connecting_clusters = np.zeros(track_length)


        # storage buffers for EOD instances (time, location)
        self.ts = np.zeros(max_peak_count)
        self.positions = np.ones((max_peak_count,2))*99

        # storage buffers for things you want to track (debugging)
        self.debug = {v:[] for v in debug_vals}
    
    def update(self,t,spatial_feature,temp_feature,connecting_cluster,debug_dict={}):
        """
        Update cluster object with the current EOD.

        Input:
            t               : float, seconds
                EOD time
            position        : numpy array shape=(2)
                position estimate
            spatial_feature : numpy array shape=(self.spatial_features.shape[1])
                spatial EOD features
            temp_feature    : numpy array shape=(self.temp_features.shape[1])
                temporal EOD features
            **kwargs        : **dict
                dict of keywords + values of things you are tracking
        """
        
        self.ts[self.peak_count] = t
        self.positions[self.peak_count] = get_position(spatial_feature,self.grid_shape)

        self.f_spatial[int(self.peak_count%self.track_length)] = spatial_feature 
        self.f_temporal[int(self.peak_count%self.track_length)] = temp_feature
        self.f_ts[int(self.peak_count%self.track_length)] = t
        self.f_connecting_clusters[int(self.peak_count%self.track_length)] = connecting_cluster

        for key in debug_dict:
            if key in self.debug:
                self.debug[key].append(debug_dict[key])
            else:
                print('WARNING: debug attribute %s not initialized, ignoring this.'%(key))

        self.peak_count = self.peak_count + 1
    
    def empty_features(self):
        '''
        Empty cluster object.
        This function is used after writing the cluster to a file. 
        All values are emptied but the cluster features are kept, to 
        continue tracking for this cluster
        '''

        for attribute in dir(self):
            if re.search('^f_', attribute):
                # shift all features to make sure least recent features will be overwritten
                # the features are not removed as they are needed for continuous tracking
                setattr(self,attribute,np.roll(getattr(self,attribute),-(self.peak_count%self.track_length),axis=0))

        for key in self.debug:
            self.debug[key] = []

        self.ts = np.zeros(self.ts.shape)
        self.positions = np.ones(self.positions.shape)*99
        self.peak_count = 0

    def is_idle(self, current_t, track_length_t):
        if current_t - np.max(self.f_ts) > track_length_t:
            return True
        else:
            return False


        # I could state here whether the cluster is idle or not so it could be removed.

    def recently_spiked(self, current_t, peakwidth):
        if current_t - np.max(self.f_ts) < peakwidth*2:
            return True
        else:
            return False

    def delete(self,idxs):
        '''
        Delete instances from cluster object.
        This function is used to delete instances when some 
        instances have been wrongly assigned to this cluster.
        '''

        #for attribute in dir(self):
        #    if 'f_' in attribute:
        #        setattr(self,attribute,np.pad(np.delete(getattr(self,attribute),
        #        np.remainder(idxs,self.track_length)),(0,len(idxs)),'constant',constant_values=(0,-99)))

        self.f_spatial[idxs] = np.ones(self.f_spatial[idxs].shape)*99
        self.f_temporal[idxs] = np.ones(self.f_temporal[idxs].shape)*99
        self.f_ts[idxs] = np.ones(self.f_ts[idxs].shape)*-99
        self.f_connecting_clusters[idxs] = 0

        #for key in self.debug:
        #    self.debug[key].pop(idxs)
        #    self.debug[key].extend([0]*len(idxs))

        #self.ts = np.pad(np.delete(ts,idxs),(0,len(idxs)),'constant')
        #self.positions = np.pad(np.delete(positions,idxs,axis=0),(0,len(idxs)),'constant',constant_values=(0,99))
        #self.peak_count = self.peak_count - len(idxs)

    def predict_next_t(self,n=4):
        '''
        Predict next EOD time based on n last EODs
        '''
        if np.count_nonzero(self.f_ts) > n:
            return np.max(self.f_ts) + self.get_isi(n)
        else:
            return 0
    
    def get_isi(self,n):
        '''
        Get ISI of last n EODs
        '''
        return np.diff(np.sort(self.f_ts[np.nonzero(self.f_ts)])[-n:])

    def get_spike_count(self,cur_t,t=1):
        '''
        Get amount of spikes in last n seconds.
        using f_ts for this might be inaccurate as it only saves the last n features.
        using ts however also causes problems if it has been loaded from a previous file.

        if this function is only used for potential eods there is no problem. find a way to generalize it.
        '''
        print(self.f_ts)
        spike_count = len(self.f_ts[self.f_ts>(cur_t-t)])
        return spike_count

    def save(self):
        #TODO return dict to save to JSON file.
        return 0

    def get_last_feature(self,attr):
        return getattr(self,attr)[int(self.peak_count%self.track_length - 1)]

    def revert(self,t):
        # check out how much you need to go back
        print('reverting')
        rt = np.count_nonzero(self.ts>t)
        print(rt)
        print(self.ts)
        print(t)

        ts_b = np.zeros(self.ts.shape)
        pos_b = np.ones(self.positions.shape)*99

        self.peak_count = self.peak_count - rt
        ts_b[:self.peak_count] = self.ts[:self.peak_count]
        pos_b[:self.peak_count] = self.positions[:self.peak_count]

        self.ts = ts_b
        self.positions = pos_b

        # for now I dont want to revert the features because I will update them anyway.
        # return all deleted values
        i = np.mod(np.arange(self.peak_count,self.peak_count+rt),self.track_length)
        print(i)

        return self.f_ts.take(i,0), self.f_spatial.take(i,0), self.f_temporal.take(i,0)



def get_position(spatial_pattern,grid_shape=(4,8),n_elec=2):
    mask_idx = spatial_pattern.argsort()[-n_elec:][::-1]
    mask = np.zeros(len(spatial_pattern))
    mask[mask_idx] = 1

    return np.asarray(center_of_mass((spatial_pattern*mask).reshape(grid_shape)))

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
    
    data = np.concatenate([y,y2],axis=1)
    print('data loaded')
    return x,data,dt

def extract_eod_times(data, thresh, min_peakwidth, max_peakwidth, samplerate, interp_freq, win_size = 0.0005, n_stds = 1000, threshold_factor=6):

    print('extracting times')
    print(max_peakwidth)
    print(min_peakwidth)

    all_int_data = np.array([])

    channels = []
    x_peaks = []
    x_troughs = []
    eod_hights = []
    eod_widths = []

    for i in range(data.shape[1]):
        y = data[:,i]
        x_peak, x_trough, eod_hight, eod_width, int_samplerate, int_data, _ = lp.extract_eod_times(y,samplerate,peakwidth=max_peakwidth,min_peakwidth=min_peakwidth,cutwidth=0.0002,interp_freq=interp_freq)
        plt.show()
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

    return x_peaks[a], x_troughs[a], eod_hights[a], eod_widths[a], channels[a], all_int_data, int_samplerate


### functions for extracting features
def real_feat(y,dt,slope_num=4):

    w_num = slope_num - 2

    # for widths I would take the time instead of indices

    argmaxmin = extract_maxmin(y)
    maxmin = y[argmaxmin]
    slopes = np.diff(maxmin)
    widths = np.diff(argmaxmin)
    
    best_i = 0
    slope_sum = 0

    all_features = -np.ones((3,slope_num+w_num))

    if len(slopes)>=slope_num:

        all_features[:,:] = 0

        for i in range(len(argmaxmin)-slope_num):
            if np.sum(np.abs(slopes[i:i+slope_num])) > slope_sum:
                best_i = i
                slope_sum = np.sum(np.abs(slopes[i:i+slope_num]))

        for j,i in enumerate([-1,0,1]):
            s_idxs = np.arange(max(0,best_i+i),min(best_i+i+slope_num,len(slopes)))
            w_idxs = np.arange(max(1,best_i+i+1),min(best_i+i+slope_num-1,len(slopes)-1))

            if i == -1:
                all_features[j,slope_num-len(s_idxs):slope_num] = slopes[s_idxs]
                all_features[j,-len(w_idxs):] = widths[w_idxs]*dt            
            else:
                all_features[j,:len(s_idxs)] = slopes[s_idxs]
                all_features[j,-w_num:slope_num + len(w_idxs)] = widths[w_idxs]*dt

    return all_features

def extract_maxmin(y):

    ig = argrelextrema(y, np.greater)
    il = argrelextrema(y, np.less)
    
    argmaxmin = np.sort(np.append(ig,il)).astype('int')
    argmaxmin = np.append(np.append(0,argmaxmin),len(y)-1)

    return argmaxmin

def single_eod(pattern, grid_shape=(4,8)):
    # check if all blocks are connected. Each block should be connected to at least one other block.
    # for each x, check if connected to one other x

    # compute two max channels
    maxchan = np.argsort(pattern)[:2]
    
    # then check if they connect.
    xpos = np.mod(maxchan,grid_shape[1])
    ypos = np.floor(maxchan/grid_shape[1])

    if (np.abs(np.diff(xpos))<=1) and (np.abs(np.diff(ypos))<=1):
        return True
    else:
        return False


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
def extract_eel_times(data):
    print('extracting eeltimes')
    all_eel_times = []
    for i in range(data.shape[1]):
        y = data[:,i]
        et = zero_runs(np.diff(y))
        aplist = np.mean(et[et[:,1] - et[:,0] > 5],axis=1) 
        if len(aplist) > 0:
            all_eel_times = np.append(all_eel_times,aplist)

    if len(all_eel_times) > 0:
        all_eel_times = np.unique(all_eel_times)

    return np.round(all_eel_times)

def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def plot_snippet(eods,x,p,t,c):
    for i in range(eods.shape[0]):
        plt.subplot(4,8,i+1)
        plt.plot(x,eods[i])
        if i in c:
            try:
                plt.plot(p[c==i],eods[i,(p[c==i]-x[0]).astype('int')],'x')
                plt.plot(t[c==i],eods[i,(t[c==i]-x[0]).astype('int')],'x')
            except:
                plt.plot(x,eods[i],c='r')
                pass
        plt.ylim([np.min(eods),np.max(eods)])
        plt.axis('off')

def to_coordinates(i,shape):
    return [np.floor(i/shape[1]),i%shape[1]]

def get_relevant_cluster_keys(all_clusters,c_time,track_length_t,peakwidth):
    relevant_keys = []
    recent_spike = False
    # always keep all clusters but only get relevant keys?
    # then when I want to save the clusters later, I dump the irrelevant ones as deleted_clusters.

    # I only want to use recent clusters, 
    # but somehow I also need to save all clusters for later
    for ckey, cval in all_clusters.items():
        # delete clusters that are too far away (10 sec.)

        # if its not found, and a cluster that has recently spiked was not analyzed, dont add as potential eod.
        if not cval.is_idle(c_time, track_length_t):
            if not cval.recently_spiked(c_time, peakwidth):
                relevant_keys.append(ckey)     
            else:
                recent_spike = True

                
    return np.array(relevant_keys), recent_spike

def get_cluster_candidates(clusters, cluster_keys, spatial_pattern, spatial_feature_error_threshold,noise_bound=1):
    print('cluster keys')
    print(cluster_keys)
    print(len(cluster_keys[cluster_keys!='potential_eod']))
    if len(cluster_keys)>0 and len(cluster_keys[cluster_keys!='potential_eod']) > 0:
        # get spatial features from clusters
        #c_feats = np.array([])
        #print(cluster_keys)
        c_feats = np.stack([clusters[key].f_spatial for key in cluster_keys])

        # alternative way by subtracting
        c_diff = -c_feats + spatial_pattern
        c_diff[c_diff > 0] = 0 #only negative values can be errors

        # get the error as a fraction of the cluster variance
        c_error = np.sum(np.abs(c_diff),axis=2)/np.sum(c_feats,axis=2)

        cluster_candidate_keys = cluster_keys[np.count_nonzero(c_error<spatial_feature_error_threshold,axis=1)>=noise_bound]

        '''
        # get differences between current spatial pattern and saved spatial patterns for clusters
        c_diff = np.linalg.norm((c_feats - spatial_pattern)**2, axis=2)
        amin = np.argmin(c_diff,axis=1)

        # for each cluster, use one representative spatial feature for linear regression
        x_glm = np.stack([c_feats[i,amin[i]] for i in range(c_feats.shape[0])])

        print(x_glm.shape)

        # do a linear regression on these spatial patterns
        reg = lsq_linear(np.transpose(x_glm),spatial_pattern,(0,1.5))

        # what if I do linear regression on all patterns and use the sum of the ones that are used as result??
        c_error = np.abs(reg.x - 1)

        print('spatial error')
        print(c_error)

        print('error of fit')
        print(reg.cost)

        # select candidate clusters based on linear regression error.
        cluster_candidate_keys = cluster_keys[c_error < spatial_feature_error_threshold]

        #if len(cluster_candidate_keys) == 0:
        #    plt.figure()
        #    plt.imshow(spatial_pattern.reshape(4,8))
        #    plt.colorbar()

        #    for xglm in x_glm:
        
        #        plt.figure()
        #        plt.imshow(xglm.reshape(4,8))
        #        plt.colorbar()
        #    plt.show()

        # maybe unneccesary now I dont use this anymore.
        #cc_spatial_features = get_spatial_features(cluster_candidate_keys,spatial_pattern,reg.x[c_error < spatial_feature_error_threshold],x_glm[c_error < spatial_feature_error_threshold])
        '''
        return cluster_candidate_keys, c_error
    else:
        return [],[]

# somehow check for maxchan here. so I'd already have to extract the spatial features here
def get_spatial_features(cc_keys,spatial_pattern, glm_coef,glm_x):
    
    cc_spatial_features = {}

    for i,key in enumerate(cc_keys):
        if len(cc_keys) > 1:                            
            coef = np.delete(glm_coef,i,axis=0)
            inp = np.delete(glm_x,i,axis=0)
            cc_spatial_features[key] = spatial_pattern - np.sum(np.multiply(np.transpose(inp),coef),axis=1)
        else:
            cc_spatial_features[key] = spatial_pattern

    return cc_spatial_features

def ask_the_user(spatial_pattern,eod,time):
# ask for user input.
    manual_input = True
    
    plt.subplot(121)
    plt.imshow(spatial_pattern.reshape(4,8))
    plt.subplot(122)
    plt.plot(eod)
    plt.show()
    answer = ''

    while answer not in ['y','n','d']:
        answer = input('is this a new and single EOD at time=%fs? (type y or n): '%(time))
    if answer == 'd':
        manual_input = False

    return answer, manual_input

#def plot_cluster_features()
#    f, ax = plt.subplots(2,10)

#    for i in range(10):
#        ax[].plot()


def assess_candidates(clusters,cc_keys,maxchan,data,peaks,throughs,channels,temporal_feature_error_threshold,dt,noise_bound=1):
    
    ac_keys = []
    min_dist_norm = [0]
    err = []
    
    # get temp features for the right channels
    for cc_key in cc_keys:

        compare_features = clusters[cc_key].f_temporal[:,:,maxchan,:]
        width = compare_features.shape[-1]

        if len(peaks[channels==maxchan])>1:
            continue
        cfeats = np.vstack([data[maxchan, int(peaks[channels==maxchan]-width/2):int(peaks[channels==maxchan]+width/2)],data[maxchan, int(throughs[channels==maxchan]-width/2):int(throughs[channels==maxchan]+width/2)]])
        
        compare_features = np.concatenate((compare_features,cfeats.reshape(-1,2,width)))
        print(compare_features.shape)

        keep0 = (np.sum(np.abs(np.diff(compare_features[:,0,:],axis=1)),axis=1)>0)
        keep1 = (np.sum(np.abs(np.diff(compare_features[:,1,:],axis=1)),axis=1)>0)
        print(np.sum(compare_features[:,0,:],axis=1))
        print(keep0)
        # now do pca? on what? what if I only have one channel to connect to?
        # for now just take the distance which should somehow be normalized.
        # subtract the slope and normalize the snippets
        compare_features[keep0,0,:], sr = lp.subtract_slope(compare_features[keep0,0,:])
        compare_features[keep1,1,:], sr = lp.subtract_slope(compare_features[keep1,1,:])

        compare_features[keep0,0,:] = StandardScaler().fit_transform(compare_features[keep0,0,:].T).T
        compare_features[keep1,1,:] = StandardScaler().fit_transform(compare_features[keep1,1,:].T).T

        # scale so that the absolute integral = 1.
        compare_features[keep0,0,:] = (compare_features[keep0,0,:].T/np.sum(np.abs(compare_features[keep0,0,:]),axis=1)).T
        compare_features[keep1,1,:] = (compare_features[keep1,1,:].T/np.sum(np.abs(compare_features[keep1,1,:]),axis=1)).T
        
        # now do PCA?
        #plt.figure()
        #plt.plot(compare_features[:,0,:].T)
        #plt.show()
        #plt.plot(compare_features[:,1,:].T)
        #plt.show()

        
        fsum = np.sum(compare_features[:,0,:],axis=1)

        if np.count_nonzero(~np.isnan(fsum)) > 1:

            pcs = PCA(2).fit(compare_features[~np.isnan(fsum),0,:]).transform(compare_features[~np.isnan(fsum),0,:])
            p_diff = np.linalg.norm(np.abs(pcs[:-1]-pcs[-1]),axis=1)

            pcs = PCA(2).fit(compare_features[~np.isnan(fsum),1,:]).transform(compare_features[~np.isnan(fsum),1,:])
            t_diff = np.linalg.norm(np.abs(pcs[:-1]-pcs[-1]),axis=1)

            error = np.min(np.vstack([p_diff,t_diff]),axis=1)

            if np.count_nonzero(error<temporal_feature_error_threshold) >= noise_bound:
                ac_keys.append(cc_key)
            #else:
            '''
                plt.figure()
                plt.plot(compare_features[~np.isnan(fsum),0,:].T,c='b')
                plt.plot(compare_features[~np.isnan(fsum),1,:].T,c='r')
                plt.plot(compare_features[-1,0,:],c='k')
                plt.plot(compare_features[-1,1,:],c='k')
                plt.show()  
            '''              

            err.append(np.min(error))
        else:
            err.append(1)

    print('temp errors')
    print(err)

    return np.array(ac_keys), err

def get_temp_feat(eods,channels,dt):
    a = np.zeros((len(channels),3,6))
    for i,c in enumerate(channels):
        ft = real_feat(eods[:,c],dt)
        a[i] = ft
    return a

def percent_change(a,b):
    return np.abs((a-b)/((a+b)/2))

def argmax_2d(a):
    return np.unravel_index(np.argmax(a, axis=None), a.shape)

def get_clusters(master_filepath, slave_filepath, save_path, load_buffer=10,save_buffer=60,grid_shape=(4,8), 
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
    
    # check if path for previous path exists, if it does, use those clusters and empty them.
    print("looking for this:")
    print('%s%i.pkl'%(save_path,int(starttime/60-1)))
    try: 
        co = pickle.load(open('%s%i.pkl'%(save_path,int(starttime/60-1)),'rb'))
        print('loaded old clusters from %s%i.pkl'%(save_path,int(starttime/60-1)))
        all_clusters = co.clusters

        print('loaded clusters::')
        print(all_clusters.keys())

        eel = co.eel

        print('emptying eel eods')
        eel.empty_features()

        # load relevant clusters
        all_clusters_buffer = {}

        for k,c in all_clusters.items():
            if (k=='potential_eod') or (not c.is_idle(starttime,track_length_t)):
                c.empty_features()
                if k!='potential_eod':
                    maxlabel = max(maxlabel,int(c.label))
                # add the debug vals if there are any
                c.debug = {v:[] for v in debug_vars}
                all_clusters_buffer[k] = c

        all_clusters = all_clusters_buffer

        manual_input = False     

    except:
        manual_input = True

        print('initiating new clusters')
        all_clusters = {}

        eel = cluster_object('eel',track_length)

        potential_eod = cluster_object(str(maxlabel+1),track_length,temporal_featnum=f_width,debug_vals=debug_vars)

        potential_eod.debug = {v:[] for v in debug_vars}

        all_clusters['potential_eod'] = potential_eod
   

    # load data in steps of n seconds
    starttimes = np.arange(starttime,endtime,load_buffer)
    endtimes = np.append(starttimes[1:],[endtime])

    # save data in steps of n minutes
    for i_block, (starttime, endtime) in enumerate(zip(starttimes,endtimes)):
        
        # load data and find potential eod times
        x, data, dt = load_channels(master_filepath,slave_filepath,starttime,endtime)
        cur_eel = extract_eel_times(data)*(interp_freq*dt)  
        x_peaks, x_troughs, eod_hights, eod_widths, channels, data, samplerate = extract_eod_times(data,peak_detection_threshold,f_width/interp_freq/10,f_width/interp_freq/4,1/dt,interp_freq)
        dt = 1/samplerate

        #for i, d in enumerate(data):
        #    plt.plot(d)
        #    plt.plot(wx_peaks[wchannels==i],d[wx_peaks[wchannels==i].astype('int')],'x')
        #    plt.plot(wx_troughs[wchannels==i],d[wx_troughs[wchannels==i].astype('int')],'x')
        #    plt.show()

        skip = 0

        for i_peak, (eod_time, eod_width, eod_peak, eod_trough) in enumerate(zip((x_peaks+x_troughs)/2, eod_widths, x_peaks, x_troughs)): 

            if skip > 0:
                skip = skip-1
                continue

            print(eod_time)
            print(eod_width)
            print(eod_peak)
            print(eod_trough)
            
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

            if np.min(ch)/np.max(ch) > 0.25:
                continue
                
            h_pattern[cc.astype('int')] = ch
            w_pattern[cc.astype('int')] = cw

            maxchan = np.argmax(h_pattern)
            width = f_width/2

            p_eods = np.zeros((32,f_width))
            t_eods = np.zeros((32,f_width))

            for c in np.unique(cc):
                if len(cc[cc==c]) == 1:
                    try:
                        p_eods[int(c)] = data[int(c), int(cp[cc==c]-width):int(cp[cc==c]+width)]
                        t_eods[int(c)] = data[int(c), int(ct[cc==c]-width):int(ct[cc==c]+width)]
                    except:
                        pass                
            #try:
            if len(cp[cc==maxchan]) == 1:
                m_eod = data[int(maxchan), int(cp[cc==maxchan]-width):int(cp[cc==maxchan]+width)]
            else:
                m_eod = []
            #except:
            #    m_eod = []

            artefact_mask = detect_artefacts(p_eods.T,dt)        
            wavefish_mask = detect_wavefish(p_eods.T)

            # if the detected peak(s) are potential eods, start the analysis.
            if np.sum(p_eods) == 0:
                continue

            p_eods[np.invert((artefact_mask).astype('bool'))] = 0

            artefact_mask = detect_artefacts(t_eods.T,dt)        
            wavefish_mask = detect_wavefish(t_eods.T)
            t_eods[np.invert((artefact_mask).astype('bool'))] = 0
            # if the detected peak(s) are potential eods, start the analysis.
            if np.sum(t_eods) == 0:
                continue

            #plot_snippet(eods,np.arange(int(cp[cc==maxchan]-width),int(cp[cc==maxchan]+width)),cp,ct,cc)
            #plt.show()

            print('an EOD!')

            print('all clusters:')
            print(all_clusters.keys())

            # get indices of clusters that are relevant for now. 
            # e.g. no idle clusters, no clusters that have a recent peak assigned
            cluster_keys, recent_spike = get_relevant_cluster_keys(all_clusters,eod_time*dt+starttime,track_length_t,width*dt)

            print('cluster keys:')
            print(cluster_keys)

            print(all_clusters.keys())

            # get cluster candidates and their extracted spatial features
            cc_keys, s_err = get_cluster_candidates(all_clusters,cluster_keys[cluster_keys!='potential_eod'],h_pattern,spatial_feature_error_threshold)

            print('cluster candidates:')
            print(cc_keys)

            # now for each cluster candidate, keep only the ones that have similar temporal features
            # imput both eods centered around peak and centered around trough.
            # for each candidate, I extract another width. so I pass all data + peaks and troughs.
            ac_keys, t_err = assess_candidates(all_clusters,cc_keys,maxchan,data,cp,ct,cc,temporal_feature_error_threshold,dt)

            print('accepted candidates:')
            print(ac_keys)

            
            # filter out candidates if they have the same maxchan
            # this step doesnt seem to do much..
            # if I filter them, they are not added, I should add them anyway but dont save features.
            # it is already filtered because it has two cluster candidates.

            # UNCOMMENT THIS IF I GET CLUSTER ISSUES (with this it worked for stationary fish for 32 min.)
            # maxchans = np.array([np.argmax(cc_spatial_features[k]) for k in ac_keys])
            # unique, ii, counts = np.unique(maxchans, return_inverse=True, return_counts=True)
            # ac_keys = ac_keys[counts[ii]<2]
            # print('accepted candidates:')
            # print(ac_keys)

            # only update clusters if there are no double candidates who have the same maxchan.

            # if no clusters could be assigned at all, assign to potential eod or do the manual input thing.
            if (len(ac_keys) > 0):
                # update these clusters
                for key in ac_keys:


                    # treat potential clusters as noise. only add points that recursively connect to existing clusters.

                    # only update with new features if there is one candidate selected
                    # otherwise update the times but keep old features and positions.

                    # also check if it is a single eod?

                    # if the regression fit has a high error, also dont save it.
                    # because then there might be an artefact present. -- I dont seem to have problems with artefacts.


                    # I rather have problems when fish that are close in space emit an EOD at the same time.
                    # as it is a mix of both signals, the cluster can change to accept signals from the other cluster.
                    # for now I ignore it, I should find a way to check colliding clusters as I now just assign it to no clusters
                    # which causes gaps in the spike train.


                    if (len(cc_keys)>1) or (recent_spike == True) or not (single_eod(h_pattern)) \
                    or (np.linalg.norm(get_position(h_pattern)-get_position(all_clusters[key].get_last_feature('f_spatial'))) > 1):
                        print('not a single EOD')
                        c_sf = all_clusters[key].get_last_feature('f_spatial')
                        c_tf = all_clusters[key].get_last_feature('f_temporal')
                    else:
                        print('single_eod')
                        c_sf = h_pattern
                        c_tf = np.stack([p_eods,t_eods])

                    locale=locals()
                    dd = {v:locale[v] for v in debug_vars}
                    
                    if (key =='potential_eod') and (len(ac_keys) == 2) and (single_eod(h_pattern)):
                        print('single EOD:')
                        print(single_eod(h_pattern))
                        connecting_cluster = ac_keys[1]
                    else:
                        connecting_cluster = 0

                    if key != 'potential_eod' or len(ac_keys)==1:
                        all_clusters[key].update(eod_time*dt+starttime,c_sf,c_tf,connecting_cluster,debug_dict=dd)
                
                
                if 'potential_eod' in ac_keys:

                    print('spikes in last second:')
                    print(all_clusters['potential_eod'].get_spike_count(eod_time*dt+starttime))

                    if all_clusters['potential_eod'].get_spike_count(eod_time*dt+starttime) >= 10:
                        mc =  np.argmax(all_clusters['potential_eod'].f_spatial, axis=1)
                        u, i, c = np.unique(mc, return_inverse=True, return_counts=True)

                        print('mcs:')
                        print(mc)

                        if np.max(c) >= 10:
                            
                            # update if with the potential clusters.
                            g_idxs = np.where(c[i]==np.max(c))[0]

                            # do any of these connect to existing clusters?? 
                            if (np.count_nonzero(all_clusters['potential_eod'].f_connecting_clusters)>0) and (len(np.unique(all_clusters['potential_eod'].f_connecting_clusters))==2):
                                print('appending to existing cluster')
                                print(all_clusters['potential_eod'].f_connecting_clusters)
                                
                                conc = str(int(np.max(all_clusters['potential_eod'].f_connecting_clusters)))

                                add = np.where(all_clusters['potential_eod'].f_connecting_clusters==0)[0]
                                new_ts = all_clusters['potential_eod'].f_ts[add]

                                # revert the cluster to the newest timepoint.
                                t, fs, ft = all_clusters[conc].revert(np.min(new_ts))

                                s_idx = np.argsort(np.append(t,new_ts))

                                print(fs.shape)
                                print(all_clusters['potential_eod'].f_spatial[add].shape)

                                m_t = np.append(t,new_ts)[s_idx]
                                m_fs = np.concatenate((fs,all_clusters['potential_eod'].f_spatial[add]))[s_idx]
                                m_ft = np.concatenate((ft,all_clusters['potential_eod'].f_temporal[add]))[s_idx]

                                # update them with (a combination of) the new values
                                for i in range(len(s_idx)):
                                    all_clusters[conc].update(m_t[i],m_fs[i],m_ft[i],0)

                                # check for each cluster if they connected to the potential    
                            else:                 
                                # include tf later, if it doesnt work properly oterwise.
                                # get relevant tf
                                # all_tf = potential_eod.f_temporal[c[i]==np.max(c)]
                                # compute distance matrix
                                # distances = np.linalg.norm()

                                # also I could check again here if I can connect to any of the existing clusters.
                                
                                # sth like, check if any clusters are idle? but what if this is a fast EOD it cannot connect to a slow one.
                                # either use another threshold or say, c>-10 and dt >1 second.
                                # so the tracking length should be bigger.
                                ############################################################################################################

                                # if there are at least 10 close clusters, make it a new cluster.
                                new_cluster = cluster_object(str(maxlabel+1),track_length,temporal_featnum=f_width,debug_vals=debug_vars)
                                maxlabel = maxlabel + 1 

                                #sort them in chronological order
                                s_idx = np.argsort(all_clusters['potential_eod'].f_ts[g_idxs])
                                g_idxs = g_idxs[s_idx]

                                for j in g_idxs:
                                    locale=locals()
                                    dd = {v:locale[v] for v in debug_vars}
                                    new_cluster.update(all_clusters['potential_eod'].f_ts[j],all_clusters['potential_eod'].f_spatial[j],all_clusters['potential_eod'].f_temporal[j],0,debug_dict=dd)
                            
                                # add to cluster space
                                all_clusters[str(maxlabel)] = new_cluster
                            
                            # remove these features from potential eods so they are not clustered again.
                            all_clusters['potential_eod'].delete(g_idxs)
                            

            else:#if (recent_spike == False) and (len(cc_keys)<2) and (single_eod(h_pattern)): # and (potential_eod.recently_spiked() == False):
                print('maybe its a new cluster?')
                
                c_sf = h_pattern
                #c_eod = p_eods[:,np.argmax(c_sf)]
                c = np.argmax(c_sf)
                
                c_tf = np.stack([p_eods,t_eods]) #real_feat(c_eod,dt)[1]
                
                
                # create new clusters               
                if manual_input:
                    print('manual input required')
                    answer, manual_input = ask_the_user(h_pattern,p_eods[maxchan],eod_time*dt)
                    if answer == 'y':
                        print('adding cluster')

                        # create new EOD cluster 
                        new_cluster = cluster_object(str(maxlabel+1),track_length,temporal_featnum=f_width,debug_vals=debug_vars)
                        maxlabel = maxlabel + 1 

                        locale=locals()
                        dd = {v:locale[v] for v in debug_vars}
                        
                        # for temp features, save current EOD shape.
                        # this also implies the width.

                        new_cluster.update(eod_peak*dt+starttime,c_sf,c_tf,0, debug_dict=dd)
                        all_clusters[str(maxlabel)] = new_cluster
                else:
                    print('potential_eod at time: %f s'%(eod_time*dt))
                    # add to potential EOD space
                    # -->  implement later when the rest works.

                    locale=locals()
                    dd = {v:locale[v] for v in debug_vars}

                    all_clusters['potential_eod'].update(eod_peak*dt+starttime, c_sf, c_tf, 0,debug_dict=dd)
                    
                    # check if there were 10 eods in the last second with the same maxchan.
                    # if yes, compare their temp feat
                    # if they are close, add new cluster.
                    

        #else:
        #    print('wavefish or artefact at time: %f s'%(p_idx*dt))
        # add eel if there is one.
        #if p_idx in cur_eel:
        #    print('adding to eel cluster')
        #    eel.update(p_idx*dt+starttime,np.var(eods,axis=0),real_feat(eods[:,np.argmax(np.var(eods,axis=0))],dt)[1],0)
        
        
    if endtime == endtimes[-1]:
        print('saving to: ')
        print('%s%i.pkl'%(save_path,int(endtime/60)-1))
        pickle.dump(pd(endtime-60,endtime,dt,all_clusters,master_filepath,slave_filepath,eel),open('%s%i.pkl'%(save_path,int(endtime/60)-1),'wb'))
    elif endtime%60 == 0:
        print('writing data to: %s%i.pkl'%(save_path,int(endtime/60)-1))
        pickle.dump(pd(endtime-60,endtime,dt,all_clusters,master_filepath,slave_filepath,eel),open('%s%i.pkl'%(save_path,int(endtime/60)-1),'wb'))
        print('emptying clusters')
        
        for label, cluster in all_clusters.items():
            cluster.empty_features()
        
        print('emptying eel and potential eods')
        eel.empty_features()

        # remove idle clusters from all_clusters.
        # except for potential_eod

        all_clusters_buffer = {}
        for c,v in all_clusters.items():
            if (not v.is_idle(eod_time*dt+starttime,track_length_t)) or (c=='potential_eod'):
                all_clusters_buffer[c] = v
        all_clusters = all_clusters_buffer

    return 0


if __name__ == '__main__':
    
    # command line arguments:
    parser = argparse.ArgumentParser(description='Analyze EOD waveforms of weakly electric fish.')    
    parser.add_argument('master_files', nargs=1, default='', type=str,
                        help='name of a file with time series data of an EOD recording')
    parser.add_argument('slave_files', nargs=1, default='', type=str,
                        help='name of a file with time series data of an EOD recording')
    args = parser.parse_args()

    for master_file,slave_file in zip(args.master_files,args.slave_files):
        if master_file[-1] == '/':
            save_folder = 'data/results/' + master_file.split('/')[-2] + '/'
        else:
            save_folder = 'data/results/' + master_file.split('/')[-1] + '/'

        starttime = 0
        
        if os.path.exists(save_folder):
            # check the last file that was saved and continue analysis there.
            starttime = len([name for name in os.listdir(save_folder) if '.pkl' in name])

        else:
            # make dir.
            os.mkdir(save_folder)


        # maybe first check the last file that was output to the save folder and continue analysis from there?
        # 
        get_clusters(master_file,slave_file,save_folder,starttime=starttime*60,debug_vars=['c_sf','h_pattern','m_eod','s_err','t_err'])

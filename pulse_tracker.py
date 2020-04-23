import thunderfish.dataloader as dl
import thunderfish.eventdetection as ed
import pulse_tracker_helper as pth

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

import pickle
import re

import copy


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
        self.f_temporal = np.ones((track_length, temporal_featnum))*99
        self.f_ts = np.zeros(track_length)
        self.f_pc = np.zeros(track_length)


        # storage buffers for EOD instances (time, location)
        self.ts = np.zeros(max_peak_count)
        self.positions = np.ones((max_peak_count,2))*99

        # storage buffers for things you want to track (debugging)
        self.debug = {v:[] for v in debug_vals}
    
    def update(self,t,spatial_feature,temp_feature,pc,debug_dict={}):
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
        self.f_pc[int(self.peak_count%self.track_length)] = pc

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
        self.f_pc[idxs] = 0

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

def extract_eod_times(data,thresh,peakwidth):
    print('extracting times')
    all_peaks = []
    channels = []

    for i in range(data.shape[1]):
        y = data[:,i]
        pk, tr = ed.detect_peaks(y, thresh)

        if len(pk)>1:
            peaks = pth.makeeventlist(pk,tr,y,peakwidth)
            peakindices, _, _ = pth.discardnearbyevents(peaks[0],peaks[1],peakwidth)
            peaks = np.transpose(peaks[0,peakindices])

            all_peaks.append(peaks)
            channels.append(i)

    y = data
    
    idx_arr = np.ndarray([0])
    chan_arr = np.ndarray([0])

    for channel, p in zip(channels,all_peaks):
        idx_arr = np.concatenate([idx_arr,p])
        chan_arr = np.concatenate([chan_arr,np.ones(len(p))*channel])

    sort_idx = np.argsort(idx_arr)
    idx_arr = idx_arr[sort_idx]
    chan_arr = chan_arr[sort_idx]

    idx_arr_u = np.unique(idx_arr)
    elec_mask = np.zeros((len(idx_arr_u),data.shape[1]))

    for i,t_u in enumerate(idx_arr_u):
        elec_mask[i,chan_arr[idx_arr==t_u].astype('int')] = 1

    print('times extracted')
    return idx_arr_u, elec_mask


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
    
    print(maxchan)
    # then check if they connect.
    xpos = np.mod(maxchan,grid_shape[1])
    ypos = np.floor(maxchan/grid_shape[1])

    print(xpos)
    print(ypos)

    if (np.abs(np.diff(xpos))<=1) and (np.abs(np.diff(ypos))<=1):
        return True
    else:
        return False


### functions for discarding artefacts and wavefish

def detect_artefacts(eods,dt,threshold=8000, cutoff=0.75):

    xf = np.linspace(0.0, 1.0/(2.0*dt), len(eods)/2)
    
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

def plot_snippet(eods):
    for i in range(eods.shape[1]):
        plt.subplot(4,8,i+1)
        plt.plot(eods[:,i])
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
    
    if len(cluster_keys) > 0:
        # get spatial features from clusters
        #c_feats = np.array([])

        c_feats = np.stack([clusters[key].f_spatial for key in cluster_keys])

        # alternative way by subtracting
        c_diff = -c_feats + spatial_pattern
        c_diff[c_diff > 0] = 0 #only negative values can be errors

        # get the error as a fraction of the cluster variance
        c_error = np.sum(np.abs(c_diff),axis=2)/np.sum(c_feats,axis=2)

        cluster_candidate_keys = cluster_keys[np.count_nonzero(c_error<spatial_feature_error_threshold,axis=1)>=noise_bound]


        # get differences between current spatial pattern and saved spatial patterns for clusters
        #c_diff = np.linalg.norm((c_feats - spatial_pattern)**2, axis=2)
        #amin = np.argmin(c_diff,axis=1)


        # for each cluster, use one representative spatial feature for linear regression
        #x_glm = np.stack([c_feats[i,amin[i]] for i in range(c_feats.shape[0])])

        # do a linear regression on these spatial patterns
        #reg = lsq_linear(np.transpose(x_glm),spatial_pattern,(0,1.5))

        # what if I do linear regression on all patterns and use the sum of the ones that are used as result??

        #c_error = np.abs(reg.x - 1)

        # select candidate clusters based on linear regression error.
        #cluster_candidate_keys = cluster_keys[c_error < spatial_feature_error_threshold]


        # maybe unneccesary now I dont use this anymore.
        # cc_spatial_features = get_spatial_features(cluster_candidate_keys,spatial_pattern,reg.x[c_error < spatial_feature_error_threshold],x_glm[c_error < spatial_feature_error_threshold])

        return cluster_candidate_keys, np.min(c_error,axis=1)
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

def ask_the_user(spatial_pattern,eods,time):
# ask for user input.
    manual_input = True
    
    plt.subplot(121)
    plt.imshow(spatial_pattern.reshape(4,8))
    plt.subplot(122)
    plt.plot(eods[:,np.argmax(spatial_pattern)])
    plt.show()
    answer = ''

    while answer not in ['y','n','d']:
        answer = input('is this a new and single EOD at time=%fs? (type y or n): '%(time))
    if answer == 'd':
        manual_input = False

    return answer, manual_input

def assess_candidates(clusters,cc_keys,eods,temporal_feature_error_threshold,dt,noise_bound=1):
    
    ac_keys = []
    min_dist_norm = [0]
    err = []
    
    # get temp features for the right channels
    for cc_key in cc_keys:
        # get channels of interest
        coi = np.argmax(clusters[cc_key].f_spatial,axis=1)
        # get temp feat of current eods for coi
        cur_feat = get_temp_feat(eods,coi,dt)

        # get difference
        f_dist = np.linalg.norm(clusters[cc_key].f_temporal - cur_feat.transpose(1,0,2),axis=2)
        f_dist_norm = f_dist/np.linalg.norm(clusters[cc_key].f_temporal)

        # change the differences that were computed for coi where eod==0
        f_dist_norm[:,np.count_nonzero(eods[:,coi],axis=0)== 0] = 99

        # as there are three for each, use min errors
        min_dist = np.min(f_dist,axis=0)
        min_dist_norm = np.min(f_dist_norm,axis=0)

        if np.count_nonzero(min_dist_norm<temporal_feature_error_threshold) >= noise_bound:
            ac_keys.append(cc_key)
        err.append(np.min(min_dist_norm))

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
    starttime=0, endtime=10000, peak_detection_threshold=0.001, peakwidth=0.0005, track_length=10, 
    spatial_feature_error_threshold=0.25, temporal_feature_error_threshold=0.1, 
    track_length_t = 10, debug_vars=[], save_artefacts=False, save_wavefish=False,mode=''):
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

    # check if path for previous path exists, if it does, use those clusters and empty them.
    print("looking for this:")
    print('%sresults_%s_%i.pkl'%(save_path,mode,int(starttime/60-1)))
    try: 
        co = pickle.load(open('%sresults_%s_%i.pkl'%(save_path,mode,int(starttime/60-1)),'rb'))
        print('loaded old clusters from %sresults_%s_%i.pkl'%(save_path,mode,int(starttime/60-1)))
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
        potential_eod = cluster_object('potential_eod',track_length)
        potential_eod.debug = {v:[] for v in debug_vars}

        all_clusters['potential_eod'] = potential_eod
   

    # load data in steps of n seconds
    starttimes = np.arange(starttime,endtime,load_buffer)
    endtimes = np.append(starttimes[1:],[endtime])

    # save data in steps of n minutes
    for i_block, (starttime, endtime) in enumerate(zip(starttimes,endtimes)):
        
        # load data and find potential eod times
        x, data, dt = load_channels(master_filepath,slave_filepath,starttime,endtime)
        idx_arr, elec_masks = extract_eod_times(data,peak_detection_threshold,peakwidth/dt)
        
        # extract eel times
        cur_eel = extract_eel_times(data)

        # width for cutting EODs
        cutwidth = peakwidth/dt

        # go through each peak
        for i_peak, (p_idx, elec_mask) in enumerate(zip(idx_arr,elec_masks)): 

            # cut along peak to get the EOD shape and convert to mV
            if p_idx < cutwidth:
                eods = data[0:int(p_idx+cutwidth)]*1000
            else:
                eods = data[int(p_idx-cutwidth):int(p_idx+cutwidth)]*1000
          
            # set artefacts and waves to zero
            artefact_mask = detect_artefacts(eods,dt)        
            wavefish_mask = detect_wavefish(eods)

            eods[:,np.invert((artefact_mask*wavefish_mask).astype('bool'))] = 0

            # if the detected peak(s) are potential eods, start the analysis.
            if np.sum(artefact_mask*wavefish_mask*elec_mask) > 0:

                print('an EOD!')

                # get indices of clusters that are relevant for now. 
                # e.g. no idle clusters, no clusters that have a recent peak assigned
                cluster_keys, recent_spike = get_relevant_cluster_keys(all_clusters,p_idx*dt+starttime,track_length_t,peakwidth)

                print('cluster keys:')
                print(cluster_keys)

                print(all_clusters.keys())

                # only use spatial pattern not resulting of wavefish or artefacts          
                spatial_pattern = np.var(eods,axis=0)

                # get cluster candidates and their extracted spatial features
                cc_keys, sf_err = get_cluster_candidates(all_clusters,cluster_keys,spatial_pattern,spatial_feature_error_threshold)

                print('cluster candidates:')
                print(cc_keys)

                # now for each cluster candidate, keep only the ones that have similar temporal features
                ac_keys, tf_err = assess_candidates(all_clusters,cc_keys,eods,temporal_feature_error_threshold,dt)

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


                        if (len(cc_keys)>1) or (recent_spike == True) or not (single_eod(spatial_pattern)) \
                        or (np.linalg.norm(get_position(spatial_pattern)-get_position(all_clusters[key].get_last_feature('f_spatial'))) > 1):
                            c_sf = all_clusters[key].get_last_feature('f_spatial')
                            c_eod = eods[:,np.argmax(spatial_pattern)]
                            c_tf = all_clusters[key].get_last_feature('f_temporal')
                        else:
                            c_sf = spatial_pattern
                            c_eod = eods[:,np.argmax(c_sf)]
                            c_tf = real_feat(c_eod,dt)[1]

                        locale=locals()
                        dd = {v:locale[v] for v in debug_vars}
                        
                        if (key =='potential_eod') and (len(ac_keys) == 2) and (single_eod(spatial_pattern)):
                            print('single EOD:')
                            print(single_eod(spatial_pattern))
                            pc = ac_keys[1]
                        else:
                            pc = 0

                        all_clusters[key].update(p_idx*dt+starttime,c_sf,c_tf,pc,debug_dict=dd)

                    if 'potential_eod' in ac_keys:

                        print('spikes in last second:')
                        print(all_clusters['potential_eod'].get_spike_count(p_idx*dt+starttime))

                        if all_clusters['potential_eod'].get_spike_count(p_idx*dt+starttime) >= 10:
                            mc =  np.argmax(all_clusters['potential_eod'].f_spatial, axis=1)
                            u, i, c = np.unique(mc, return_inverse=True, return_counts=True)

                            print('mcs:')
                            print(mc)

                            if np.max(c) >= 10:
                                
                                # update if with the potential clusters.
                                g_idxs = np.where(c[i]==np.max(c))[0]

                                # do any of these connect to existing clusters?? 
                                if (np.count_nonzero(all_clusters['potential_eod'].f_pc)>0) and (len(np.unique(all_clusters['potential_eod'].f_pc))==2):
                                    print('appending to existing cluster')
                                    print(all_clusters['potential_eod'].f_pc)
                                    
                                    conc = str(int(np.max(all_clusters['potential_eod'].f_pc)))

                                    add = np.where(all_clusters['potential_eod'].f_pc==0)[0]
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

                                    # if there are at least 10 close clusters, make it a new cluster.
                                    new_cluster = cluster_object(str(maxlabel+1),track_length,debug_vals=debug_vars)
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

                else: #if (recent_spike == False) and (len(cc_keys)<2) and (single_eod(spatial_pattern)): # and (potential_eod.recently_spiked() == False):
                    print('maybe its a new cluster?')
                    
                    c_sf = spatial_pattern
                    c_eod = eods[:,np.argmax(c_sf)]
                    c_tf = real_feat(c_eod,dt)[1]
                    
                    
                    # create new clusters               
                    if manual_input:
                        print('manual input required')
                        answer, manual_input = ask_the_user(spatial_pattern,eods,p_idx*dt)
                        if answer == 'y':
                            print('adding cluster')

                            # create new EOD cluster 
                            new_cluster = cluster_object(str(maxlabel+1),track_length,debug_vals=debug_vars)
                            maxlabel = maxlabel + 1 

                            locale=locals()
                            dd = {v:locale[v] for v in debug_vars}
                            new_cluster.update(p_idx*dt+starttime,c_sf,c_tf,0, debug_dict=dd)
                            
                            all_clusters[str(maxlabel)] = new_cluster
                    else:
                        print('potential_eod at time: %f s'%(p_idx*dt))
                        # add to potential EOD space
                        # -->  implement later when the rest works.

                        locale=locals()
                        dd = {v:locale[v] for v in debug_vars}

                        all_clusters['potential_eod'].update(p_idx*dt+starttime, c_sf, c_tf, 0,debug_dict=dd)
                        
                        # check if there were 10 eods in the last second with the same maxchan.
                        # if yes, compare their temp feat
                        # if they are close, add new cluster.
                        

            else:
                print('wavefish or artefact at time: %f s'%(p_idx*dt))
            # add eel if there is one.
            if p_idx in cur_eel:
                print('adding to eel cluster')
                eel.update(p_idx*dt+starttime,np.var(eods,axis=0),real_feat(eods[:,np.argmax(np.var(eods,axis=0))],dt)[1],0)

        if endtime == endtimes[-1]:
            print('saving to: ')
            print('%sresults_%s_%i.pkl'%(save_path,mode,int(endtime/60)-1))
            pickle.dump(pd(endtime-60,endtime,dt,all_clusters,master_filepath,slave_filepath,eel),open('%sresults_%s_%i.pkl'%(save_path,mode,int(endtime/60)-1),'wb'))
        elif endtime%60 == 0:
            print('writing data to: %sresults_%s_%i.pkl'%(save_path,mode,int(endtime/60)-1))
            pickle.dump(pd(endtime-60,endtime,dt,all_clusters,master_filepath,slave_filepath,eel),open('%sresults_%s_%i.pkl'%(save_path,mode,int(endtime/60)-1),'wb'))
            print('emptying clusters')
            
            for label, cluster in all_clusters.items():
                cluster.empty_features()
            
            print('emptying eel and potential eods')
            eel.empty_features()

            # remove idle clusters from all_clusters.
            # except for potential_eod

            all_clusters_buffer = {}
            for c,v in all_clusters.items():
                if (not v.is_idle(p_idx*dt+starttime,track_length_t)) or (c=='potential_eod'):
                    all_clusters_buffer[c] = v
            all_clusters = all_clusters_buffer

    return 0
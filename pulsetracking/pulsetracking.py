import thunderfish.dataloader as dl
import thunderfish.pulses as lp
from thunderfish.eventdetection import detect_peaks

import matplotlib.pyplot as plt
from matplotlib import gridspec

import numpy as np
import os

from scipy.ndimage.measurements import center_of_mass
from scipy.spatial import distance_matrix
from sklearn.cluster import OPTICS

import warnings

cmap = plt.get_cmap("tab10")

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

def zero_runs(a):
    """ Create an array that is 1 where a is 0, and pad each end with an extra 0.

    Parameters
    ----------
    a : numpy array

    Returns
    -------
    ranges : numpy array

    """
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def get_position(spatial_pattern,grid_shape=(4,8),n_elec=2):
    """ Get fish position estimate from spatial pattern.

    Parameters
    ----------
    spatial_pattern : numpy array
        Spatial pattern of electric activity, given as one value per electrode. 
        For multiple position estimate, a stacked array of spatial patterns 
        can be entered (n_samples, n_electrodes).
    grid_shape : tuple (optional)
        Shape of electrode grid.
        Defaults to (4,8)
    n_elec : int (optional)
        Number of electrodes to use for position estimation.
        Defaults to 2.
    Returns
    -------
    positions : numpy array
        A 2D position estimate for each input patter, 
        given in electrode coordinates.
    """
    spatial_pattern[~np.isfinite(spatial_pattern)] = 0
    mask_idx = spatial_pattern.argsort()[-n_elec:][::-1]
    mask = np.zeros(len(spatial_pattern))
    mask[mask_idx] = 1

    return np.asarray(center_of_mass((spatial_pattern*mask).reshape(grid_shape)))

def load_channels(filepath,starttime,endtime,verbose=0):
    
    # TODO: instead of two files, have list of filenames.

    """Load data from all channels for a given timeperiod.
    Parameters
    ----------
    filepaths : string or list of strings
        Paths to the data of all channels for one recording.
    starttime : float or int
        Starttime for extracting data in seconds.
    endtime : float or int
        Endtime for extracting data in seconds.
    verbose : int (optional)
        Verbosity level.
        Defaults to 0.

    Returns
    -------
    x : numpy array of floats
        Time array for extracted data.
    data : 2D numpy array of floats (len(x),n_channels)
        Electric activity for each timepoint in x on each channel.
    dt : float
        1/samplerate of extracted data.
    """

    if isinstance(filepath,str):
        with dl.open_data(filepath, -1, endtime-starttime) as data:
            dt = 1/data.samplerate

            # do something with the content of the file:        
            if starttime == None:
                starttime = 0
            if endtime == None:
                endtime = len(data)*dt

            x = np.arange(starttime,endtime,dt)
            y = data[starttime*data.samplerate:endtime*data.samplerate]
            
            if len(y)==0:
                return [],[],0

            if verbose>0:
                print('loaded T=%.2f:%.2f from %s'%(starttime,endtime,f))

        return x, y, dt

    else:
        ys = []

        for f in filepath:
            with dl.open_data(f, -1, endtime-starttime) as data:
                dt = 1/data.samplerate

                # do something with the content of the file:        
                if starttime == None:
                    starttime = 0
                if endtime == None:
                    endtime = len(data)*dt

                x = np.arange(starttime,endtime,dt)
                y = data[starttime*data.samplerate:endtime*data.samplerate]
                
                if len(y)==0:
                    return [],[],0
                
                ys.append(y)

            if verbose>0:
                print('loaded T=%.2f:%.2f from %s'%(starttime,endtime,f))

        try:
            data = np.concatenate(ys,axis=1)
        except:
            return [],[],0

        return x, data, dt

def extract_eod_times(data, samplerate, interp_freq, min_peakwidth, max_peakwidth, width_factor,verbose=0):
    """ Extract all timepoints with potential EOD activity based on peaks, troughs and peak-trough widths.

    Parameters
    ----------
    data : 2D numpy array
        Electric activity on all
    samplerate
    interp_freq
    min_peakwidth
    max_peakwidth
    width_factor
    verbose : int (optional)

    Returns
    -------
    x_peaks
    x_troughs
    eod_heights
    eod_widths
    channels
    all_int_data
    int_samplerate
    interp_f
    """

    all_int_data = np.array([])

    channels = []
    x_peaks = []
    x_troughs = []
    eod_hights = []
    eod_widths = []

    for i in range(data.shape[1]):
        y = data[:,i]
        
        x_peak, x_trough, eod_hight, eod_width, int_samplerate, int_data, interp_f, _ = lp.extract_eod_times(y, samplerate, width_factor=width_factor, interp_freq=interp_freq, min_peakwidth=min_peakwidth, max_peakwidth=max_peakwidth)

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
    
    if verbose>0:
        print('%i potential EOD times extracted'%len(a))

    return x_peaks[a], x_troughs[a], eod_hights[a], eod_widths[a], channels[a], all_int_data, int_samplerate, interp_f


### functions for extracting features


### functions for discarding artefacts and wavefish


## functions for extracting eel times
# TODO: make this a numba function.

def extract_eel_times(data,verbose=0):
    all_eel_times = []
    for i in range(data.shape[1]):
        y = data[:,i]
        et = zero_runs(np.diff(y))
        aplist = np.mean(et[et[:,1] - et[:,0] > 5], axis=1)

        if len(aplist) > 0:
            all_eel_times = np.append(all_eel_times,aplist)

    if len(all_eel_times) > 0:
        all_eel_times = np.unique(all_eel_times)

    if verbose>0:
        print('%i eeltimes extracted'%len(all_eel_times))

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

def analyse_window(spatial_patterns,ts,maxchans,starttime,endtime,samplerate,data,nchannels,
    x_peaks,x_troughs,eod_width,channels,eod_hights,cutwidth,window_size,window_dt,min_samples,
    max_clus,min_correlation=0.99,min_correlation_moving=0.75,coverage_factor=0.9,verbose=0,plot_level=0):                  
    
    # initialize storage arrays   
    mean_eods = np.zeros((int(max_clus*(endtime-starttime)/window_dt), int(nchannels*cutwidth*samplerate)))
    times = np.zeros(int(max_clus*(endtime-starttime)/window_dt))
    freqs = np.zeros(int(max_clus*(endtime-starttime)/window_dt))
    spiketimes = []
    counter_c = 0
    
    for t in np.arange(starttime,endtime-window_size+window_dt,window_dt):

        # remember where analysis started for this timeblock
        prev_counter_c = counter_c

        sl = (np.where((ts/samplerate+starttime>=t)&(ts/samplerate+starttime<t+window_size))[0]).astype('int')
        ctt = ts[sl]
        s_patterns = spatial_patterns[sl]

        if len(s_patterns)<min_samples:
            continue

        #take root of spatial patterns
        s_patterns[np.isnan(s_patterns)] = 0
        pc = OPTICS(min_samples=min_samples).fit(s_patterns).labels_

        for cl in np.unique(pc[pc!=-1]):
            
            # get the maxchan of this cluster
            # not by the average but the most prominent maxchan??
            all_mc, counts = np.unique(maxchans[sl][pc==cl],return_counts=True)
            maxchan = all_mc[np.argmax(counts)]
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
                    p_eods = np.zeros((nchannels,int(cutwidth*samplerate)))
                    t_eods = np.zeros((nchannels,int(cutwidth*samplerate)))
                    # center around biggest pt pair.
                    p_eods = data[:, int(cp[cc==maxchan][np.argmax(ch[cc==maxchan])]-cutwidth*samplerate/2):int(cp[cc==maxchan][np.argmax(ch[cc==maxchan])]+cutwidth*samplerate/2)]
                    p_eods[np.isnan(p_eods)] = 0 # set nan to zero as broken elctrodes give NaN
                    t_eods = data[:, int(ct[cc==maxchan][np.argmax(ch[cc==maxchan])]-cutwidth*samplerate/2):int(ct[cc==maxchan][np.argmax(ch[cc==maxchan])]+cutwidth*samplerate/2)]  
                    t_eods[np.isnan(t_eods)] = 0
                    mean_snip_p.append(p_eods.flatten())
                    mean_snip_t.append(t_eods.flatten())

            mean_snip_p = np.vstack(mean_snip_p)
            mean_snip_t = np.vstack(mean_snip_t)
            
            if np.mean(np.std(mean_snip_p,axis=0))<np.mean(np.std(mean_snip_t,axis=0)):
                mean_snip = np.mean(mean_snip_p,axis=0).reshape(nchannels,-1)
            else:
                mean_snip = np.mean(mean_snip_t,axis=0).reshape(nchannels,-1)

            mean_eods[counter_c] = mean_snip.flatten()
            times[counter_c] = t
            spiketimes.append(ts[sl][pc==cl]/samplerate+starttime)
            freqs[counter_c] = np.median(1/np.diff(ts[sl][pc==cl]/samplerate))
            counter_c += 1
               
        if plot_level > 0:
            gs = gridspec.GridSpec(counter_c - prev_counter_c,3)
            fig = plt.figure()
            ax = fig.add_subplot(gs[:,0])
            for i,tp in enumerate(spiketimes[prev_counter_c:counter_c]):
                ax.plot(tp,-i*np.ones(len(tp)),'o')
            ax.set_xlim([t,t+window_size])
            ax.axis('off')

            s_patterns = np.std(mean_eods[prev_counter_c:counter_c].reshape(mean_eods[prev_counter_c:counter_c].shape[0],nchannels,-1),axis=2)**(1/3)
            for i,pattern in enumerate(s_patterns):
                ax = fig.add_subplot(gs[i,1])
                ax.imshow(pattern.reshape(4,8),vmin=np.min(s_patterns),vmax=np.max(s_patterns))
                ax.axis('off')
                ax.set_title(i)

            ax = fig.add_subplot(gs[:,2])
            if len(s_patterns)>1:
                pos=ax.imshow(np.corrcoef(s_patterns),vmin=0,vmax=1)
                fig.colorbar(pos, ax=ax)
            plt.title('t = %.2f'%t)
            print('---------------------')
        
        cur_freqs = freqs[prev_counter_c:counter_c]
        cur_mpe = mean_eods[prev_counter_c:counter_c]
        cur_tt = np.array(spiketimes[prev_counter_c:counter_c])

        if len(cur_mpe)>1:

            # first only merge stationary fish, correlation should be high.
            s_patterns = np.std(cur_mpe.reshape(cur_mpe.shape[0],nchannels,-1),axis=2)**(1/3)
            cors = np.corrcoef(s_patterns)
            np.fill_diagonal(cors,0)
            pops=[]

            if np.max(cors)>min_correlation:
                idxs = np.argwhere(cors>min_correlation)
                for i in idxs:
                    
                    x = i[0]
                    y = i[1]
                        
                    if np.arange(prev_counter_c,counter_c)[y] not in pops and np.arange(prev_counter_c,counter_c)[x] not in pops:
   
                        # append to whichever is more centered in the current timeblock
                        center_d = t+window_size/2
                        new_times = np.sort(np.append(cur_tt[x],cur_tt[y])).tolist()

                        if np.abs(center_d - np.mean(cur_tt[x])/samplerate+starttime) < np.abs(center_d - np.mean(cur_tt[y])/samplerate+starttime):
                            spiketimes[np.arange(prev_counter_c,counter_c)[x]] = new_times
                            pops.append(np.arange(prev_counter_c,counter_c)[y])
                        else:
                            spiketimes[np.arange(prev_counter_c,counter_c)[y]] = new_times
                            pops.append(np.arange(prev_counter_c,counter_c)[x])

                        if verbose>0 or plot_level>0:
                            print('merging cluster %i and %i with correlation of %.3f'%(x,y,cors[x,y]))

            if np.max(cors[cors<min_correlation])>min_correlation_moving:
                idxs = np.argwhere((cors>min_correlation_moving)&(cors<min_correlation)) #sort from high to low
                sort_idx = np.argsort(-cors[(cors>min_correlation_moving)&(cors<min_correlation)])
                idxs = idxs[sort_idx]
                for i in idxs:

                    x = i[0]
                    y = i[1]

                    new_times = np.sort(np.append(cur_tt[x],cur_tt[y]))
                    
                    # only combine if the new coverage is (almost) equal to the seperate coverages combined.
                    # I ignore the two largest ISIs, as they could be outliers.
                    coverage_x = len(cur_tt[x])*np.mean(np.sort(np.diff(cur_tt[x]))[:-2])
                    coverage_y = len(cur_tt[y])*np.mean(np.sort(np.diff(cur_tt[y]))[:-2])
                    coverage_merge = len(new_times)*np.mean(np.sort(np.diff(new_times))[:-2])

                    if coverage_merge < coverage_factor*(coverage_x + coverage_y):
                        continue

                    if np.arange(prev_counter_c,counter_c)[y] not in pops and np.arange(prev_counter_c,counter_c)[x] not in pops:
   
                        # append to whichever is more centered in the current timeblock
                        center_d = t+window_size/2

                        new_times = np.sort(np.append(cur_tt[x],cur_tt[y])).tolist()
                        
                        if np.abs(center_d - np.mean(cur_tt[x])/samplerate+starttime) < np.abs(center_d - np.mean(cur_tt[y])/samplerate+starttime):
                            spiketimes[np.arange(prev_counter_c,counter_c)[x]] = new_times
                            pops.append(np.arange(prev_counter_c,counter_c)[y])
                        else:
                            spiketimes[np.arange(prev_counter_c,counter_c)[y]] = new_times
                            pops.append(np.arange(prev_counter_c,counter_c)[x])

                        if verbose>0 or plot_level>0:
                            print('merging cluster %i and %i with correlation of %.3f and coverage scores of %.3f + %.3f > %.2f*%.3f'%(x,y,cors[x,y],coverage_x,coverage_y,coverage_factor,coverage_merge))

            for p in sorted(pops, reverse=True):
                spiketimes.pop(p)
                times = np.delete(times,p,0)
                mean_eods = np.delete(mean_eods,p,0)
                freqs = np.delete(freqs,p,0)
                counter_c -= 1
    
        if plot_level>0:
            plt.show()

    return mean_eods[:counter_c], times[:counter_c], spiketimes[:counter_c], freqs[:counter_c]


def get_clusters(file_paths, save_path, starttime=0, endtime=60*60*48, grid_shape=(4,8), interp_freq=200000, 
    min_pt_width=0.8e-4, max_pt_width=1.7e-4, cutwidth=7.5e-4, width_factor=3, LFR_threshold=0.7, max_EOD_phases=4, window_size=1, 
    window_dt=0.25, block_size=5, min_samples=10, min_correlation_block=0.5, max_clus=15, max_freq=130, save_block=60, verbose=0, plot_level=0):

    '''
    TODO: write that min_pt_width and max_pt_width is species dependent

    TODO: write Peter email on monday on the milestone. evidence: git/docs/thunderfish -> pulsefish annotation.
    
    '''

    # check if path for previous path exists, if it does, use those clusters and empty them.
    if verbose>0:
        print("looking for this:")
        print('%scache_%i.pkl'%(save_path,int(starttime/60-1)))
    try: 
        loaded_data = np.load('%scache_%i.npz'%(save_path,int(starttime/60-1)), allow_pickle=True)
        if verbose>0:
            print('loaded old clusters from %s%i.npz'%(save_path,int(starttime/60-1)))

        # load previous values to continue tracking from here.
        prev_clusters = loaded_data['clusters']
        prev_times = loaded_data['times']
        prev_spiketimes = loaded_data['spiketimes'].tolist()
        prev_mean_eods = loaded_data['mean_eods']   
        cluster_counter = loaded_data['cluster_counter'] 
        prev_freqs = loaded_data['frequencies']
        eel_times = loaded_data['eeltimes'].tolist()
        eel_positions = [ep for ep in loaded_data['eel_positions']]

        starttimes = np.arange(starttime+block_size,endtime,block_size)
        endtimes = np.append(starttimes[1:],[endtime]) + window_size-window_dt
        data_loaded = True
   
    except:   
        if verbose>0:
            print('initiating new clusters')
        # load data in steps of n seconds
        # first ever should be 15 s, after that only load 5s at a time.
        starttimes = np.append(np.array([starttime]),np.arange(starttime+block_size*3,endtime,block_size)) 
        endtimes = np.append(starttimes[1:],[endtime]) + window_size-window_dt
        prev_clusters = []
        prev_times = []
        prev_spiketimes = []
        prev_mean_eods = []
        prev_freqs = []
        eel_times = []
        eel_positions = []
        cluster_counter = 0
        data_loaded = False

    # initialize empty arrays for saving traces and positions
    all_clusters = np.ones(int(max_freq*max_clus*save_block/window_dt))*-1
    all_ts = np.zeros(int(max_freq*max_clus*save_block/window_dt))
    all_positions = np.ones((int(max_freq*max_clus*save_block/window_dt),2))*-1

    # initialize counter to keep track of which positions of these arrays are filled
    a_counter = 0

    # save data in steps of n minutes
    for i_block, (starttime, endtime) in enumerate(zip(starttimes,endtimes)):

        # load data and find potential eod times
        x, data, dt = load_channels(file_paths,starttime,endtime,verbose=verbose-1)
        
        # if it is not possible to load more data, break out of loop
        if len(data)==0:
            break

        # extract eel positions + times
        cur_eeltimes = extract_eel_times(data,verbose=verbose-1)

        # cut blocks from the data and determine eeltimes.
        for cet in cur_eeltimes:
            s_pattern = np.var(data[max(0,int(cet-cutwidth/dt)):min(len(data),int(cet+cutwidth/dt))],axis=0)
            eel_positions.append(get_position(s_pattern))

        # append eel times
        eel_times.extend((cur_eeltimes*dt+starttime).tolist())
        
        # extract eod timepoints and interpolated data
        x_peaks, x_troughs, eod_hights, eod_widths, channels, data, samplerate, interp_factor = extract_eod_times(data, 1/dt, interp_freq, min_pt_width, max_pt_width, cutwidth/min_pt_width, verbose=verbose-1)
        # set new dt for interpolated data
        dt = 1/samplerate

        # initialize storage for eods and timepoints for the current analysis block
        spatial_patterns = np.zeros((len(x_peaks),data.shape[0]))
        ts = np.zeros(len(x_peaks))
        max_channels = np.zeros(len(x_peaks))
        num=0

        # initialize skip variable
        skip = 0

        # go through each detected peak
        for i_peak, (eod_time, eod_width, eod_peak, eod_trough) in enumerate(zip((x_peaks+x_troughs)/2, eod_widths, x_peaks, x_troughs)): 

            # skip if peak has already been visited
            if skip > 0:
                skip -= 1
                continue
            
            # check out surrounding peaks, troughs, and hights.
            idxs = (((x_peaks>(eod_time-eod_width)) & (x_peaks<(eod_time+eod_width)) & (x_troughs>(eod_time-eod_width*width_factor)) & (x_troughs<(eod_time+eod_width*width_factor))) | ((x_troughs>(eod_time-eod_width)) & (x_troughs<(eod_time+eod_width)) & (x_peaks>(eod_time-eod_width*width_factor)) & (x_peaks<(eod_time+eod_width*width_factor))))
            
            # select current peaktimes, troughtimes, channels and EOD heights
            cp = x_peaks[idxs]
            ct = x_troughs[idxs]
            cc = channels[idxs]
            ch = eod_hights[idxs]

            # set skip value so these timepoints are not analysed twice.
            skip = len(cc)

            if len(ch)>1 and np.min(ch)/np.max(ch) > 0.25:
                continue

            # TODO: only use grid shape for position estimates, get nr of electrodes from data.

            # create EOD height pattern            
            h_pattern = np.zeros((grid_shape[0]*grid_shape[1]))
            h_pattern[cc.astype('int')] = ch

            # compute centered EODs. Center around peak of channel with highest EOD.
            p_eods = np.zeros((grid_shape[0]*grid_shape[1],int(cutwidth*samplerate)))
            p_eods = data[:, int(cp[np.argmax(ch)]-samplerate*cutwidth/2):int(cp[np.argmax(ch)]+samplerate*cutwidth/2)]
            p_eods[np.isnan(p_eods)] = 0 # set nan to zero as broken elctrodes give NaN

            # check if it is centered around the highest peak in the snip.
            if np.max(ch)<np.max(np.abs(p_eods)) or np.sum(p_eods)==0:
                skip=0
                continue

            # check if the signals in p_eods are realistic
            if np.sum(p_eods) == 0:
                continue

            # check if the signal in the most active channel is an EOD or an artefact.
            mean_eod = p_eods[np.argmax(np.var(p_eods,axis=1))]
            mean_eod = mean_eod-np.mean(mean_eod)
            snip_peaks, snip_troughs = detect_peaks(mean_eod,np.std(mean_eod))
            cut_fft = int(len(np.fft.fft(mean_eod))/2)
            low_frequency_ratio = np.sum(np.abs(np.fft.fft(mean_eod))[:int(cut_fft/(2*interp_factor))])/np.sum(np.abs(np.fft.fft(mean_eod))[:int(cut_fft)])           
            if ((len(snip_peaks)+len(snip_troughs)) > max_EOD_phases) or (low_frequency_ratio<LFR_threshold):
                continue

            # append time, spatial pattern and max channel.
            ts[num] = eod_time
            max_channels[num] = cc[np.argmax(ch)]

            # normalize eods and then compute spatial pattern
            p_n, _ = lp.subtract_slope(p_eods,h_pattern.T)
            p_n = (p_n.T - np.mean(p_n,axis=1)).T
            spatial_patterns[num] = np.std(p_n,axis=1)**(1/3)  # TODO is this one of the things that slows it down?
            num+=1

        ts = ts[:num]
        max_channels = max_channels[:num]
        spatial_patterns = spatial_patterns[:num]

        # check if there were enough good EODs to cluster
        if len(spatial_patterns)<min_samples:
            if verbose>0:
                print('NO FISH')
            mean_eods = prev_mean_eods
            times = prev_times
            freqs = prev_freqs
            spiketimes = prev_spiketimes
        else:
            # analyse the new window of EODs
            mean_eods, times, spiketimes, freqs = analyse_window(spatial_patterns, ts, max_channels, starttime, endtime, 
                samplerate, data, grid_shape[0]*grid_shape[1], x_peaks, x_troughs, max_pt_width*samplerate, channels, eod_hights, cutwidth, window_size, window_dt, 
                min_samples, max_clus, verbose=verbose-1, plot_level=plot_level-1)

            # concatenate previous values.
            if len(prev_mean_eods)>0:
                mean_eods = np.concatenate((prev_mean_eods,mean_eods))
                times = np.concatenate((prev_times,times))
                freqs = np.concatenate((prev_freqs,freqs))
                spiketimes = prev_spiketimes + spiketimes

        # load previous clusters
        clusters = np.ones(len(mean_eods))*-1
        clusters[:len(prev_clusters)] = prev_clusters

        if len(mean_eods)>0:

            # modify patterns so that smaller values are more prominent (inverse relation of r**3)
            s_patterns = np.std(mean_eods.reshape(mean_eods.shape[0],grid_shape[0]*grid_shape[1],-1),axis=2)**(1/3)

            # spatial distance matrix
            dm_spatial = distance_matrix(s_patterns,s_patterns).flatten()

            # compute the raw score based on spatial pattern distance ranks
            temp = dm_spatial.argsort()
            score_spatial = np.empty_like(temp)
            score_spatial[temp] = np.arange(len(dm_spatial))
            
            # compute frequency component + rank
            dm_frequency = distance_matrix((freqs).reshape(-1,1),(freqs).reshape(-1,1)).flatten()
            temp = dm_frequency.argsort()
            score_frequency = np.empty_like(temp)
            score_frequency[temp] = np.arange(len(dm_frequency))
            
            # add distance matrices before sorting
            score_combined = (2/3)*score_spatial + (1/3)*score_frequency
            idxs = np.argsort(score_combined)
            score_combined_flat = score_combined[idxs][len(mean_eods):][::2]
            idxs = idxs[len(mean_eods):][::2]

            # go over data from lowest to highest distance scores
            for i in idxs:

                x = int(np.mod(i,len(mean_eods)))
                y = int(np.floor(i/len(mean_eods)))

                # dont connect if correlation <.5
                # break loop then.
                cor = np.corrcoef(s_patterns[x],s_patterns[y])
                if cor[0,1] < min_correlation_block:
                    continue

                # get timepoints
                t1 = times[x]
                t2 = times[y]

                # only connect if they are not at the same timepoint.
                if (t1 != t2):

                    # is either already in a cluster? if no, connect.
                    if clusters[x] == -1 and clusters[y] == -1:
                        # check if t for either of the connecting eods already exists?
                        clusters[x] = cluster_counter
                        clusters[y] = cluster_counter
                        cluster_counter = cluster_counter + 1

                    # if they already belong to the same cluster, continue
                    elif clusters[x] == clusters[y]:
                        continue

                    # if they both belong to different clusters, check out if they have any overlapping timpoints.
                    # if no, connect them.
                    elif (clusters[x] >=0) and (clusters[y] >= 0):

                        t_existing_x = times[clusters==clusters[x]]
                        t_existing_y = times[clusters==clusters[y]]

                        if len(times[(clusters==clusters[x])|(clusters==clusters[y])]) == len(np.unique(times[(clusters==clusters[x])|(clusters==clusters[y])])):                                                    
                        
                            clusters[clusters==clusters[x]] = min(clusters[x],clusters[y])
                            clusters[clusters==clusters[y]] = min(clusters[x],clusters[y])
                        
                    elif clusters[y] >=0:
                        t_existing = times[clusters==clusters[y]]
                        
                        if times[x] not in t_existing:
                            clusters[x] = clusters[y]

                    elif clusters[x] >=0:
                        t_existing = times[clusters==clusters[x]]        
     
                        # does the existing cluster already have a connection at this timepoint?
                        if times[y] not in t_existing:
                            clusters[y] = clusters[x]

        if verbose>0:
            print('%i fish in window'%len(np.unique(clusters[clusters!=-1])))
                    
        if plot_level>0:
            gs = gridspec.GridSpec(len(np.unique(clusters[clusters!=-1])),2)
            fig = plt.figure()
            ax = fig.add_subplot(gs[:,0])
            ax.plot(times,freqs,'o',c='k',alpha=0.5)

            for i,c in enumerate(np.unique(clusters[clusters!=-1])):
                ax.plot(times[clusters==c],freqs[clusters==c],c=cmap(i),label=c)
                ax_new = fig.add_subplot(gs[i,1])
                ax_new.imshow(np.mean(np.var(mean_eods[clusters==c].reshape(mean_eods[clusters==c].shape[0],grid_shape[0]*grid_shape[1],-1),axis=2)**(1/3),axis=0).reshape(4,8))
                ax_new.axis('off')
                ax_new.set_title(c)
            ax.legend()
            plt.tight_layout()

            gs = gridspec.GridSpec(len(np.unique(clusters[clusters!=-1])),int((block_size*3)/window_dt))
            fig = plt.figure()
            for i,c in enumerate(np.unique(clusters[clusters!=-1])):
                for j,tp in enumerate(times[clusters==c]):
                    ax_new = fig.add_subplot(gs[i,j])
                    ax_new.imshow((np.var(mean_eods[(clusters==c)&(times==tp)].reshape(grid_shape[0]*grid_shape[1],-1),axis=1)**(1/3)).reshape(4,8))
                    ax_new.axis('off')
            plt.tight_layout()
            plt.show()
       
        # Now save the middle block of connections.
        for c in np.unique(clusters[clusters!=-1]):
            # unpack individual EOD times and save these, with according positions and cluster labels.
            cur_stimes = []
            if not data_loaded:
                if len(np.where((clusters==c)&(times<block_size*2+starttime))[0])>0:
                    cur_stimes = np.concatenate(([spiketimes[i] for i in (np.where((clusters==c))[0]).astype('int')]))
                    cur_stimes = cur_stimes[cur_stimes<starttime+block_size*2]
            else:
                if len(np.where((clusters==c)&(times<starttime)&(times>=starttime-block_size))[0])>0:
                    cur_stimes = np.concatenate(([spiketimes[i] for i in (np.where((clusters==c))[0]).astype('int')]))
                    cur_stimes = cur_stimes[(cur_stimes>starttime-block_size)&(cur_stimes<starttime)]
           
            all_clusters[a_counter:a_counter+len(np.unique(cur_stimes))] = np.ones(len(np.unique(cur_stimes)))*c
            all_ts[a_counter:a_counter+len(np.unique(cur_stimes))] = np.sort(np.unique(cur_stimes))

            ccounter=0
            for i,t in zip((np.where((clusters==c))[0]).astype('int'),times[clusters==c]):
                all_positions[a_counter+ccounter:a_counter+ccounter+len(spiketimes[i])] = get_position(np.var(mean_eods[(clusters==c)&(times==t)].reshape(32,-1),axis=1))
                ccounter += len(spiketimes[i])
            a_counter += len(np.unique(cur_stimes))


        # save data of the last blocks of data to use for the next connecting iteration.
        if not data_loaded:
            # keep clusters of block 2
            prev_clusters = clusters[(times>=block_size+starttime) & (times<block_size*2+starttime)]

            # keep features+spiketimes of block 2 and 3
            prev_mean_eods = mean_eods[(times>=block_size+starttime)]
            prev_spiketimes = [spiketimes[i] for i in np.where(times>=block_size+starttime)[0].astype('int')]
            prev_times = times[(times>=block_size+starttime)]
            prev_freqs = freqs[(times>=block_size+starttime)]
            data_loaded = True

        else:
            # keep clusters of block 2
            prev_clusters = clusters[(times>=starttime-block_size) & (times<starttime)]

            # keep features+spiketimes of block 2 and 3
            prev_mean_eods = mean_eods[(times>=starttime-block_size)]
            prev_spiketimes = [spiketimes[i] for i in np.where(times>=starttime-block_size)[0].astype('int')]
            prev_times = times[(times>=starttime-block_size)]
            prev_freqs = freqs[(times>=starttime-block_size)]

        if (endtime == endtimes[-1]) or (starttime>(starttimes[0]) and (starttime%save_block==0)):
            if verbose>0:
                print('saving data to: %s%i.npz and %scache_%i.npz'%(save_path,int(endtime/60)-1,save_path,int(endtime/60)-1))

            if len(eel_times)>0:
                cur_et = np.array(eel_times)
                cur_ep = np.vstack(eel_positions)
                prev_eel_times = cur_et[cur_et>starttime]
                prev_eel_positions = cur_ep[cur_et>starttime]
                save_eel_times = cur_et[cur_et<=starttime]
                save_eel_positions = cur_ep[cur_et<=starttime]
            else:
                prev_eel_times = []
                prev_eel_positions = []
                save_eel_times = []
                save_eel_positions = []

            np.savez('%s%i.npz'%(save_path,int(endtime/60)-1), t=all_ts[:a_counter], c=all_clusters[:a_counter], p=all_positions[:a_counter],et=save_eel_times,ep=save_eel_positions)
            np.savez('%scache_%i.npz'%(save_path,int(endtime/60)-1), clusters=prev_clusters, cluster_counter=cluster_counter, mean_eods=prev_mean_eods, spiketimes=prev_spiketimes, times=prev_times, frequencies=prev_freqs, eeltimes=prev_eel_times, eel_positions=prev_eel_positions)

            # reset arrays for saving data
            all_clusters = np.ones(int(max_freq*max_clus*save_block/window_dt))*-1
            all_ts = np.zeros(int(max_freq*max_clus*save_block/window_dt))
            all_positions = np.ones((int(max_freq*max_clus*save_block/window_dt),2))*-1
            a_counter = 0

            if len(eel_times)>0:
                eel_times = prev_eel_times.tolist()
                eel_positions = [ep.tolist() for ep in prev_eel_positions]

    if len(eel_times)>0:
        cur_et = np.array(eel_times)
        cur_ep = np.vstack(eel_positions)
        prev_eel_times = cur_et[cur_et>starttime]
        prev_eel_positions = cur_ep[cur_et>starttime]
        save_eel_times = cur_et[cur_et<=starttime]
        save_eel_positions = cur_ep[cur_et<=starttime]
    else:
        prev_eel_times = []
        prev_eel_positions = []
        save_eel_times = []
        save_eel_positions = []
    
    if a_counter>0:
        if verbose>0:
            print('saving data to: %s%i.npz and %scache_%i.npz'%(save_path,int(np.ceil(endtime/60))-1,save_path,int(np.ceil(endtime/60))-1))

        np.savez('%s%i.npz'%(save_path,int(np.ceil(endtime/60))-1), t=all_ts[:a_counter], c=all_clusters[:a_counter], p=all_positions[:a_counter],et=save_eel_times,ep=save_eel_positions)
        np.savez('%scache_%i.npz'%(save_path,int(np.ceil(endtime/60))-1), clusters=prev_clusters, cluster_counter=cluster_counter, mean_eods=prev_mean_eods, spiketimes=prev_spiketimes, times=prev_times, frequencies=prev_freqs, eeltimes=prev_eel_times, eel_positions=prev_eel_positions)

    return 0

if __name__ == '__main__':

    # path to data
    path = '../data/'

    # go through data files and analyse them. If data already exists in save_path, continue analysis where it left off.
    m_files = ['2019-10-17-12_36', '2019-10-17-19_48',  '2019-10-18-09_52',  '2019-10-19-08_39',  '2019-10-20-08_30']
    s_files = ['2019-10-17-13_35', '2019-10-17-19_48',  '2019-10-18-09_44',  '2019-10-19-08_21',  '2019-10-20-08_30']

    for master_file,slave_file in zip(m_files,s_files):
        if master_file[-1] == '/':
            save_folder = path+'results/' + master_file.split('/')[-2] + '/'
        else:
            save_folder = path+'results/' + master_file.split('/')[-1] + '/'

        starttime = 0
        
        if os.path.exists(save_folder):
            # check the last file that was saved and continue analysis there.
            starttime = len([name for name in os.listdir(save_folder) if 'cache' in name])
        else:
            # make dir to save results in
            os.mkdir(save_folder)

        get_clusters([path+'master/'+master_file, path+'slave/'+slave_file],save_folder,starttime=starttime*60,verbose=3,plot_level=0)

# README file -> short.

# TODO: readme file for data on kraken.
# rename hour file.
# notes: write down how channels are arranged.

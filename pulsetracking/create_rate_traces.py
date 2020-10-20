import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from datetime import datetime, timedelta
import os, sys
import pickle as pkl

import warnings
warnings.filterwarnings("ignore")

<<<<<<< HEAD
=======

>>>>>>> f286f59c9a4db372d116be4b72aeef2b3a7b17a8
def RunningFunc(x, N, func=np.nanmedian, loc='middle'):
    """ Use sliding window of size N and run it over data x using function func.

    Parameters
    ----------
    x : numpy array
        Input data.
    N : int
        Sliding window size.
    func : python function (optional)
        Function to comute on sliding window.
        Defaults to np.nanmedian.
    loc : string (optional)
        Centering of sliding window. Options are: 'right', 'left' and 'middle'.
        Defaults to 'middle'

    Returns
    -------
    results : numpy array
        Modified array of size x.shape.
    """

    if len(x)<N+1:
        return np.ones(len(x))*func(x)
    
    result = np.zeros(len(x))
    
    idx = np.arange(N) + np.arange(len(x)-N+1)[:,None]
    b = [row[row>0] for row in x[idx]]
    
    if loc == 'middle':
        start = int(N/2)
    elif loc == 'right':
        start = 0
    else:
        start = int(N)

    result[start:start+len(idx)] = np.array([func(c) for c in b])
    result[:start] = result[int(N/2)-1]
    result[start+len(idx):] = result[-int(N/2)-1]

    return result

def fill_gaps(st,isi,N=200, max_ratio=1.5, min_freq=10):
    """ Fill gaps in EOD trace by computing the running median and setting all values above the average median ISI to NaN.

    Parameters
    ----------
    st : dict
        Dictionary with EOD times for each fish cluster in seconds.
    isi : dict
        Dictionary with EOD intervals for each fish cluster in seconds.
    N : int (optional)
        Running median window size.
        Defaults to 200.
    max_ratio : float
        Values of max_ratio*average median_ISI are set to NaN.
        Defaults to 1.5.
    min_freq : float
        Minimum EOD rate.
        Defaults to 10.

    Returns
    -------
    st : dict
        Dictionary with EOD times for each fish cluster in seconds.
    isi : dict
        Dictionary with EOD intervals for each fish cluster in seconds, where gaps have been set to NaN.
    """
    
    # just delete all isis above a certain number
    running_median = RunningFunc(isi,N)
    isi[isi>running_median*max_ratio] = np.nan
    isi[1/(isi)<min_freq] = np.nan

    return st, isi

def delete_artifacts(st,isi,positions,N=100):
    """ Delete unrealistic inter-EOD intervals from trace based on running median.

    Parameters
    ----------
    st : dict
        Dictionary with EOD times for each fish cluster in seconds.
    isi : dict
        Dictionary with EOD intervals for each fish cluster in seconds.
    positions : dict
        Dictionary with fish position coordinates for each fish cluster.
    N : int
        Running median window size.

    Returns
    -------
    st : dict
        Dictionary with EOD times for each fish cluster in seconds.
    isi : dict
        Dictionary with EOD intervals for each fish cluster in seconds.
    positions : dict
        Dictionary with fish position coordinates for each fish cluster.
    """

    runmed = RunningFunc(isi,N)
    
    while True:
        
        # delete these from the spiketrain.
        del_idxs = np.where(isi<runmed*0.7)[0]

        if len(del_idxs) == 0:
            break
        if len(del_idxs) > 1:
            c_del_idxs = del_idxs[np.append([0],np.where(np.diff(del_idxs) > 1)[0]+1)]
        else:
            c_del_idxs = del_idxs

        st = np.delete(st,c_del_idxs)
        runmed = np.delete(runmed,c_del_idxs)
        positions = np.delete(positions,c_del_idxs,axis=0)
        
        isi = np.diff(st)
        
    return st, isi, positions
                    
def smooth_traces(spiketimes,isi,N):
    """ Smooth EOD rate traces by taking a running median with window size N, 
    and create an array spiketimes and EOD intervals without gaps, so an interpolated version of the rate trace can be plotted.

    Parameters
    ----------
    spiketimes : dict
        Dictionary with EOD times for each fish cluster in seconds.
    isi : dict
        Dictionary with EOD intervals for each fish cluster in seconds.
    N : int
        Running median window size.

    Returns
    -------
    stsDict : dict
        Dictionary with EOD times for each fish cluster in seconds with NaNs.
    fstsDict : dict
        Dictionary with EOD times for each fish cluster in seconds without NaNs.
    isisDict : dict
        Dictionary with EOD intervals for each fish cluster in seconds with NaNs.
    fisisDict : dict
        Dictionary with EOD intervals for each fish cluster in seconds without NaNs.

    """
        
    isisDict = {}
    stsDict = {}
    fisisDict = {}
    fstsDict = {}
    
    for i,sti in enumerate(spiketimes.keys()):
        if len(spiketimes[sti])>0:
            
            to_nan_idx = np.where(np.diff(spiketimes[sti])>1)[0]
            del_idx = np.where(np.isnan(isi[sti]))[0]
            
            cisi = isi[sti]
            isis = RunningFunc(cisi,N)

            isis[to_nan_idx] = np.nan

            fisisDict[sti] = 1/(isis)
            fstsDict[sti] = spiketimes[sti]

            isis[del_idx.astype('int')] = np.nan

            isisDict[sti] = 1/(isis)
            stsDict[sti] = spiketimes[sti]
    
    return stsDict,fstsDict,isisDict,fisisDict


def load_data(fi, file_count, path):
    """ Load data from .npz pulsetracking files per minumte of data and concatenate them.

        Parameters
        ----------
        fi : int
            Number of first file.
        file_count : int
            Number of total files to load.
        path : string
            Path to data

        Returns
        -------
        st : dict
            Dictionary with EOD times in seconds for each fish cluster.
        positions : dict
            Dictionary with fish position coordinates for each fish cluster.
        et : numpy array
            Eel EOD times in seconds.
        ep : 2D numpy array
            Eel position estimate coordinates.
    """
    et = []
    ep = []
    st = {}
    positions = {}

    for i in range(fi,fi+file_count):
        np_path = '%s%i.npz'%(path,i)
        data = np.load(np_path)
            
        idx = []
        used_clusters = []
        used_labels = []
        peakwidth = 20
        
        clusters = data['c']
        t = data['t']
        pos = data['p']
        
        if len(data['et']) > 0:
            et.extend(data['et'])
            try:
                ep.extend(data['ep'])
            except:
                pass
        
        for k in np.unique(clusters[clusters!=-1]):
            if k in st:
                st[k] = np.append(st[k],t[clusters==k])
                positions[k] = np.concatenate((positions[k],pos[clusters==k]))
            else:
                st[k] = t[clusters==k]
                positions[k] = pos[clusters==k]

    if len(ep)>0:
        ep = np.vstack(ep)
    return st, positions, np.array(et), ep

def process_traces(st,positions):
    """ Process raw EOD rate traces.
        Fill gaps, delete artefacts and smooth the traces.

        Parameters
        ----------
        st : dict
            Dictionary with EOD times in seconds for each fish cluster.
        positions : dict
            Dictionary with fish position coordinates for each fish cluster.

        Returns
        -------
        sts : dict
            Dictionary with EOD times for each fish cluster in seconds with NaNs.
        fsts : dict
            Dictionary with EOD times for each fish cluster in seconds without NaNs.
        isis : dict
            Dictionary with EOD intervals for each fish cluster in seconds with NaNs.
        fisis : dict
            Dictionary with EOD intervals for each fish cluster in seconds without NaNs.
        positions : dict
            Dictionary with fish position coordinates.

    """
    spti = {}
    isis = {}
    rav = {}

    nspti = {}
    nisis = {}
    nrav = {}

    for i,sti in enumerate(st.keys()):
        print('analysing trace nr %i'%i)
        cst = st[sti]
        cp = positions[sti]
        cisi = np.diff(cst)
        if len(cisi) > 0:
            spti[sti], isis[sti], positions[sti] = delete_artifacts(cst,cisi,cp)
            nspti[sti], nisis[sti] = fill_gaps(spti[sti][:-1],np.diff(spti[sti]))

    print('smooting traces')

    return smooth_traces(nspti,nisis,5), positions


def plot_traces(sts, fsts, isis, fisis, positions, et, ep, fi, file_count, dati, path, N=500, fmin=0, fmax=120):
    """ Plot and save the processed EOD rate traces and their position estimates for each minute of data.

        Parameters
        ----------
        sts : dict
            Dictionary with EOD times for each fish cluster in seconds with NaNs.
        fsts : dict
            Dictionary with EOD times for each fish cluster in seconds without NaNs.
        isis : dict
            Dictionary with EOD intervals for each fish cluster in seconds with NaNs.
        fisis : dict
            Dictionary with EOD intervals for each fish cluster in seconds without NaNs.
        positions : dict
            Dictionary with fish position coordinates.
        et : numpy array
            Eel EOD times in seconds.
        ep : 2D numpy array
            Eel position estimates in coordinates.
        fi : int
            First timepoint that was analysed (in minutes).
        file_count : int
            Amount of minutes of data.
        dati : datetime object
            Date and time at start of recording.
        path : string
            Path to save output file to.
        N : int (optional)
            Minimum ammount of EODs that should be in an EOD cluster to be plot with colour and alpha.
            Anything below this number is plotted transparantly in black.
            Defaults to 500.
        fmin : float (optional)
            Minimum EOD rate to display on graph.
            Defaults to 0.
        fmax : float (optional)
            Maximum EOD rate to display on graph.
            Defaults to 130.
    """

    cmap = plt.get_cmap("tab10")

    prev_clus=[]
    cluscol = {}

    for t in np.arange(fi,fi+file_count):
        
        fig = plt.figure(figsize=(15,5))
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        
        ax1.set_ylim([3,0])
        ax1.set_xlim([0,7])
        ax1.set_aspect('equal')
        
        if len(et)>0:
            try:
                ax1.scatter(ep[(et>t*60) & (et<((t+1)*60)),0],ep[(et>t*60) & (et<(t+1)*60),1],color='k',marker='x',alpha=0.1)
            except:
                pass
        
        ymin=fmax
        ymax=fmin
        num = 0
        
        curcols=[]
        for i,pk in enumerate(sts.keys()):
            
            t_idx = (sts[pk]>t*60) & (sts[pk]<(t+1)*60)
            
            if np.count_nonzero(t_idx) > 1:            
                p = positions[pk][:-1][t_idx]
                x = sts[pk][t_idx]
                y = isis[pk][t_idx]
                fx = fsts[pk][t_idx]
                fy = fisis[pk][t_idx]
                
                if len(sts[pk])>N and np.count_nonzero(np.isnan(y)-1)>1:
                    if pk in cluscol:
                        color = cluscol[pk]
                    else:
                        for c in range(-20,1):
                            if -c not in curcols:
                                color = -c
                        cluscol[pk] = color
                        
                    curcols.append(color)    
                    ax2.plot(x,y,c=cmap(color))
                    ax2.plot(fx,fy,alpha=0.5,c=cmap(color))
                    ax1.scatter(p[(~np.isnan(y)),1],p[(~np.isnan(y)),0],alpha=0.5,color=cmap(color))
                    
                    num=num+1
                   
                    if np.nanmax(y)>ymax:
                        ymax=np.nanmax(y)
                    if np.nanmin(y)<ymin:
                        ymin=np.nanmin(y)
                    ax2.set_ylim([-0.05*(min(fmax,ymax)-max(fmin,ymin))+max(fmin,ymin),0.05*(min(fmax,ymax)-max(fmin,ymin))+min(fmax,ymax)])
                else:
                    ax2.plot(x,y,color='k',alpha=0.2)
                    ax2.plot(fx,fy,alpha=0.1,color='k')
                    ax1.scatter(p[(~np.isnan(y)),1],p[(~np.isnan(y)),0],alpha=0.005,color='k')
                   
        ax2.set_xlabel('time [s]')
        ax2.set_ylabel('EOD frequency [Hz]')

        if len(et) > 0 and len(et[(et>t*60) & (et<(t+1)*60)]) > 0:
            for xc in et[(et>t*60) & (et<(t+1)*60)]:
                plt.axvline(x=xc,color='k',alpha=0.1)

        directory = path + 'pics/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.title('%s'%(dati + timedelta(0,0,0,0,int(t))))
        print('saving to %s%i.png'%(directory,t))
        plt.savefig('%s%i.png'%(directory,t))
        plt.close()


if __name__ == '__main__':

    # the first command line argument is the path to the data.
    # if no path is given, try the standard path ../data/results/
    if len(sys.argv) == 1:
        path = '../data/results/'
    else:
        path = sys.argv[1]

    # go through data files and analyse them. If data already exists in save_path, continue analysis where it left off.
    subdirs = ['2019-10-17-12_36/', '2019-10-17-19_48/',  '2019-10-18-09_52/',  '2019-10-19-08_39/',  '2019-10-20-08_30/']
    dates = [datetime(2019,10,17,13,36), datetime(2019,10,17,19,48), datetime(2019,10,18,9,52), datetime(2019,10,19,8,39), datetime(2019,10,20,8,30)]

    for subdir,date in zip(subdirs,dates):

        if not os.path.exists(path+subdir):
            break

        file_count = int(np.floor(len([n for n in os.listdir(path+subdir) if ('.npz' in n) and ('traces' not in n)])/2))
        save_name = 'traces_%i.pkl'%file_count

        # first check if data already exists.
        if not os.path.exists(path+subdir+save_name):

            # check if there were other files with shorter traces and delete them
            for n in os.listdir(path+subdir):
                if 'traces' in n:
                    os.remove(path+subdir+n)

            # create data.
            st, positions, et, ep = load_data(0,file_count,path+subdir)
            (sts, fsts, isis, fisis), positions = process_traces(st,positions)
            plot_traces(sts, fsts, isis, fisis, positions, et, ep, 0, file_count, date, path+subdir)
            pkl.dump({'sts':sts, 'fsts':fsts, 'isis':isis, 'fisis':fisis, 'positions':positions, 'et':et, 'ep':ep}, open(path+subdir+save_name,'wb'))
        else:
            print('plotting data exists! delete it if you wish to recreate results')

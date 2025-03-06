'''
# utils.py
# Author: Xiaoqian Sun, 07/03/2024
# utilities function
'''


# Import Packages
#========================================================================================
import os 
import re
import math
import random
import pickle
import scipy as sp
import numpy as np
import pandas as pd
import seaborn as sns
from tslearn import metrics
from scipy.stats import expon
from scipy.stats import kstest
import scipy.special as special
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests

from spycon.coninf import GLMCC, GLMPP
import matplotlib.pyplot as plt
from collections import defaultdict

import neuron, simulation_addOn
from visualization import *; from utils import *

import warnings
warnings.filterwarnings('ignore')



# functions
#========================================================================================
#-------------------------------------- General --------------------------------------#
def orderByNumber(neuron):
    '''
    used in 1_replicatesGeneration.py or oder column by number, not lexicographically
    i.e., "post1", "post10", "post2" instead of "post1", "post2", "post10"
    '''
    return int(re.search(r'\d+', neuron).group())





#---------------------------- histogram, exp distribution ----------------------------#
def histo_info(datapoints):
    
    '''get distribution info from a list of datapoints'''
    
    hist, bin_edges = np.histogram(datapoints, density=True)
    x_hist_middle = [(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(hist))]

    data_mean = np.mean(datapoints)
    

    return(hist, x_hist_middle, data_mean)

def calculate_exp(meanValue, x):
    
    '''given mean and x, generate exponential y values'''
    
    rate = 1/meanValue
    y_exp = rate * np.exp(-rate*np.array(x))

    return(y_exp)

def drawFrom_exp(meanValue, numPoints):
    '''randomly draw points from exponential distribution given mean and number of points'''
    
    datapoints = np.random.exponential(scale=meanValue, size=numPoints) 
    
    # gather histogram info of this array
    hist, x_middle, dataMean = histo_info(datapoints)
    
    return(datapoints, hist, x_middle, dataMean)
        
def compare_rmse(y_actual, y_fit):
    '''calculate the Root_Mean_Squared_Error between two arries'''
    
    rmse = np.sqrt(mean_squared_error(y_actual, y_fit))
    
    return(rmse)

def ks_test(data, ifVerbose=False):
    
    '''
    compare data with an exponential distribution
        - ks_statistic ranges [0,1]
            0 - perfect match
            Higher values indicate a greater difference 

        - p_value < 0.05, accept
    '''
    
    ks_statistic, p_value = kstest(data, 'expon', args=(0, np.mean(data)))
    if ifVerbose:
        print("KS Score:", round(ks_statistic,4), "| P-value:", round(p_value,4))

    
    return(ks_statistic, p_value)

def evaluate_FR_exp(firing_rates, histTitle=None, thresholdPer=0.2, ifPlot=True, ifVerbose=False):
    
    '''
    given firing rate, exam:
        - if the distribution of firing rates fits an exp
        - if histo bars is an exp distribution'''
    
    
    hist, x_m, mean = histo_info(firing_rates)
    
    # 1st if
    hist_exp = calculate_exp(mean, x_m)
    rmse = compare_rmse(hist, hist_exp)
    rmse_threshold = np.sum(hist*thresholdPer)
    if rmse < rmse_threshold and ifVerbose:
        print('rmse within', str(thresholdPer*100)+'% fluctuation. RMSE =', rmse)
        
    # 2nd if
    ks_statistic, p_value = ks_test(firing_rates, ifVerbose=ifVerbose)
    

    if ifPlot:
        # Plot the histogram and the fitted curve
        plt.figure(figsize=(4, 2))
        
        plt.hist(firing_rates, density=True, alpha=0.6, color='g')

        plt.plot(x_m, hist, c='k', label='Mid Bin Connect')
        plt.plot(x_m, hist_exp, 'r-', lw=2, alpha=0.6, label='Fitted Exponential Curve')
        
        plt.title(histTitle+' RMSE='+str(round(rmse, 6))); plt.ylabel('Density'); plt.legend(loc=0); plt.show()
            
    
    return(rmse, ks_statistic, p_value)


#------------------------------------- sampling -------------------------------------#
def sample_gaussian(sampleSize, mu, sigma, lower_bound, upper_bound):

    '''
    draw randomly from gaussian distribution, if range truncated to [LB, UB]
    '''
    from scipy.stats import truncnorm
    
    # standar normal bounds
    sLB, sUB = (lower_bound-mu)/sigma, (upper_bound-mu)/sigma, 

    # draw samples
    samples = truncnorm.rvs(sLB, sUB, loc=mu, scale=sigma, size=sampleSize)

    return samples






#-------------------------- firing rate, F-I curve, Rheobase --------------------------#
def cal_firingRate(inputData, T, inDataType='spkTrain'):
    
    '''
    calculate firing rate (HZ), spikes/second
    
    Argument:
        - spkTrain: 1D/2D array, if 2D array, the input spkTrain should be in shape (num_neuron, num_timesteps)
        - duration: in timeUnit (ms/s)
    '''

    
    if inDataType == 'spkTrain':
    
        inputData = np.asarray(inputData)
        if len(inputData.shape)==1:
            numSpikes = inputData.sum()
            numSpk_array = np.asarray([numSpikes])
        else:        
            numSpk_array = np.asarray([inputData[i].sum() for i in range(inputData.shape[0])])
            
        return(numSpk_array/(T/1000))
    
    elif inDataType == 'spkTimes':
        inputData = preprocess_spkTimes(inputData)
        return (len(inputData)/(T/1000))
    else:
        raise ValueError('Input must be either spkTrain or spkTimes.')

def get_firingRates(neuronObj_list, neuron_idx, T, dt, spkKey='spkTrain'):
    '''
    given neuron idx, extract spk train of that neuron and calcualte firing rate
    2 options for spkKey: 'spkTrain', 'assignSpkTrain'
    '''
    
    spkTrains = get_keyValues_2_2DArray(neuronObj_list, neuron_idx, T, dt, Key=spkKey)
    FR_array = cal_firingRate(spkTrains, T, inDataType='spkTrain')
        
    return(FR_array)

def get_keyValues_2_2DArray(neuronObj_list, neuron_idx, T, dt, Key='spkTrain'):
    '''
    given neuron idx, extra data to 2D array for further analysis
    note that the output shape will be (num_neurons, num_datapoints)
    data can be formed into 2D array including: 
        'assignSpkTrain', 'spkTrains', 'memPotential', 'synTrace'
        'gE', 'gI', 'EPSP', 'IPSP', 'Xcurr', 'LeakC', 'spkTimes'
    '''

    N = len(neuron_idx)
    Lt = int(T/dt)
    outputArray = np.zeros((N, Lt))

    neuronIdx = 0
    for i in neuron_idx:
        neuronObj = neuronObj_list[i]
        data = neuronObj.get(Key)
        outputArray[neuronIdx][0:len(data)] = data
        neuronIdx+=1

    return(outputArray)

def loop_xInput_simulation(N_kwargs, T, dt, gE_bar, gI_bar, neuronType, N, Ne, Ni, maxns, connectivityM, startC, endC, numC, step=None):
    '''
    convinently adjust startC and endC and get reasonable F-I-curve
    '''
    
    if type(numC) != type(None):
        currents = np.linspace(startC, endC, numC).astype('int')
    elif type(step) != type(None):
        currents = list(range(startC, endC, step))
    else:
        raise ValueError('numC and step can not both be None to generate current list')
    
    FRs = []; numSpikes = []
    for I_b in currents:

        N_kwargs['externalInput'] = I_b
        N_obj = neuron.Neuron(T, dt, gE_bar, gI_bar, neuronType, **N_kwargs)

        simulator = simulation_addOn.Simulator(T, dt, N, Ne, Ni, maxns, [N_obj], CM=connectivityM)
        simulator.run(ifVerbose=False, pickN=None)    

        FRs.append(cal_firingRate(N_obj.get('spkTrain'), T, inDataType='spkTrain')[0])
        numSpikes.append(simulator.get('ns'))
        
    return(currents, FRs)

def check_spkOccurPosition(currents, FRs, startC, endC):
    
    '''
    check spike position and adjust current range accordingly
    '''
    
    ifChange = False
    
    spikePosition = np.where(np.array(FRs) != 0)[0]

    if len(spikePosition) == 0:
        # no spikes for all current
        startC = endC
        endC = endC + 200
        ifChange = True
    elif len(spikePosition) == len(FRs):
        startC = max(startC-200, 0)
        endC = startC+200
        ifChange = True
    else:
        spikeStartPosition = spikePosition[0] 

        if spikeStartPosition/len(FRs) > 0.75:
            startC = int(np.percentile(currents, 50))
            endC = startC  + 200
            ifChange = True
        elif spikeStartPosition/len(FRs) < 0.25:
            startC = int(startC * 0.75)
            endC = startC  + 200
            ifChange = True
            
    return (ifChange, startC, endC)

def linear_regression(x, y):
    
    slope, intercept, r, p, std_err = sp.stats.linregress(x, y)
    fit_y = np.array(x) * slope + intercept 
    
    return (slope, intercept, fit_y)

def F_I_curve(N_kwargs, T, dt, N, Ne, Ni, maxns, neuronType=0, startC=100, endC=300, numC=20, step=None, ifPlot=False, ifSave=False, savePath=None, filename=None):
    
    '''
    input various input current and get the firing rate – VS – input current curve
    which helps to get an idea of the neuron sensitivity to exc input
    also help to find Rheobase value of a neuron
    '''

    gE_bar, gI_bar = 0, 0
    connectivityM = np.zeros((N, N))
    
    
    # using while loop to find desired current range to generate firing rates with some 0s and some FRs
    ifChange = True
    while ifChange:
        # loop and get FRs
        currents, FRs = loop_xInput_simulation(N_kwargs, T, dt, gE_bar, gI_bar, neuronType, N, Ne, Ni, maxns, 
                                               connectivityM, startC=startC, endC=endC, numC=numC, step=None)
        # check
        ifChange, startC, endC = check_spkOccurPosition(currents, FRs, startC, endC)
        print('ifChange =', ifChange, 'current in range ['+str(startC)+', '+str(endC)+']')
        
    
    # calculate slope
    spikeStartPosition = np.where(np.array(FRs) != 0)[0][0]
    LR_startPosition = spikeStartPosition - 1

    nonZero_FRs = FRs[LR_startPosition:]
    nonZero_currents = currents[LR_startPosition:]
    
    slope, intercept, fit_FRs = linear_regression(nonZero_currents, nonZero_FRs)
    
        
    if ifPlot:
        fig, ax = plt.subplots(1,1, figsize=(5, 3))
        ax.plot(currents, FRs, '--o', color='b')
        ax.plot(nonZero_currents, fit_FRs, color='orange')
        ax.set_xlabel('Current (pA)');ax.set_ylabel('Spikes/sec')
        ax.spines[['right', 'top']].set_visible(False)

        
    # save/show
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()
        
        
    return(currents, FRs, slope, spikeStartPosition)

def find_Rheobase(N_kwargs, T, dt, N, Ne, Ni, maxns, searchStart, searchEnd, neuronType=0):
    '''
    by using F-I curve, we try to fine search Rheobase
    the minimum constant current amplitude required to depolarize a neuron to the threshold for generating an action potential. 
    '''
    
    gE_bar, gI_bar = 0, 0
    connectivityM = np.zeros((N, N))
    
    current, FRs= loop_xInput_simulation(N_kwargs, T, dt, gE_bar, gI_bar, neuronType, N, Ne, Ni, maxns,connectivityM, 
                                         startC=searchStart, endC=searchEnd, numC=None, step=1)
    Rheobase = current[np.where(np.array(FRs) !=0 )[0][0]]
    
    return(Rheobase)





#------------------------------------ ISIs related ------------------------------------#
def preprocess_spkTimes(spkTimes):
    '''
    remove any extra 0s/nans from spkTimes array (besides first 0)
    '''
    if len(spkTimes) < 2:
        raise ValueError ('Input spkTimes must contain a least 2 spikes.') 
    

    if spkTimes[0] == 0:
        spkTimes = [spkTimes[0]] + [x for x in spkTimes[1:] if x != 0]
    elif math.isnan(spkTimes[0]):
        raise ValueError ('Input spkTimes does not have any valid timepoints.') 
    else:
        spkTimes = [x for x in spkTimes if x != 0]
        spkTimes = [x for x in spkTimes if not math.isnan(x)]


    return list(spkTimes)

def spkTrain_2_spkTimes(spkTrain, dt):
    '''
    convert spkTrain into spkTimes in ms
    '''
    
    spike_indices = np.where(spkTrain == 1)[0]
    spkTimes = spike_indices * dt

    return spkTimes

def cal_interSpikeIntervals(spkTimes):
    '''calculate the time intervals between consecutive spikes'''
    

    spkTimes = preprocess_spkTimes(spkTimes)
        
    InterSpikeIntervals = []
    for i in range(len(spkTimes)-1):
        InterSpikeIntervals.append(round(spkTimes[i+1]-spkTimes[i], 4))
        
    return(InterSpikeIntervals)

def detect_spkTrain_burst(ISIs, spkTimes, burstThreshold=5, ifVerbose=True):
    '''
    If there is a spikes in a row and the interval between each spike is smaller than burstThreshold, then we call this a burst.
    E.g., ISIs = [9.9, 1.73, 2.06, 1.47, 1.53, 1.72, 2.04, 1.65...]
    
    The return is a list of sublists, sublist contains the start and end index of ISIs and spkTimes
    E.g., 
        - burstPeriod: [[3, 13], [17, 23]]
        - burst in spkTrain: [[3, 14], [17, 24]]: [3, 14] means from the 3th spike to 14th spike, including 14th, these are in burst period
    '''
    if ifVerbose:
        print('ISIs:', ISIs)
    burst_starts=[]; burst_ends=[]; inside_burst = False
    for i in range(len(ISIs)):
        if ISIs[i] < burstThreshold:
            if not inside_burst:
                burst_starts.append(i)
                inside_burst = True
        else:
            if inside_burst:
                burst_ends.append(i - 1)
                inside_burst = False
    
    # if burst reaches the end of the spkTrain
    if inside_burst:
        burst_ends.append(len(ISIs) - 1)
    
    # burst index in ISI or spkTrain
    burstPeriod = []; spkPerids = []
    for start, end in zip(burst_starts, burst_ends):
        if end-start > 2: # more than 3 spks in a row
            burstPeriod.append([start, end])
            spkPerids.append([start, end+1])
    
            if ifVerbose:
                print(f"  -Burst from ISI {start} to {end}: {ISIs[start:end+1]}")
                print(f"  -Burst from spkTimes {start} to {end+1}: {spkTimes[start:end+2]}")
                print('  ---------------------------------------------------------------------')
            
    return(burstPeriod, spkPerids)

def cal_nonBursty_firingRate(spkTimes, T, spkBurstPerids):

    '''
    only calculate firing rate in non-bursty period
    that is exclude bursty period from spkTrain and then use remaining T and spkTrain to calculate firing rate
    this function only process 1 spkTimes array (associated with the spkBurstPeriods for this spkTimes array)
    '''
    remainingT = T
    non_bursty_spkTimes = spkTimes.copy()
    
    for burstSpk in spkBurstPerids:
    
        periodTime = spkTimes[burstSpk[1]] - spkTimes[burstSpk[0]]
        remainingT -= periodTime
        
        period_spkTimes = spkTimes[burstSpk[0]:burstSpk[1]+1]
        indices = np.where(np.isin(non_bursty_spkTimes, period_spkTimes))
        non_bursty_spkTimes = np.delete(non_bursty_spkTimes, indices)
    
    non_bursty_fr = len(non_bursty_spkTimes)/(T/1000)
    return(non_bursty_fr)
        
def calculate_burstFraction(spkTimes, burstTresh=7):
    '''
    level of burstness of the spkTrain, calculated as BF = num_burst_spike / total_num_spikes
    '''

    ISIs = cal_interSpikeIntervals(spkTimes)

    burst_ISIs = sum(isi < burstTresh for isi in ISIs)
    BF = burst_ISIs / (len(ISIs)+1) if len(ISIs) > 0 else 0.0

    return round(BF, 3)

def interSpikeIntervals_Stats(spkTimes, burstThreshold=7, ifVerbose=True, ifPlotHist=True):
    
    '''
    std: indicates the variability in Inter-Spike-Intervals
    cv: coefficient of variation (CV = Std / Mean) quantifies the relative variability of ISIs
        - higher CV indicates more irregular firing
        - cv = 0: no variability
        - cv < 1: low, regular firing, second-order gamma distribution (k=1 with cs~0.71)
        - cv ≈ 1: moderate, Poisson-like
        - cv > 1: high, irregular
        
    serial correlation :
        - positive: long ISIs are likely to follow long ISIs, short ISIs follow short ISIs
        - negative: long ISIs are followed by short ISIs and vice versa
        - 0: no predictable pattern between consecutive ISIs (which is the case we want)
    '''

    # first, get ISIs
    #-------------------------------------------------------------
    InterSpikeIntervals = cal_interSpikeIntervals(spkTimes)

    # stats
    #-------------------------------------------------------------
    mean_ISI = np.mean(InterSpikeIntervals)
    median_ISI = np.median(InterSpikeIntervals)
    std_ISI = np.std(InterSpikeIntervals)
    cv_ISI = std_ISI / mean_ISI 
    serial_CC = np.corrcoef(InterSpikeIntervals[:-1], InterSpikeIntervals[1:])[0, 1]

    # burst
    #-------------------------------------------------------------
    burstPeriod, spkPerids = detect_spkTrain_burst(InterSpikeIntervals, spkTimes, burstThreshold=burstThreshold, ifVerbose=ifVerbose)
    burstingFraction = calculate_burstFraction(spkTimes, burstTresh=burstThreshold)

    # pirnt out
    #-------------------------------------------------------------
    if ifVerbose:
        print()
        print('Bursting fraction =', burstingFraction)
    
    
    if ifVerbose:
        print()
        if cv_ISI>=0 and cv_ISI<0.2:
            print('CV = '+str(round(cv_ISI, 4))+'. No variability in Inter-Spike-Intervals.')
        elif cv_ISI<0.8 and cv_ISI>=0.2: 
            print('CV = '+str(round(cv_ISI, 4))+'. Low variability in Inter-Spike-Intervals.')
        elif cv_ISI<1 and cv_ISI>=0.8:
            print('CV = '+str(round(cv_ISI, 4))+'. (Moderate)Poisson-like variability in Inter-Spike-Intervals.')
        else:
            print('CV = '+str(round(cv_ISI, 4))+'. (High)Poisson-like variability in Inter-Spike-Intervals.')
            
    if ifVerbose:
        print()
        if serial_CC == 0:
            print('serial correlation = 0. No Predicatble pattern between consecutive ISIs.')
        elif serial_CC > 0:
            print('serial correlation = '+str(round(serial_CC, 4))+'. Long Spks Follow Short Ones.')
        else:
            print('serial correlation = '+str(round(serial_CC, 4))+'. Long Spks Follow Long Ones.')

        
    if ifPlotHist:
        plt.figure(figsize=(4, 2))
        plt.hist(InterSpikeIntervals)
        plt.xlabel('Inter-Spike Interval (ms)'); plt.ylabel('Frequency'); plt.title('ISIs histogram'); plt.show()


    return (burstPeriod, spkPerids, burstingFraction, mean_ISI, median_ISI, std_ISI, cv_ISI, serial_CC)

def evenly_selectFromArray(input_array, numElement):
    '''evenly select from an array given number of elements to select from it'''

    step = len(input_array)//numElement
    select_array = input_array[::step][:numElement]

    return(select_array)

def downSample_burstSpk(spkTimes, spkTrain, T, range_t, fr_method='mean', selectMethod='random', burstThreshold=5, ifVerbose=False, ifPlot=False):

    '''
    The idea is that in burst period, randomly select # number of spikes, which is calculated based on the whole spkTrain's 
    firing rate, out of the total num spikes.    
    Steps:
        - calculate firing rate of this spkTrain
        - loop through each burst period
            - calculate how many spikes could happen given firing rate
            - randomly/evenly downsampled certain spikes and disgard the rest

    Note that 2 methods can be used for downsampling
        - random: np.random.choice
        - even: evenly_selectFromArray(input_array, numElement)
    '''
    InterSpikeIntervals = cal_interSpikeIntervals(spkTimes)
    burstPeriod, spkBurstPerids = detect_spkTrain_burst(InterSpikeIntervals, spkTimes, burstThreshold=burstThreshold, ifVerbose=ifVerbose)
    

    if fr_method == 'mean':
        fr = cal_firingRate(spkTrain, T, inDataType='spkTrain')[0]
    elif fr_method == 'non_bursty':
        fr = cal_nonBursty_firingRate(spkTimes, T, spkBurstPerids)
    else:
        raise ValueError("Choose firing rate calculation method from ['mean', 'non_bursty']")

    
    spkTimes_downSampled = spkTimes.copy()
    # loop through
    for burstSpk in spkBurstPerids:
        
        spkStartTime = spkTimes[burstSpk[0]]
        spkEndTime = spkTimes[burstSpk[1]]
        periodTime = spkEndTime - spkStartTime
    
        avgSpkNum = round(fr * periodTime /1000)
        oriSpkNum = len(spkTimes[burstSpk[0]:burstSpk[1]+1])
        if avgSpkNum < burstSpk[1]+1 - burstSpk[0]:

            if selectMethod == 'random':
                keepSpks = np.random.choice(spkTimes[burstSpk[0]:burstSpk[1]+1], avgSpkNum, replace=False)
                keepSpks.sort()
            elif selectMethod == 'even':
                keepSpks = evenly_selectFromArray(spkTimes[burstSpk[0]:burstSpk[1]+1], avgSpkNum)
            else:
                raise ValueError("Choose select method from ['random', 'even']")

            spkTimes_downSampled[burstSpk[0]:burstSpk[0]+avgSpkNum] = keepSpks
            spkTimes_downSampled[burstSpk[0]+avgSpkNum:burstSpk[1]+1] = 0
        
            if ifVerbose:
                print('Removing Spikes: ')
                print('  Based on '+fr_method+' firing rate, there should be', avgSpkNum, 'but now have', oriSpkNum, 'spikes')
                print('  Original spkTimes period:', spkTimes[burstSpk[0]:burstSpk[1]+1])
                print('  Downsamples spkTimes periodL:', keepSpks)
                print('  ---------------------------------------------------------------------')
        else:
            if ifVerbose:
                print("  Skip: The burst period contains fewer spikes than what would be expected based on avg firing rate.")
                print('  ---------------------------------------------------------------------')
        
    spkTimes_downSampled = spkTimes_downSampled[spkTimes_downSampled>0]


    # get spkTrain based on spkTimes
    spkTrain_downSampled = np.zeros_like(range_t)
    spike_indices = np.searchsorted(range_t, spkTimes_downSampled)
    spkTrain_downSampled[spike_indices] = 1
    
    
    # plot
    if ifPlot:
        plt.figure(figsize=(20, 1))
        plt.plot(spkTimes_downSampled, 1*np.ones(len(spkTimes_downSampled)), '|', color='orange', ms=20, markeredgewidth=2, label='downSampled')
        plt.plot(spkTimes, 2*np.ones(len(spkTimes)), '|', color='b', ms=20, markeredgewidth=2, label='original')
        plt.legend(loc=0); plt.show()

    return(spkTrain_downSampled, spkTimes_downSampled)

def spkTrain_df_burstProcess(spkTrain_df, T, range_t, cols=[], selectMethod='random', burstThreshold=5, ifVerbose=False, ifPlot=False, savePath=None, fileName=None):

    '''
    perform downSample_burstSpk() to each spkTrain in a dataframe, and save for later use
    '''


    
    if len(cols) == 0:
        target_cols = spkTrain_df.columns.tolist()
    else:
        target_cols = cols


    downSampled_spkTrain_df = spkTrain_df.copy()
    for target in target_cols:
        t_spkTrain = spkTrain_df[target].values
        t_spkTimes = range_t[t_spkTrain > 0.5]
    
        t_spkTrain_ds, t_spkTimes_ds = downSample_burstSpk(t_spkTimes, t_spkTrain, T, range_t, 
                                                           selectMethod=selectMethod, burstThreshold=burstThreshold, 
                                                           ifVerbose=ifVerbose, ifPlot=ifPlot)
        downSampled_spkTrain_df[target] = t_spkTrain_ds

    # save
    downSampled_spkTrain_df.to_csv(os.path.join(savePath, fileName))




#------------------------------------- CCG / dcCCH / GLMCC ------------------------------------#
def merge_spkTimes(pre_spkTimes, post_spkTimes):
    
    '''
    This function process pre/post spkTimes as input to CCG/GLMCC
        - merge pre/post spkTimes together
        - sort by time
        - sort neuron idx by time as well
    Return:
        - sorted times in ms
        - sorted idx
    
    '''
    
    # preprocess
    pre_spkT = preprocess_spkTimes(pre_spkTimes); pre_idx=[1]*len(pre_spkT)
    post_spkT = preprocess_spkTimes(post_spkTimes); post_idx=[2]*len(post_spkT)
    nspks1 = len(pre_spkT); nspks2 = len(post_spkT)  # num of spikes in pre/post-spkTrain

    # merge
    times = np.array(pre_spkT + post_spkT); idx = np.array(pre_idx + post_idx)
    
    if len(times) != len(idx):
        raise ValueError("The lens of 'spkTimes' and 'neuronIDs' must match")

    # sort, unique idx (1, 2)
    sort_idx = np.argsort(times); times = times[sort_idx]; idx = idx[sort_idx]


    return (times, idx, nspks1, nspks2)

def cal_CCG(times, idx, duration, bin_size, ifPlot=False, figsize=(6, 5), ifSave=False, savePath=None, filename=None):
    '''
    calculate CCG from spkTrain with refernece FMAToolbox/Analyses/CCG.m   
    this function applies to a pair of pre-post spkTimes, if apply to multiple pairs, call function in for loops 
    see tech details in `InferConnectivity/2-0-CCG.ipynb`
    '''

    
    # WIN - used in GLMCC CCG - half duration 
    WIN = int(duration/2)
    unique_idx = np.unique(idx); n_idx = len(unique_idx)

    # get num of bins
    half_bins = int(np.round(duration / (2 * bin_size))); n_bins = 2 * half_bins + 1
    t = np.arange(-half_bins, half_bins + 1) * bin_size
    
    # compute
    ccg = np.zeros((n_bins, n_idx, n_idx))
    for i, id1 in enumerate(unique_idx):
        for j, id2 in enumerate(unique_idx):
            
            '''
            Note:
                if i==j, it's calculating auto-correlogram
                case i=1, j=2 is the same as i=2, j=1
            '''
            if i > j:  # symmetric
                ccg[:, i, j] = ccg[:, j, i] 
                continue

            # time difference
            times1 = times[idx == id1]; times1 = times1.astype(np.float32)
            times2 = times[idx == id2]; times2 = times2.astype(np.float32)
            
            diffs = times2[:, None] - times1[None, :]; diffs = diffs.flatten()
            if i==0 and j==1:
                diffs_win = diffs[np.abs(diffs) <= WIN] # for GLMCC

            # binning
            bin_edges = np.arange(-half_bins - 0.5, half_bins + 1.5) * bin_size
            hist, _ = np.histogram(diffs, bins=bin_edges)

            # store
            ccg[:, i, j] = hist
            
    
    if ifPlot:
        fig, ax = plt.subplots(2, 1, figsize=figsize)
        ax[0].plot(t, ccg[:, 0, 0], label='Pre', lw=2)
        ax[0].plot(t, ccg[:, 1, 1], label='Post', lw=2); ax[0].set_title('ACH'); ax[0].legend()
        ax[1].bar(t, ccg[:, 0, 1], width=1, color='peru', alpha=0.8, edgecolor='none'); ax[1].set_title('CCH-pre-post') 
        plt.tight_layout()
        
        
        if ifSave:
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            plt.savefig(os.path.join(savePath, filename))
            plt.close()
        else:
            plt.show()


    cch = ccg[:, 0, 1]       # cross-correlation histogram
    cch_diffs = diffs_win
    ach1 = ccg[:, 0, 0]      # auto-correlation for trigger train (pre-spkTrain)
    ach2 = ccg[:, 1, 1]      # auto-correlation for referred train (post-spkTrain)
    
    return (cch, cch_diffs, ach1, ach2, n_bins, half_bins, t)

def cal_dccch(nIdx, pre_spkTimes, post_spkTimes, duration, bin_size, featureSummary, ifPlot=False, ifSave=False, savePath=None, filename=None):
    
    '''
    calculate dc-CCG from ccg with refernece EranStarkLab/CCH-deconvolution/blob/main/cchdeconv.m  
    this function applies to a pair of pre-post spkTimes, if apply to multiple pairs, call function in for loops 
    see tech details in `InferConnectivity/2-1-dcCCH.ipynb`
    '''
    
    ccg, cch, ach1, ach2, nspks1, nspks2, n_bins, half_bins, t = cal_CCG(pre_spkTimes, post_spkTimes, duration=duration, bin_size=bin_size, ifPlot=False)


    # make sure inputs are np.array
    cch, ach1, ach2 = map(np.asarray, (cch, ach1, ach2))
    nspks1, nspks2 = map(int, (nspks1, nspks2))
    
    # valide shape
    if cch.shape != ach1.shape or cch.shape != ach2.shape:
        raise ValueError("CCH, ACH1, and ACH2 must have the same shape.")
    if cch.shape[0] % 2 == 0:
        raise ValueError("CCH must have an odd number of bins.")

    
    # preprocess ach1
    ach1_normed = (ach1 - ach1.mean()) / nspks1 # extract mean + normalization
    ach1_normed[half_bins] = 1 - np.sum(ach1_normed[np.arange(n_bins) != half_bins])
    
    # preprocess ach2
    ach2_normed = (ach2 - ach2.mean()) / nspks2 # extract mean + normalization
    ach2_normed[half_bins] = 1 - np.sum(ach2_normed[np.arange(n_bins) != half_bins])
    
    # deconvolution
    # time domain to frequency domain 
    fft_ach1 = np.fft.fft(ach1_normed); freqs_ach1 = np.fft.fftfreq(len(ach1_normed))
    fft_ach2 = np.fft.fft(ach2_normed); freqs_ach2 = np.fft.fftfreq(len(ach2_normed))
    fft_cch = np.fft.fft(cch); freqs_cch = np.fft.fftfreq(len(cch))
    
    # remove achs effect & inverse
    den = fft_ach1 * fft_ach2 
    dccch = np.fft.ifft(fft_cch / den).real 
    dccch = np.roll(dccch, -1); dccch[dccch < 0] = 0

    if ifPlot:
        plot_dcCCH_process(nIdx, ach1, ach1_normed, freqs_ach1, fft_ach1, 
                           ach2, ach2_normed, freqs_ach2, fft_ach2, 
                           cch, freqs_cch, fft_cch, dccch, t, featureSummary, ifSave=ifSave, savePath=savePath, filename=filename)


    return dccch    

def cal_GLMCC(times, idx, preName, postName):
    '''
    use GLMCC() object in spycon.coninf (from eANN)
    '''
    
    
    # convert times from ms to s
    times_s = times / 1000
    
    # call 
    glmccM = GLMCC()
    glmccM_result = glmccM.infer_connectivity(times_s, idx)

    # parameters
    default_params = glmccM.default_params
    default_params['threshold'] = glmccM_result.threshold

    # summary
    all_weights = glmccM_result.all_weights
    graph = pd.DataFrame(glmccM_result.stats, columns=['outgoing_nodes', 'incoming_nodes', 'edge'])
    graph['weights'] = all_weights

    graph.outgoing_nodes[graph.outgoing_nodes==1] = preName
    graph.outgoing_nodes[graph.outgoing_nodes==2] = postName
    graph.incoming_nodes[graph.incoming_nodes==1] = preName
    graph.incoming_nodes[graph.incoming_nodes==2] = postName

    return (default_params, graph)

def cal_GLMPP(times, idx, preName, postName):
    
    # convert back to s
    times_s = times/1000
    
    glmppM = GLMPP()
    glmppM_result = glmppM.infer_connectivity(times_s, idx)

    # retrive
    default_params = glmppM.default_params
    default_params['threshold'] = glmppM_result.threshold

    all_weights = glmppM_result.all_weights
    graph = pd.DataFrame(glmppM_result.stats, columns=['outgoing_nodes', 'incoming_nodes', 'edge'])
    graph['weights'] = all_weights
    graph.outgoing_nodes[graph.outgoing_nodes==1] = preName
    graph.outgoing_nodes[graph.outgoing_nodes==2] = postName
    graph.incoming_nodes[graph.incoming_nodes==1] = preName
    graph.incoming_nodes[graph.incoming_nodes==2] = postName

    return (default_params, graph)












# ------------------------ save neuron parameters, neuron objs ------------------------#
def accumulated_sum(input_list):
    '''return accumulated sum of a list
    e.g., input_list = [50, 50, 50], 
    return [25, 75, 125]'''

    accu_sum = []
    for i in range(len(input_list)):
        half_accusum = input_list[i]/2 + np.sum(input_list[:i])
        accu_sum.append(half_accusum)

    return(accu_sum)

def dict_to_df(input_dict, columns=None, ifSave=False, savePath=None, filename=None):

    ''' dictionary to dataframe. Can be used to output neuron parameter space to dataframe'''
    df = pd.DataFrame.from_dict(input_dict, orient='index').reset_index()

    if type(columns) != type(None):
        if df.shape[1] != len(columns):
            raise ValueError('Input columns does not match dataframe number of columns')
        df.columns = columns
    
    # save
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        df.to_excel(os.path.join(savePath, filename+'.xlsx'))

    return(df)

def extract_neuronParas_fromObjs(obj_list, keys=[],neuronNames=[], ifSave=False, savePath=None, filename=None):


    '''
    specifically save 'paras' in Neuron object to dataframe
    '''

    paras_df_list = []
    for obj in obj_list:
        dic = obj.__dict__['paras']
        df = pd.DataFrame.from_dict(dic, orient='index')
        paras_df_list.append(df)

    Paras_df = pd.concat(paras_df_list, axis=1).T
    if len(keys) != 0:
        Paras_df = Paras_df[keys]
    if len(neuronNames) != 0:
        Paras_df['neuron_name'] = neuronNames
        Paras_df = Paras_df.set_index('neuron_name', drop=True)
        

    # save
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        Paras_df.to_csv(os.path.join(savePath, filename+'.csv'))

    return(Paras_df)

def neuronProperity_Report(T, dt, nType_list, nName_list, wE_list, wI_list, gEBar_list, gIBar_list, N_kwargs_list, ifSave=False, savePath=None, filename=None):
    
    '''
    generate a dataframe of neurons in the network, including:
        - gL ($gL = 1/R$)
        - input resistence
        - time constant
        - rheobase
        - F-I curve slope
        - gE/gI/WI/WE
        - EPSP/IPSP
    
    '''
    maxns=1000
    units = ['', 'nS', 'MΩ', 'ms', 'mV', 'mV', 'mV', 'nS', 'nS', '', '', 'nS', 'nS', 'mV', 'mV','', 'mV']
    
    
    propertyDF_list = []
    for neuronIdx in range(len(nType_list)):

        print('working on', nName_list[neuronIdx])

        neuron_type = nType_list[neuronIdx]
        if neuron_type==0: Ne = 1; Ni = 0; N = Ni + Ne
        else: Ne = 0; Ni = 1; N = Ni + Ne
            
    
        # Rheobase, slope
        N_kwargs = N_kwargs_list[neuronIdx] 
        current, FRs, slope, spikeStartPosition = F_I_curve(N_kwargs, T, dt, N, Ne, Ni, maxns, neuron_type,
                                                          startC=100, endC=300, numC=20, ifPlot=False)
        searchStart = current[spikeStartPosition-1]
        searchEnd = current[spikeStartPosition]
        Rheobase = find_Rheobase(N_kwargs, T, dt, N, Ne, Ni, maxns, searchStart, searchEnd, neuron_type)

        # other parameters
        wE = wE_list[neuronIdx]; wI = wI_list[neuronIdx]
        gE_bar = gEBar_list[neuronIdx]; gI_bar = gIBar_list[neuronIdx]
        N_obj = neuron.Neuron(T, dt, gE_bar, gI_bar, neuron_type, **N_kwargs)
        
        # summarize
        propertyDic = {'neuronName':nName_list[neuronIdx],
                       'g_Leak': N_obj.get('g_Leak'), 
                       'memResistance':N_obj.get('memResistance'),
                       'timeConstant':N_obj.get('tau_m'), 
                       
                       'V_rest/V_reset':N_obj.get('V_reset'), 
                       'V_excReversal':N_obj.get('V_excReversal'),
                       'V_inhReversal':N_obj.get('V_inhReversal'),
               
                       'gE_bar':gE_bar, 'gI_bar':gI_bar, 
                       'wE':wE, 'wI':wI, 
                       'gE':gE_bar*wE, 'gI':gI_bar*wI, 
                       'EPSP':-1/N_obj.get('g_Leak')*(gE_bar*wE*(N_obj.get('V_reset')-N_obj.get('V_excReversal'))),
                       'IPSP':-1/N_obj.get('g_Leak')*(gI_bar*wI*(N_obj.get('V_reset')-N_obj.get('V_inhReversal'))),
               
                       'F-I-Slope':slope, 'Rheobase':Rheobase}
        propertyDF = pd.DataFrame.from_dict(propertyDic, orient='index')
        propertyDF_list.append(propertyDF)
        
    
    allN_propertyDF = pd.concat(propertyDF_list, axis=1)
    allN_propertyDF['unit'] = units
    
    
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        allN_propertyDF.to_excel(os.path.join(savePath, filename+'.xlsx'))

        
    return(allN_propertyDF)


def save_data2Pickle(nameList, dataList, save_path, file_name):
    
    saveDic = {}
    
    if len(nameList) != len(dataList):
        raise ValueError("Input nameList and dataList must have same length")
    
    for i in range(len(nameList)):
        key = nameList[i]; data = dataList[i]
        saveDic[key] = data
        
    # save
    with open(os.path.join(save_path, file_name+'.pkl'), 'wb') as fp:
        pickle.dump(saveDic, fp)
        print(file_name+' pickle saved')

def generate_netSpk_report(T, netSpk, neuronName_list, neuronType_list, ifSave=False, savePath=None, filename=None):
    
    '''generate a dataframe to show neuron firings in the simulation'''
    
    spkTimes = netSpk[0]; spkTimes_list = list(spkTimes)
    spikeNeurons = netSpk[1]; spikeNeurons_list = list(spikeNeurons)

    # how many times each neuron fire
    occurrence = {int(item): spikeNeurons_list.count(item) for item in spikeNeurons_list}
    # what are the names/types of neuron
    spk_neuronNames = np.array(neuronName_list)[list(occurrence.keys())] #[neuronName_list[i] for i in occurrence.keys()]
    spk_neuronTypes = np.array(neuronType_list)[list(occurrence.keys())]
    # create a base dataframe
    spkNeuron_Summary = pd.DataFrame.from_dict(occurrence, orient='index', columns=['firing_times'])
    spkNeuron_Summary['neuron_name'] = spk_neuronNames
    spkNeuron_Summary['neuron_type'] = spk_neuronTypes
    spkNeuron_Summary['firing_rate'] = spkNeuron_Summary['firing_times']/(T/1000)

    # spk times of each neuron
    neuornspk_times = {}
    for key in occurrence.keys():
        neuornspk_times[key] = spkTimes[np.where(spikeNeurons==key)]
    neuornspk_time_df = pd.DataFrame.from_dict(neuornspk_times, orient='index')

    # concat
    netSpk_Summary = pd.concat([spkNeuron_Summary, neuornspk_time_df], axis=1)
    netSpk_Summary = netSpk_Summary.sort_values('firing_times', ascending=False)

    # save
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        netSpk_Summary.to_excel(os.path.join(savePath, filename+'.xlsx'))


    return(netSpk_Summary)
    
def ORN_PN_firingSummary(neuronObj_list, T, dt, num_ORNs, num_PNs, ifEvaluate_FRExp=True, ifVerbose=True, ifSave=False, savePath=None, filename=None):
    
    '''
    get ORN actual firing rate, PN firing rate
    '''
    ORN_actualFRs = get_firingRates(neuronObj_list, list(range(num_ORNs)), T, dt, spkKey='assignSpkTrain')
    PN_FRs = get_firingRates(neuronObj_list, list(range(num_ORNs, num_ORNs+num_PNs)), T, dt, spkKey='spkTrain')

    
    if ifEvaluate_FRExp:
        evaluate_FR_exp(ORN_actualFRs, histTitle='ORN', thresholdPer=0.2, ifPlot=True, ifVerbose=ifVerbose)
        evaluate_FR_exp(PN_FRs, histTitle='PN', thresholdPer=0.2, ifPlot=True, ifVerbose=ifVerbose)
        
    
    # generate PN/ORN firing ranking summary
    ORN_FR_Rank = pd.DataFrame.from_dict({'ORN_input_firing_rate':ORN_actualFRs, 
                                      'ORN_neuorn/obj_id':range(num_ORNs)}
                                    ).sort_values('ORN_input_firing_rate', ascending=False)
    PN_FR_Rank = pd.DataFrame.from_dict({'PN_firing_rate':PN_FRs, 
                                         'PN_neuorn_id':range(num_PNs),
                                         'PN_Obj_id':range(num_ORNs, num_ORNs+num_PNs)}
                                       ).sort_values('PN_firing_rate', ascending=False)
    ORN_PN_FR_Rank = pd.concat([ORN_FR_Rank, PN_FR_Rank], axis=1).reset_index(drop=True)
    
    
    if ifVerbose:
        print('ORN top 5 firing neurons:', ORN_PN_FR_Rank['ORN_neuorn/obj_id'][0:5].values)
        print('PN top 5 firing enurons:', ORN_PN_FR_Rank['PN_Obj_id'][0:5].values)
    
    # save
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        ORN_PN_FR_Rank.to_excel(os.path.join(savePath, filename+'.xlsx'))


    return(ORN_PN_FR_Rank, ORN_actualFRs, PN_FRs)  
    
def savesimulatorObjResult(simulatorObj, neuronObj_list, ORNs_FRs, num_ORNs, num_PNs, neuronName_list, neuronType_list, 
                           descriptiveWords, savePath, ifEvaluate_FRExp=True, netSpk_Summary=None, ORN_PN_FR_Rank=None):
    
    
    '''
    specifically write for fruit fly olfactory simualtion result
    '''
    
    T = simulatorObj.T; dt = simulatorObj.dt; N = simulatorObj.N

    if type(netSpk_Summary) == type(None):
        netSpk_Summary = generate_netSpk_report(T, simulatorObj.get('netSpk'), neuronName_list, neuronType_list,
                                            ifSave=True, savePath=savePath, filename='netSpkSummary')
    if type(ORN_PN_FR_Rank) == type(None):
        ORN_PN_FR_Rank,ORN_actualFRs,PN_FRs = ORN_PN_firingSummary(neuronObj_list, T, dt, num_ORNs, num_PNs, ifEvaluate_FRExp=ifEvaluate_FRExp, 
                                              ifVerbose=False,ifSave=True, savePath=savePath, filename='ORN_PN_FR_Rank')
    
    
    
    simuResult_dict = {}

    # general
    simuResult_dict['T'] = T
    simuResult_dict['dt'] = dt
    simuResult_dict['N'] = N
    simuResult_dict['Ne'] = simulatorObj.get('Ne')
    simuResult_dict['Ni'] = simulatorObj.get('Ni')
    simuResult_dict['ns'] = simulatorObj.get('ns')
    simuResult_dict['range_t'] = neuronObj_list[0].get('range_t')


    # about ORN
    simuResult_dict['ORNs_FRs'] = ORNs_FRs
    simuResult_dict['ORN_actualFRs'] = ORN_actualFRs
    simuResult_dict['ORNs_spkTrains'] = get_keyValues_2_2DArray(neuronObj_list, list(range(num_ORNs)), 
                                                                T, dt, Key='assignSpkTrain')

    # about PN
    PN_neuronIdx = list(range(num_ORNs, num_ORNs+num_PNs))
    simuResult_dict['PNs_para'] = dict_to_df(neuronObj_list[num_ORNs+1].get('paras'), columns=['PN Parameter', 'Values'])
    simuResult_dict['PNs_FRs'] = PN_FRs
    simuResult_dict['PNs_spkTrains'] = get_keyValues_2_2DArray(neuronObj_list, PN_neuronIdx, T, dt, Key='spkTrain')
    simuResult_dict['PNs_memPotential'] = get_keyValues_2_2DArray(neuronObj_list, PN_neuronIdx, T, dt, Key='memPotential')
    simuResult_dict['PNs_synTrace'] = get_keyValues_2_2DArray(neuronObj_list, PN_neuronIdx, T, dt, Key='synTrace')
    simuResult_dict['PNs_gE'] = get_keyValues_2_2DArray(neuronObj_list, PN_neuronIdx, T, dt, Key='gE')
    simuResult_dict['PNs_gI'] = get_keyValues_2_2DArray(neuronObj_list, PN_neuronIdx, T, dt, Key='gI')
    simuResult_dict['PNs_EPSP'] = get_keyValues_2_2DArray(neuronObj_list, PN_neuronIdx, T, dt, Key='EPSP')
    simuResult_dict['PNs_IPSP'] = get_keyValues_2_2DArray(neuronObj_list, PN_neuronIdx, T, dt, Key='IPSP')
    simuResult_dict['PNs_Xcurr'] = get_keyValues_2_2DArray(neuronObj_list, PN_neuronIdx, T, dt, Key='Xcurr')
    simuResult_dict['PNs_LeakC'] = get_keyValues_2_2DArray(neuronObj_list, PN_neuronIdx, T, dt, Key='LeakC')
    simuResult_dict['PNs_spkTimes'] = get_keyValues_2_2DArray(neuronObj_list, PN_neuronIdx, T, dt, Key='spkTimes')

    
    # about LIN
    LIN_neuronIdx = list(range(num_ORNs+num_PNs, N))
    simuResult_dict['LINs_para'] = dict_to_df(neuronObj_list[num_ORNs+num_PNs+1].get('paras'), columns=['LIN Parameter', 'Values'])
    simuResult_dict['LINs_FRs'] = get_firingRates(neuronObj_list, LIN_neuronIdx, T, dt, spkKey='spkTrain')
    simuResult_dict['LINs_spkTrains'] = get_keyValues_2_2DArray(neuronObj_list, LIN_neuronIdx, T, dt, Key='spkTrain')
    simuResult_dict['LINs_memPotential'] = get_keyValues_2_2DArray(neuronObj_list, LIN_neuronIdx, T, dt, Key='memPotential')
    simuResult_dict['LINs_synTrace'] = get_keyValues_2_2DArray(neuronObj_list, LIN_neuronIdx, T, dt, Key='synTrace')
    simuResult_dict['LINs_gE'] = get_keyValues_2_2DArray(neuronObj_list, LIN_neuronIdx, T, dt, Key='gE')
    simuResult_dict['LINs_gI'] = get_keyValues_2_2DArray(neuronObj_list, LIN_neuronIdx, T, dt, Key='gI')
    simuResult_dict['LINs_EPSP'] = get_keyValues_2_2DArray(neuronObj_list, LIN_neuronIdx, T, dt, Key='EPSP')
    simuResult_dict['LINs_IPSP'] = get_keyValues_2_2DArray(neuronObj_list, LIN_neuronIdx, T, dt, Key='IPSP')
    simuResult_dict['LINs_Xcurr'] = get_keyValues_2_2DArray(neuronObj_list, LIN_neuronIdx, T, dt, Key='Xcurr')
    simuResult_dict['LINs_LeakC'] = get_keyValues_2_2DArray(neuronObj_list, LIN_neuronIdx, T, dt, Key='LeakC')
    simuResult_dict['LINs_spkTimes'] = get_keyValues_2_2DArray(neuronObj_list, LIN_neuronIdx, T, dt, Key='spkTimes')


    # about connectivity
    simuResult_dict['CMWParas'] = dict_to_df(simulatorObj.get('CMWParas'), columns=['Connectivity', 'Values'])
    simuResult_dict['CM'] = simulatorObj.get('CM')
    simuResult_dict['CMW'] = simulatorObj.get('CMW')
    simuResult_dict['CMWs'] = simulatorObj.get('CMWs')
    simuResult_dict['outgoingCs'] = simulatorObj.get('outgoingCs')
    simuResult_dict['incomingCs'] = simulatorObj.get('incomingCs')

    
    # about network summary
    simuResult_dict['netSpk'] = simulatorObj.get('netSpk')
    simuResult_dict['netSpk_Summary'] = netSpk_Summary
    simuResult_dict['ORN_PN_FR_Rank'] = ORN_PN_FR_Rank


    # a note
    simuResult_dict['notes'] = descriptiveWords

    
    # save
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    with open(os.path.join(savePath, 'simuResult.pkl'), 'wb') as fp:
        pickle.dump(simuResult_dict, fp)
        print('saved successfully')
        
def timeSeries_stationary_test(spike_train):

    adf_result = adfuller(spike_train)
    print(f"ADF Statistic: {adf_result[0]} with p-value = {adf_result[1]}. And critical values at all levels are {adf_result[4]}" )

    if adf_result[1] < 0.05:
        return True
    else:
        return False

def cal_dynamicTimeWarping_plot(df, col1, col2, figsize=(4, 4), col1_styple='b-', col2_styple='b-', ifPlot=False, ifSave=False, savePath=None, filename=None):

    '''
    perform dynamic time warping, plot path and return distance

    '''
    
    s1 = df[col1].to_numpy().reshape((-1, 1))
    s2 = df[col2].to_numpy().reshape((-1, 1))
    path, sim = metrics.dtw_path(s1, s2)

    # plot
    if ifPlot:
        plt.figure(1, figsize=figsize)
        left, bottom = 0.01, 0.1; w_ts = h_ts = 0.05; left_h = left + w_ts + 0.02; width = height = 0.6; bottom_h = bottom + height + 0.02
        rect_s_y = [left, bottom, w_ts, height]; rect_gram = [left_h, bottom, width, height]; rect_s_x = [left_h, bottom_h, width, h_ts]
        ax_gram = plt.axes(rect_gram); ax_s_x = plt.axes(rect_s_x); ax_s_y = plt.axes(rect_s_y)
        
        mat = cdist(s1, s2)
        ax_gram.imshow(mat, origin='lower')
        ax_gram.axis("off"); ax_gram.autoscale(False)
        ax_gram.plot([j for (i, j) in path], [i for (i, j) in path], "w-", linewidth=3.)
        
        ax_s_x.plot(np.arange(s2.shape[0]), s2, col2_styple, linewidth=3.)
        ax_s_x.axis("off")
        
        ax_s_y.plot(- s1, np.arange(s1.shape[0]), col1_styple, linewidth=3.)
        ax_s_y.axis("off")
        
        plt.title('distance = '+str(round(sim,4)),  x=8, y=1.2); plt.tight_layout()
        
        # save/show
        if ifSave:
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            plt.savefig(os.path.join(savePath, filename))
            plt.close()
        else:
            plt.show()

    return(sim)









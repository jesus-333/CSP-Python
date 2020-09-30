# -*- coding: utf-8 -*-
"""
File containing various support function 

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

#%%
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
import scipy.signal
import scipy.linalg as la

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#%% Function for 100Hz dataset (Dataset IV-1) and data handling
# This function are specific for the dataset IV-1

def loadDataset100Hz(path, idx, type_dataset):
    tmp = loadmat(path + idx + '.mat');
    data = tmp['cnt'].T
    
    if(type_dataset == 'train'):
        b = tmp['mrk'][0,0]
        cue_position = b[0]
        labels  = b[1]
    else:
        cue_position = labels = None
    
    other_info = tmp['nfo'][0][0]
    sample_rate = other_info[0][0,0]
    channe_name = retrieveChannelName(other_info[2][0])
    class_label = [str(other_info[1][0, 0][0]), str(other_info[1][0, 1][0])]
    n_class = len(class_label)
    n_events = len(cue_position)
    n_channels = np.size(data, 0)
    n_samples = np.size(data, 1)
    
    other_info = {}
    other_info['sample_rate'] = sample_rate
    other_info['channel_name'] = channe_name
    other_info['class_label'] = class_label
    other_info['n_class'] = n_class
    other_info['n_events'] = n_events
    other_info['n_channels'] = n_channels
    other_info['n_samples'] = n_samples

    
    return data, labels, cue_position, other_info


def retrieveChannelName(channel_array):
    channel_list = []
    
    for el in channel_array: channel_list.append(str(el[0]))
    
    return channel_list


def computeTrial(data, cue_position, labels, fs, class_label = None):
    """
    Transform the 2D data matrix of dimensions channels x samples in various 3D matrix of dimensions trials x channels x samples.
    The number of 3D matrix is equal to the number of class.
    Return everything inside a dicionary with the key label/name of the classes. If no labels are passed a progressive numeration is used.

    Parameters
    ----------
    data : Numpy matrix of dimensions channels x samples.
         Obtained by the loadDataset100Hz() function.
    cue_position : Numpy vector of length 1 x samples.
         Obtained by the loadDataset100Hz() function.
    labels : Numpy vector of length 1 x trials
        Obtained by the loadDataset100Hz() function.
    fs : int/double.
        Sample frequency.
    class_label : string list, optional
        List of string with the name of the class. Each string is the name of 1 class. The default is ['1', '2'].

    Returns
    -------
    trials_dict : dictionair
        Diciotionary with jey the various label of the data.

    """
    
    trials_dict = {}
    
    windows_sample = np.linspace(int(0.5 * fs), int(2.5 * fs) - 1, int(2.5 * fs) - int(0.5 * fs)).astype(int)
    n_windows_sample = len(windows_sample)
    
    n_channels = data.shape[0]
    labels_codes = np.unique(labels)
    
    if(class_label == None): class_label = np.linspace(1, len(labels_codes), len(labels_codes))
    
    for label, label_code in zip(class_label, labels_codes):
        # print(label)
        
        # Vector with the initial samples of the various trials related to that class
        class_event_sample_position = cue_position[labels == label_code]
        
        # Create the 3D matrix to contain all the trials of that class. The structure is n_trials x channel x n_samples
        trials_dict[label] = np.zeros((len(class_event_sample_position), n_channels, n_windows_sample))
        
        for i in range(len(class_event_sample_position)):
            event_start = class_event_sample_position[i]
            trials_dict[label][i, :, :] = data[:, windows_sample + event_start]
            
    return trials_dict

#%% Signal related Functions

def PSDEvaluation(trials_matrix, fs = 1):
    """
    Evaluate the PSD (Power Spectral Density) for a matrix for trial. 

    Parameters
    ----------
    trials_matrix : numpy matrix
        Numpy matrix with the various EEG trials. The dimensions of the matrix must be n_trial x n_channel x n_samples
    fs : int/double, optional
        Frequency Sampling. The default is 1.

    Returns
    -------
    PSD_trial : numpy matrix
        Numpy matrix with the PSD of the various EEG trials. The dimensions of the matrix will be n_trial x n_channel x (n_samples / 2 + 1).
    freq_list : python list
         A list containing the frequencies for which the PSD was computed. 

    """
    
    PSD_trial = np.zeros((trials_matrix.shape[0], trials_matrix.shape[1], int(trials_matrix.shape[2] / 2) + 1))
    freq_list = []
    windows_size = len(trials_matrix[0,0,:])
    
    for trial, i in zip(trials_matrix, range(trials_matrix.shape[0])):
        tmp_list = []
        for channel, j in zip(trial, range(trial.shape[0])):
            freq, PSD = scipy.signal.welch(channel, fs = fs, noverlap = 0, nfft = windows_size, nperseg = windows_size)
                
            PSD_trial[i,j, :] = PSD
        
        freq_list.append(freq)
            
    return PSD_trial, freq_list

def bandFilterTrials(trials_matrix, fs, low_f, high_f, filter_order = 3):
    """
    Applying a pass-band fitlering to the data. The filter implementation was done with scipy.signal

    Parameters
    ----------
    trials_matrix : numpy matrix
        Numpy matrix with the various EEG trials. The dimensions of the matrix must be n_trial x n_channel x n_samples
    fs : int/double
        Frequency sampling.
    low_f : int/double
        Low band of the pass band filter.
    high_f : int/double
        High band of the pass band filter..
    filter_order : int, optional
        Order of the filter. The default is 3.

    Returns
    -------
    filter_trails_matrix : numpy matrix
         Numpy matrix with the various filtered EEG trials. The dimensions of the matrix must be n_trial x n_channel x n_samples.

    """
    
    # Evaluate low buond and high bound in the [0, 1] range
    low_bound = low_f / (fs/2)
    high_bound = high_f / (fs/2)
    
    # Check input data
    if(low_bound < 0): low_bound = 0
    if(high_bound > 1): high_bound = 1
    if(low_bound > high_bound): low_bound, high_bound = high_bound, low_bound
    if(low_bound == high_bound): low_bound, high_bound = 0, 1
    
    b, a = scipy.signal.butter(filter_order, [low_bound, high_bound], 'bandpass')
    
    # Check to work ho filtfilt work on N-dimensional array
    # filt_1 = scipy.signal.filtfilt(b, a, trials_matrix)
    # filt_2 = np.zeros(trials_matrix.shape)
    # for i in range(trials_matrix.shape[0]):
    #     for j in range(trials_matrix.shape[1]):
    #         filt_2[i, j, :] = scipy.signal.filtfilt(b, a, trials_matrix[i,j, :])
            
    # return filt_1, filt_2
    
    return scipy.signal.filtfilt(b, a, trials_matrix)

def logVarEvaluation(trials):
    """
    Evaluate the log (logarithm) var (variance) of the trial matrix along the samples axis.
    The sample axis is the axis number 2, counting axis as 0,1,2. 

    Parameters
    ----------
    trials : numpy 3D-matrix
        Trial matrix. The dimensions must be trials x channel x samples

    Returns
    -------
    features : Numpy 2D-matrix
        Return the features matrix. DImension will be trials x channel

    """
    features = np.var(trials, 2)
    features = np.log(features)
    
    return features
    
#%% CSP Related Function

def trialCovariance(trials):
    """
    Calculate the covariance for each trial and return their average

    Parameters
    ----------
    trials : numpy 3D-matrix
        Trial matrix. The dimensions must be trials x channel x samples

    Returns
    -------
    mean_cov : Numpy matrix
        Mean of the covariance alongside channels.

    """
    
    n_trials, n_channels, n_samples = trials.shape
    
    covariance_matrix = np.zeros((n_trials, n_channels, n_channels))
    
    for i in range(trials.shape[0]):
        trial = trials[i, :, :]
        covariance_matrix[i, :, :] = np.cov(trial)
        
    mean_cov = np.mean(covariance_matrix, 0)
        
    return mean_cov

def whitening(sigma, mode = 2):
    """
    Calculate the whitening matrix for the input matrix sigma

    Parameters
    ----------
    sigma : Numpy square matrix
        Input matrix.
    mode : int, optional
        Select how to evaluate the whitening matrix. The default is 1.

    Returns
    -------
    x : Numpy square matrix
        Whitening matrix.
    """
    [u, s, vh] = np.linalg.svd(sigma)
    
      
    if(mode != 1 and mode != 2): mode == 1
    
    if(mode == 1):
        # Whitening constant: prevents division by zero
        epsilon = 1e-5
        
        # ZCA Whitening matrix: U * Lambda * U'
        x = np.dot(u, np.dot(np.diag(1.0/np.sqrt(s + epsilon)), u.T))
    else:
        # eigenvalue decomposition of the covariance matrix
        d, V = np.linalg.eigh(sigma)
        fudge = 10E-18
     
        # A fudge factor can be used so that eigenvectors associated with
        # small eigenvalues do not get overamplified.
        D = np.diag(1. / np.sqrt(d+fudge))
     
        # whitening matrix
        x = np.dot(np.dot(V, D), V.T)
        
    return x

def evaluateW(trials_1, trials_2):
    """
    Evaluate the spatial filter of the CSP algorithm

    Parameters
    ----------
    trials_1 : numpy 3D-matrix
        Trials matrix of class 1. The dimensions must be trials x channel x samples
    trials_2 : numpy 3D-matrix
        Trials matrix of class 2. The dimensions must be trials x channel x samples

    Returns
    -------
    W : numpy 2D-matrix
        Spatial fitler matrix.

    """
    
    # Evaluate covariance matrix for the two classes
    cov_1 = trialCovariance(trials_1)
    cov_2 = trialCovariance(trials_2)
    R = cov_1 + cov_2
    
    # Evaluate whitening matrix
    P = whitening(R)
    
    # The mean covariance matrices may now be transformed
    cov_1_white = np.dot(P, np.dot(cov_1, np.transpose(P)))
    cov_2_white = np.dot(P, np.dot(cov_2, np.transpose(P)))
    
    # CSP requires the eigenvalues and eigenvector be sorted in descending order
    # Find and sort the generalized eigenvalues and eigenvector
    E1, U1 = la.eig(cov_1_white, cov_2_white)
    ord1 = np.argsort(E1)
    ord1 = ord1[::-1]
    E1 = E1[ord1]
    U1 = U1[:, ord1]
    
    # The projection matrix (the spatial filter) may now be obtained
    W = np.dot(np.transpose(U1), P)
    
    return W


def spatialFilteringW(trials, W):
    trials_csp = np.zeros(trials.shape)
    
    for i in range(trials.shape[0]):
        trials_csp[i, :, :] = W.dot(trials[i, :, :])
        
    return trials_csp

#%% Classification function

def trainClassifier(features_1, features_2, train_ratio = 0.75, classifier = None):
    
    # Save both features in a single data matrix
    data_matrix = np.zeros((features_1.shape[0] + features_2.shape[0], features_1.shape[1]))
    data_matrix[0:features_1.shape[0], :] = features_1
    data_matrix[0:features_2.shape[0], :] = features_2
    
    # Create the label vector
    label = np.zeros(data_matrix.shape[0])
    label[0:features_1.shape[0]] = 1
    label[features_1.shape[0]:] = 2
    
    # Shuffle the data
    perm = np.random.permutation(len(label))
    label = label[perm]
    data_matrix = data_matrix[perm, :]
    
    # Select the portion of data used during training
    if(train_ratio <= 0 or train_ratio >= 1): train_ratio = 0.75
    index_training = int(data_matrix.shape[0] * train_ratio)
    train_data = data_matrix[0:index_training, :]
    train_label = label[0:index_training]
    test_data = data_matrix[index_training:, :]
    test_label = label[index_training:]
    
    # Select classifier
    if(classifier == None): classifier = LDA()
    else: classifier = classifier
    
    # Train Classifier
    classifier.fit(train_data, train_label)
    print("Accuracy on TRAIN set: ", classifier.score(train_data, train_label))
    
    # Test parameters
    print("Accuracy on TEST set: ", classifier.score(test_data, test_label))
    
    return classifier
    

#%% Plot Functions

def plotPSD_V1(PSD_matrix, trial_idx, ch_idx, freq_vector = None):
    """
    Plot for a single class of PSD

    Parameters
    ----------
    PSD_matrix : numpy matrix of dimensions n_trial x n_channel x (n_samples / 2 + 1).
    trial_idx : int 
        Trial Index.
    ch_idx : int or list/vector
        Channel index(indeces) to plot.
    freq_vector : vector, optional. 
        Frequencies for the x axis. The default is None.

    """
    if(type(ch_idx) == int):
        plt.figure(figsize = (15, 10))
        if(freq_vector != None): plt.plot(freq_vector[trial_idx], PSD_matrix[trial_idx, ch_idx, :])
        else: plt.plot(PSD_matrix[trial_idx, ch_idx, :]) 
        plt.xlabel('Frequency [Hz]')
        plt.title('Channel N.' + str(ch_idx))
    else:
        fig, axs = plt.subplots(len(ch_idx), 1, figsize = (15, len(ch_idx) * 6))
       
        for ax, ch in zip(axs, ch_idx):
            if(freq_vector != None): ax.plot(freq_vector[trial_idx], PSD_matrix[trial_idx, ch, :])
            else: ax.plot(PSD_matrix[trial_idx, ch, :]) 
            ax.set_xlabel('Frequency [Hz]')
            ax.set_title('Channel N.' + str(ch))
            
        fig.tight_layout()
        

def plotPSD_V2(PSD_matrix_1, PSD_matrix_2, trial_idx, ch_idx, freq_vector = None):
    """
    Plot for a two classes of PSD

    Parameters
    ----------
    PSD_matrix_1 : numpy matrix of dimensions n_trial x n_channel x (n_samples / 2 + 1).
    PSD_matrix_2 : numpy matrix of dimensions n_trial x n_channel x (n_samples / 2 + 1).
    trial_idx : int 
        Trial Index.
    ch_idx : int or list/vector
        Channel index(indeces) to plot.
    freq_vector : vector, optional. 
        Frequencies for the x axis. The default is None.

    """
    if(type(ch_idx) == int):
        plt.figure(figsize = (15, 10))
        if(freq_vector != None): 
            plt.plot(freq_vector[trial_idx], PSD_matrix_1[trial_idx, ch_idx, :])
            plt.plot(freq_vector[trial_idx], PSD_matrix_2[trial_idx, ch_idx, :])
        else: 
            plt.plot(PSD_matrix_1[trial_idx, ch_idx, :])
            plt.plot(PSD_matrix_2[trial_idx, ch_idx, :]) 
        
        plt.xlabel('Frequency [Hz]')
        plt.title('Channel N.' + str(ch_idx))
    else:
        fig, axs = plt.subplots(len(ch_idx), 1, figsize = (15, len(ch_idx) * 6))
       
        for ax, ch in zip(axs, ch_idx):
            if(freq_vector != None): 
                ax.plot(freq_vector[trial_idx], PSD_matrix_1[trial_idx, ch, :])
                ax.plot(freq_vector[trial_idx], PSD_matrix_2[trial_idx, ch, :])
            else: 
                ax.plot(PSD_matrix_1[trial_idx, ch, :]) 
                ax.plot(PSD_matrix_2[trial_idx, ch, :]) 
            ax.set_xlabel('Frequency [Hz]')
            ax.set_title('Channel N.' + str(ch))
            
        fig.tight_layout()
        
        
def plotTrial(trials_matrix, trial_idx, ch_idx):
    plt.figure(figsize = (15, 10))
    plt.plot(trials_matrix[trial_idx, ch_idx, :])
    plt.title("Trials n." + str(trial_idx) + " channel n." + str(ch_idx))
    plt.xlabel("Samples")
    plt.ylabel("Micro-Volt")
    
def plotFeatures(features_1, features_2, width  = 0.3):
    x1 = np.linspace(1, features_1.shape[1], features_1.shape[1])
    x2 = x1 + 0.35
    
    y1 = np.mean(features_1, 0)
    y2 = np.mean(features_2, 0)
    
    fig, ax = plt.subplots(figsize = (15, 10))
    ax.bar(x1, y1, width = width, color = 'b', align='center')
    ax.bar(x2, y2, width = width, color = 'r', align='center')
    ax.set_xlim(0.5, 59.5)
    
def plotFeaturesScatter(features_1, features_2):
    fig, ax = plt.subplots(figsize = (15, 10))
    ax.scatter(features_1[:, 1], features_1[:, -1], color = 'b')
    ax.scatter(features_2[:, 1], features_2[:, -1], color = 'r')
    
    ax.set_xlabel('Last Component')
    ax.set_ylabel('First Component')

      

#%% Other

def cleanWorkspaec():
    try:
        from IPython import get_ipython
        # get_ipython().magic('clear')
        get_ipython().magic('reset -f')
    except:
        pass

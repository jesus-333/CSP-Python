# -*- coding: utf-8 -*-
"""
Contain the implementation of the CSP algorithm. Developed for the train part of dataset IV-1-a of BCI competition.
This version (V2) implement the algorithm for data with two classes.

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

#%%
from CSP_support_function import cleanWorkspaec
# cleanWorkspaec()

#%%
import numpy as np

import matplotlib.pyplot as plt

import scipy.signal
import scipy.linalg as la

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#%%

class CSP():
    
    def __init__(self, data_dict, fs, band = (8, 15)):
        self.fs = fs
        self.train_sklearn = False
        self.train_LDA = False
        self.trials_dict = data_dict
        
        #Filter the data
        self.filt_trial_dict = {}
        for key in self.trials_dict.keys(): 
            self.filt_trial_dict[key] = self.bandFilterTrials(self.trials_dict[key], band[0], band[1], filter_order = 3)
            
        # CSP filter evaluation
        self.W = self.evaluateW()
        
        # Spatial filtering and features evaluation
        self.features_dict = {}
        for key in self.trials_dict.keys():
            tmp_trial = self.spatialFilteringW(self.filt_trial_dict[key])
            self.features_dict[key] = self.logVarEvaluation(tmp_trial)
            
        
    
    def bandFilterTrials(self, trials_matrix, low_f, high_f, filter_order = 3):
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
        low_bound = low_f / (self.fs/2)
        high_bound = high_f / (self.fs/2)
        
        # Check input data
        if(low_bound < 0): low_bound = 0
        if(high_bound > 1): high_bound = 1
        if(low_bound > high_bound): low_bound, high_bound = high_bound, low_bound
        if(low_bound == high_bound): low_bound, high_bound = 0, 1
        
        b, a = scipy.signal.butter(filter_order, [low_bound, high_bound], 'bandpass')
          
        return scipy.signal.filtfilt(b, a, trials_matrix)
    
    
    def logVarEvaluation(self, trials):
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
    
    def trialCovariance(self, trials):
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
    
    def whitening(self, sigma, mode = 2):
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
         
            # A fudge factor can be used so that eigenvectors associated with small eigenvalues do not get overamplified.
            D = np.diag(1. / np.sqrt(d+fudge))
         
            # whitening matrix
            x = np.dot(np.dot(V, D), V.T)
            
        return x
    
    def evaluateW(self):
        """
        Evaluate the spatial filter of the CSP algorithm
    
        Returns
        -------
        W : numpy 2D-matrix
            Spatial fitler matrix.
    
        """
        keys = list(self.filt_trial_dict.keys())
        trials_1 = self.filt_trial_dict[keys[0]]
        trials_2 = self.filt_trial_dict[keys[1]]
        
        # Evaluate covariance matrix for the two classes
        cov_1 = self.trialCovariance(trials_1)
        cov_2 = self.trialCovariance(trials_2)
        R = cov_1 + cov_2
        
        # Evaluate whitening matrix
        P = self.whitening(R)
        
        # The mean covariance matrices may now be transformed
        cov_1_white = np.dot(P, np.dot(cov_1, np.transpose(P)))
        cov_2_white = np.dot(P, np.dot(cov_2, np.transpose(P)))
        
        # Since CSP requires the eigenvalues and eigenvector be sorted in descending order we find and sort the generalized eigenvalues and eigenvector
        E, U = la.eig(cov_1_white, cov_2_white)
        order = np.argsort(E)
        order = order[::-1]
        E = E[order]
        U = U[:, order]
        
        # The projection matrix (the spatial filter) may now be obtained
        W = np.dot(np.transpose(U), P)
        
        return W
    
    
    def spatialFilteringW(self, trials):
        trials_csp = np.zeros(trials.shape)
        
        for i in range(trials.shape[0]):
            trials_csp[i, :, :] = (self.W).dot(trials[i, :, :])
            
        return trials_csp
    
    def trialPSDEvaluation(self, trials_matrix, trial_idx):
        """
        Evaluate the PSD (Power Spectral Density) for a single trial. 
    
        Parameters
        ----------
        trial_label_class : string
            String with the name of the class (label)
        trial_idx : int
            Index of the trial.
    
        Returns
        -------
        PSD_trial : numpy matrix
            Numpy matrix with the PSD of the various EEG trials. The dimensions of the matrix will be n_channel x (n_samples / 2 + 1).
        freq_list : python list
             A list containing the frequencies for which the PSD was computed. 
    
        """

        trial = trials_matrix[trial_idx, :, :]
        PSD_trial = np.zeros((trials_matrix.shape[1], int(trials_matrix.shape[2] / 2) + 1))
        freq_list = []
        windows_size = len(trials_matrix[0, 0, :])
        
        for channel, i in zip(trial, range(trial.shape[0])):
            freq, PSD = scipy.signal.welch(channel, fs = self.fs, noverlap = 0, nfft = windows_size, nperseg = windows_size)
            PSD_trial[i, :] = PSD
            freq_list.append(freq)
            
        return PSD_trial, freq_list
    
    def trainClassifier(self, n_features = 1, train_ratio = 0.75, classifier = None, print_var = False):
        """
        Divide the data in train set and test set and used the data to train the classifier.

        Parameters
        ----------
        n_features : int, optional
            The number of mixture channel to use in the classifier. It must be even and at least as big as 2. The default is 2.
        train_ratio : doble, optional
            The proportion of the data to used as train dataset. The default is 0.75.
        classifier : sklearnn classifier, optional
            Classifier used for the problem. It must be a sklearn classifier. If no classfier was provided the fucntion use the LDA classifier.

        Returns
        -------
        self.classifier : sklearn classifier trained on the data.

        """
        self.n_features = n_features
        
        features_1, features_2 = self.extractFeatures(n_features)
        self.feat_1 = features_1; self.feat_2 = features_2
    
        # Save both features in a single data matrix
        data_matrix = np.zeros((features_1.shape[0] + features_2.shape[0], features_1.shape[1]))
        data_matrix[0:features_1.shape[0], :] = features_1
        data_matrix[features_1.shape[0]:, :] = features_2
        self.tmp_data_matrix = data_matrix
        
        # Create the label vector
        label = np.zeros(data_matrix.shape[0])
        label[0:features_1.shape[0]] = -1
        label[features_1.shape[0]:] = 1
        self.tmp_label = label
        
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
        self.tmp_train = [train_data, train_label]
        self.tmp_test = [test_data, test_label]
        
        # Select classifier
        if(classifier == None): self.classifier = LDA()
        else: self.classifier = classifier
        
        # Train Classifier
        self.classifier.fit(train_data, train_label)
        if(print_var): print("Accuracy on TRAIN set: ", self.classifier.score(train_data, train_label))
        
        # Test parameters
        if(print_var): print("Accuracy on TEST set: ", self.classifier.score(test_data, test_label), "\n")
        self.train_sklearn = True
        
        return self.classifier
    
    def extractFeatures(self, n_features = 1, features = np.zeros(0)):
        """
        Extract the first n_features and the last n_features

        Parameters
        ----------
        n_features : int, optional
            Number of features to extract from each side. The default is 2.
            
        features: numpy array, optional
            When provided the function extract the features from this vector. This parameter is only provided when the function is used to classify unknow trial thorugh the evaluation() function.

        """
        
        if(features.shape[0] == 0):
            self.n_features = n_features
            
            keys = list(self.features_dict.keys())
            features_1_tmp = self.features_dict[keys[0]]
            features_2_tmp = self.features_dict[keys[1]]
            
            if(n_features < 1 or n_features > features_1_tmp.shape[1]): n_features = 1
            
            features_1 = np.zeros((features_1_tmp.shape[0], n_features * 2))
            features_2 = np.zeros((features_2_tmp.shape[0], n_features * 2))
        else:
            features_1 = np.zeros((1, n_features * 2))
            features_1_tmp = features

        for i in range(n_features):
            features_1[:, i] = features_1_tmp[:, i]
            features_1[:, -(i + 1)] = features_1_tmp[:, -(i + 1)]
            if(features.shape[0] == 0):
                features_2[:, i] = features_2_tmp[:, i]
                features_2[:, -(i + 1)] = features_2_tmp[:, -(i + 1)]
        
        if(features.shape[0] == 0):
            return features_1, features_2
        else: 
            return features_1
    
    
    def trainLDA(self, n_features = 1):
        """
        Hand-made implementation of the LDA classifier

        Parameters
        ----------
        n_features : int, optional
            The number of mixture channel to use in the classifier. It must be even and at least as big as 2. The default is 2.
        """
        features_1, features_2 = self.extractFeatures(n_features)
        
        m1 = np.mean(features_1, 0)
        m2 = np.mean(features_2, 0)
        
        den = np.linalg.inv(np.cov(features_1.T) + np.cov(features_2.T))
        
        self.W1 = np.dot((m2 - m1), den)
        self.b1 = np.dot((m1 + m2), self.W1) / 2
        
        self.train_LDA = True
        
        
    def evaluate(self, trial, filt = True, mode = 1):
        """
        Evalaute a single trial. The dimension of the input must be "n. channels x n.samples"

        Parameters
        ----------
        trial : Numpy 2D matrix
            Input trial to evaluate.
        filt : Boolean, optional
            If true the siganl will be band-pass filtered between 8 and 15 Hertz. Otherwise the raw signal will be used. The default is True.
        mode : int, optional
            Decide if use the sklearn classifier or the handmade classifier. 1 for sklearn, 2 for the handmade The default is 1.

        Returns
        -------
        y : int
            Label of trial. The value of the two label can be "1" or "-1"

        """
        
        if(filt): 
            tmp_trial = np.expand_dims(trial, axis = 0)
            trial = self.bandFilterTrials(tmp_trial, 8, 15)
        
        CSP_filtered_trial = np.expand_dims((self.W).dot(trial[0, :, :]), axis = 0)
        features = self.logVarEvaluation(CSP_filtered_trial)
        features = self.extractFeatures(n_features = self.n_features, features = features)

        if(mode == 1 and self.train_sklearn):
            y = self.classifier.predict(features)[0]
        elif(mode == 2 and self.train_LDA):
            y = np.sign((self.W1).dot(np.squeeze(features)) + self.b1)
        else:
            y = self.classifier.predict(features)[0]
            
        return int(y)
    
    def plotTrial(self, trial_label_class, trial_idx, ch_idx, figsize = (15, 10)):
        trials_matrix = self.trials_dict[trial_label_class]
        plt.figure(figsize = figsize)
        plt.plot(trials_matrix[trial_idx, ch_idx, :])
        plt.title("Trials n." + str(trial_idx) + " channel n." + str(ch_idx))
        plt.xlabel("Samples")
        plt.ylabel("Micro-Volt")
        
    def plotPSD(self, trial_idx, ch_idx):
        """
        Plot for a two classes of PSD
    
        Parameters
        ----------
        trial_idx : int 
            Trial Index.
        ch_idx : int or list/vector
            Channel index(indeces) to plot.
        freq_vector : vector, optional. 
            Frequencies for the x axis. The default is None.
    
        """
        
        PSD_list = []
        for key in self.filt_trial_dict.keys():
            PSD, freq = self.trialPSDEvaluation(self.filt_trial_dict[key], trial_idx)
            PSD_list.append(PSD)
            
        PSD_matrix_1 = PSD_list[0]
        PSD_matrix_2 = PSD_list[1]
        print(PSD_matrix_1.shape)
        
        if(type(ch_idx) == int):
            plt.figure(figsize = (15, 10))

            plt.plot(PSD_matrix_1[ch_idx, :])
            plt.plot(PSD_matrix_2[ch_idx, :]) 
            
            plt.xlabel('Frequency [Hz]')
            plt.title('Channel N.' + str(ch_idx))
        else:
            fig, axs = plt.subplots(len(ch_idx), 1, figsize = (15, len(ch_idx) * 6))
           
            for ax, ch in zip(axs, ch_idx):
                ax.plot(PSD_matrix_1[ch, :]) 
                ax.plot(PSD_matrix_2[ch, :]) 
                ax.set_xlabel('Frequency [Hz]')
                ax.set_title('Channel N.' + str(ch))
                
            fig.tight_layout()
        
            
    def plotFeatures(self, width  = 0.3):
        keys = list(self.features_dict.keys())
        features_1 = self.features_dict[keys[0]]
        features_2 = self.features_dict[keys[1]]
        
        x1 = np.linspace(1, features_1.shape[1], features_1.shape[1])
        x2 = x1 + 0.35
        
        # Mean through trials of all the features
        y1 = np.mean(features_1, 0)
        y2 = np.mean(features_2, 0)
        
        fig, ax = plt.subplots(figsize = (15, 10))
        ax.bar(x1, y1, width = width, color = 'b', align='center')
        ax.bar(x2, y2, width = width, color = 'r', align='center')
        ax.set_xlim(0.5, 59.5)
        
        # print(y1.shape, features_1.shape)
        
        
    def plotFeaturesScatter(self, path = None, name = None):
        keys = list(self.features_dict.keys())
        features_1 = self.features_dict[keys[0]]
        features_2 = self.features_dict[keys[1]]
        
        fig, ax = plt.subplots(figsize = (15, 10))
        ax.scatter(features_1[:, 0], features_1[:, -1], color = 'b')
        ax.scatter(features_2[:, 0], features_2[:, -1], color = 'r')
        
        if(self.train_sklearn == True and self.n_features == 1 and
           (self.classifier.__class__.__name__ == 'LinearDiscriminantAnalysis' or (self.classifier.__class__.__name__ == 'SVC' and self.classifier.kernel == 'linear'))
           ):
            coef = self.classifier.coef_
            bias = self.classifier.intercept_[0]
            
            min_x = min(min(features_1[:, 0]), min(features_2[:, 0]))
            max_x = max(max(features_1[:, 0]), max(features_2[:, 0]))
            x = np.linspace(min_x, max_x)
            
            y1 = - (bias + coef[0, 0] * x) / coef[0, 1]
            
            ax.plot(x, y1, color = 'k')
            # print(coef, bias)
            
        if(self.train_LDA and self.n_features == 1):
            min_x = min(min(features_1[:, 0]), min(features_2[:, 0]))
            max_x = max(max(features_1[:, 0]), max(features_2[:, 0]))
            x = np.linspace(min_x, max_x)
            y2 = (self.b1 - self.W1[0] * x) / self.W1[1]
            
            ax.plot(x, y2, color = 'g')
        
        ax.set_xlabel('Last Component')
        ax.set_ylabel('First Component')
        
        if(path != None and name != None):
            fig.savefig(path + '/' + str(name) + '.png')
    

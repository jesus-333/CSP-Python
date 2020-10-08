# -*- coding: utf-8 -*-
"""
Contain the implementation of the FBCSP algorithm. Developed for the train part of dataset IV-1-a of BCI competition.

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

#%%
from CSP_support_function import cleanWorkspaec
# cleanWorkspaec()

#%%
from CSP_support_function import loadDataset100Hz, computeTrial
from CSP import CSP

import numpy as np

from sklearn.svm import SVC

# x1 = np.linspace(1, sx_0.shape[0], sx_0.shape[0])
# x2 = x1 + 0.35

# # Mean through trials of all the features
# y1 = sx_0
# y2 = dx_0

# fig, ax = plt.subplots(figsize = (15, 10))
# ax.bar(x1, y1, width = 0.3, color = 'b', align='center')
# ax.bar(x2, y2, width = 0.3, color = 'r', align='center')
# ax.set_xlim(0.5, 59.5)

#%%
tmp_string = 'abcdefg'
# tmp_string = 'f'

for idx in tmp_string:

    #%% Data load
    
    path = 'Dataset/D1_100Hz/Train/BCICIV_calib_ds1'
    # path = 'Dataset/D1_100Hz/Test/BCICIV_eval_ds1'
    # idx = 'a'
    
    plot_var = False
    
    data, labels, cue_position, other_info = loadDataset100Hz(path, idx, type_dataset = 'train')
    
    #%% Extract trials from data (Works only on dataset IV-1-a of BCI competition)
    
    fs = other_info['sample_rate']
    
    trials_dict = computeTrial(data, cue_position, labels, fs,other_info['class_label'])
    
    #%%
    
    CSP_clf = CSP(trials_dict, fs)
    
    # CSP_clf.plotFeatures()
    # CSP_clf.plotPSD(15, 12)
    
    CSP_clf.trainClassifier()
    CSP_clf.trainLDA()
    
    # CSP_clf.trainClassifier(classifier = SVC(kernel = 'linear'))
    
    CSP_clf.plotFeaturesScatter('Plot', idx)
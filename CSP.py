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
from CSP_support_function import PSDEvaluation, plotPSD_V1, plotPSD_V2
from CSP_support_function import bandFilterTrials, plotTrial
from CSP_support_function import logVarEvaluation, evaluateW, evaluateW_V2, spatialFilteringW, plotFeatures, plotFeaturesScatter

import numpy as np

#%% Data load

path = 'Dataset/D1_100Hz/Train/BCICIV_calib_ds1'
# path = 'Dataset/D1_100Hz/Test/BCICIV_eval_ds1'
idx = 'c'

plot_var = False

data, labels, cue_position, other_info = loadDataset100Hz(path, idx, type_dataset = 'train')

#%% Extract trials from data (Works only on dataset IV-1-a of BCI competition)

fs = other_info['sample_rate']

trials_dict = computeTrial(data, cue_position, labels, fs,other_info['class_label'])

if(plot_var): plotTrial(trials_dict[other_info['class_label'][0]], 1, 1)
        
#%% PSD Evaluation and Visualization (Pre filtering)

psd_class_1, freq_class_1 = PSDEvaluation(trials_dict[other_info['class_label'][0]], fs = fs)
psd_class_2, freq_class_2 = PSDEvaluation(trials_dict[other_info['class_label'][1]], fs = fs)

if(plot_var): plotPSD_V1(psd_class_1, 1, [2,5,4], freq_class_1)
if(plot_var): plotPSD_V2(psd_class_1, psd_class_2, 1, [2,5,4], freq_class_1)


#%% Filtering of the data and PSD new evalaution

filt_1 = bandFilterTrials(trials_dict[other_info['class_label'][0]], fs, 8, 15)
filt_2 = bandFilterTrials(trials_dict[other_info['class_label'][1]], fs, 8, 15)

psd_class_1_filter, freq_class_1_filter = PSDEvaluation(filt_1, fs = fs)
psd_class_2_filter, freq_class_2_filter = PSDEvaluation(filt_2, fs = fs)

if(plot_var): plotPSD_V1(psd_class_1_filter, 1, [2,5,4], freq_class_1_filter)
if(plot_var): plotPSD_V2(psd_class_1_filter, psd_class_2_filter, 1, [2,5,4], freq_class_1_filter)

#%% LogVar Evaluation
features_1 = logVarEvaluation(filt_1)
features_2 = logVarEvaluation(filt_2)

if(plot_var): plotFeatures(features_1, features_2)

#%% Spatial filter evaluation

# W = evaluateW(filt_1, filt_2)
W = evaluateW_V2(filt_1, filt_2)
features_1_CSP = logVarEvaluation(spatialFilteringW(filt_1, W))
features_2_CSP = logVarEvaluation(spatialFilteringW(filt_2, W))

# plotFeatures(features_1, features_2)
if(plot_var): plotFeatures(features_1_CSP, features_2_CSP)
plotFeaturesScatter(features_1_CSP, features_2_CSP)
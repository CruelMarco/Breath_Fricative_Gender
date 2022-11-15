# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 14:53:53 2022

@author: Spirelab
"""

import IPython
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pylab import plot, axis, xlabel, ylabel, grid, log10
import scipy.signal
import os
import pandas as pd
from pandas import DataFrame as df
import sklearn
import math
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import random
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/control_mfcc/mfcc_only'

test_dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/control_mfcc/5_sec_chunk_mfcc_stat'

os.chdir(dir)

mfcc_files = os.listdir(dir)

mfcc_file_path = [dir + '/' + j for j in mfcc_files]


for i in mfcc_file_path:
    
    mfcc_df = pd.read_csv(i)
    
    sub_mfcc_df = []
    
    ran_mfcc_df_mean = []
    
    wheeze_chunk_mfcc_df = []
    
    wheeze_chunk_mfcc_df_mean = []
    
    sub_mfcc_df_mean = [] 
    
    wheeze_chunk_mfcc_df_median = []
    
    sub_mfcc_df_median = []
    
    wheeze_chunk_mfcc_df_floor = []
    
    sub_mfcc_df_mode = []
    
    wheeze_chunk_mfcc_df_std = [] 
    
    sub_mfcc_df_std = []
    
    ran_st_idx = random.sample(range(1, len(mfcc_df)-200),10)
    
    
    for k in range(10): 
    
        ran_idx = list(range(ran_st_idx[k],ran_st_idx[k]+250))
        
        ran_mfcc_df = mfcc_df.loc[ran_idx]
        
        wheeze_chunk_mfcc_df = ran_mfcc_df
        
        sub_mfcc_df.append(wheeze_chunk_mfcc_df)
        
        ##########MFCCs Mean##############
        
        wheeze_chunk_mfcc_df_mean = np.array(ran_mfcc_df[['F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12']].mean())
        
        wheeze_chunk_mfcc_df_mean = pd.DataFrame(wheeze_chunk_mfcc_df_mean)
        
        sub_mfcc_df_mean.append(wheeze_chunk_mfcc_df_mean)
        
        ########## MFCC Median #################
        
        wheeze_chunk_mfcc_df_median = np.array(wheeze_chunk_mfcc_df[['F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12']].median())
        
        wheeze_chunk_mfcc_df_median = pd.DataFrame(wheeze_chunk_mfcc_df_median)
        
        sub_mfcc_df_median.append(wheeze_chunk_mfcc_df_median)
        
        ########## MFCC Mode ##################
        
        wheeze_chunk_mfcc_df_floor = np.array(wheeze_chunk_mfcc_df[['F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12']].apply(np.floor))
        
        wheeze_chunk_mfcc_df_mode = stats.mode(wheeze_chunk_mfcc_df_floor)[0].T
        
        wheeze_chunk_mfcc_df_mode = pd.DataFrame(wheeze_chunk_mfcc_df_mode)
        
        sub_mfcc_df_mode.append(wheeze_chunk_mfcc_df_mode)
        
        ########### MFCC SD ###############
        
        wheeze_chunk_mfcc_df_std = np.array(wheeze_chunk_mfcc_df[['F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12']].std())
        
        wheeze_chunk_mfcc_df_std = pd.DataFrame(wheeze_chunk_mfcc_df_std)
        
        sub_mfcc_df_std.append(wheeze_chunk_mfcc_df_std)
    
    df_idx = [str(x) for x in range(10)]    
    
    sub_mfcc_df = pd.concat(sub_mfcc_df)
    
    ############ Mean ###################
    
    sub_mfcc_df_mean = pd.concat(sub_mfcc_df_mean, axis=1).T
    
    sub_mfcc_df_mean.columns = ['F0_mean' , 'F1_mean' , 'F2_mean' , 'F3_mean' , 'F4_mean' , 'F5_mean' , 'F6_mean' , 'F7_mean' , 'F8_mean' , 'F9_mean' , 'F10_mean' , 'F11_mean' , 'F12_mean']
    
    sub_mfcc_df_mean.index = df_idx
    
    ########### Median #################
    
    sub_mfcc_df_median = pd.concat(sub_mfcc_df_median,axis=1).T
    
    sub_mfcc_df_median.columns = ['F0_median' , 'F1_median' , 'F2_median' , 'F3_median' , 'F4_median' , 'F5_median' , 'F6_median' , 'F7_median' , 'F8_median' , 'F9_median' , 'F10_median' , 'F11_median' , 'F12_median']
    
    sub_mfcc_df_median.index = df_idx
    
    ########### Mode ###############
    
    sub_mfcc_df_mode = pd.concat(sub_mfcc_df_mode,axis=1).T
    
    sub_mfcc_df_mode.columns = ['F0_mode' , 'F1_mode' , 'F2_mode' , 'F3_mode' , 'F4_mode' , 'F5_mode' , 'F6_mode' , 'F7_mode' , 'F8_mode' , 'F9_mode' , 'F10_mode' , 'F11_mode' , 'F12_mode']
    
    sub_mfcc_df_mode.index = df_idx
    
    ########### SD ################
    
    sub_mfcc_df_std = pd.concat(sub_mfcc_df_std,axis=1).T
    
    sub_mfcc_df_std.columns = ['F0_std' , 'F1_std' , 'F2_std' , 'F3_std' , 'F4_std' , 'F5_std' , 'F6_std' , 'F7_std' , 'F8_std' , 'F9_std' , 'F10_std' , 'F11_std' , 'F12_std']
    
    sub_mfcc_df_std.index = df_idx
    
    ########## 10x100 Chunk MFCC Stat DF #############
    
    mfcc_stat_df = pd.concat([sub_mfcc_df_mean, sub_mfcc_df_median, sub_mfcc_df_mode, sub_mfcc_df_std ], axis = 1, join = 'inner')
    
    ######### Exporting to csv ##############
    
    file_name = i.split("/")[8]
    
    name = file_name.split("_")[5]
    
    gender = file_name.split("_")[8]
    
    mfcc_stat_df.insert(0,'Name' , name , True)
    
    mfcc_stat_df['Gender'] = gender
    
    new_file_name = file_name[0 : -4] + '_100_random_chunk_stats' + '.csv'
    
    new_file_path = os.path.join(test_dir , new_file_name)
    
    mfcc_stat_df.to_csv(new_file_path)
    
    
    
    #ran_df_mean = pd.concat(ran_mfcc_df_mean, axis=1).T
    
    #ran_df_mean.columns = ['F0_mean' , 'F1_mean' , 'F2_mean' , 'F3_mean' , 'F4_mean' , 'F5_mean' , 'F6_mean' , 'F7_mean' , 'F8_mean' , 'F9_mean' , 'F10_mean' , 'F11_mean' , 'F12_mean']
    
    #sub_mfcc_df_mean.index = df_idx
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 20:50:40 2022

@author: Shaique
"""

import IPython
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from tqdm import tqdm
import os
from scipy.io.wavfile import write
import scipy.signal
from spectrum import aryule
from pylab import plot, axis, xlabel, ylabel, grid, log10
import scipy.signal
from nara_wpe.wpe import wpe
from nara_wpe.wpe import get_power
from nara_wpe.utils import stft, istft, get_stft_center_frequencies
from nara_wpe import project_root
import os
import librosa
import pandas as pd
from pandas import DataFrame as df
import glob, os
import shutil
import json
import sklearn
import shutil
import math
from scipy import stats
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from numpy import loadtxt
import xgboost
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data'

os.chdir(dir)

check_list_csv_dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/Checklist.csv'

check_list_csv = pd.read_csv(check_list_csv_dir, sep = ",")

subject_names = check_list_csv["Name"]

subject_wheeze_idx = check_list_csv["index_col"]

subject_gender = check_list_csv["Gender"]

subject_pred = check_list_csv["predvalues"]

subs = [*set(subject_names)]

for i in subs :
    
    sub_idx = [j for j in range(len(subject_names)) if i in subject_names[j]]
    
    sub_name = subject_names[sub_idx[0]]
    
    print(sub_name)
    
    #print(sub_idx)
    
    sub_gen = subject_gender[sub_idx[0]]
    
    print(sub_gen)
    
    sub_pred = subject_pred[sub_idx]
    
    preds = pd.DataFrame(pd.value_counts(np.array(sub_pred), dropna=True), columns = ["Pred_gen"])
    
    preds_top = preds.head()
    
    print(preds_top)
    
    row = [j for j in preds_top.index]
    
    print(row)
    
    # total_wheeze = len(sub_idx)
    
    # male_pred_count = sub_pred.count("M")
    
    # print(male_pred_count)
    
    # female_pred_count = sub_pred.count("F")
    
    # print(female_pred_count)
    
    #preds = pd.value_counts(np.array(sub_pred), dropna=True)
    
    #print(preds)
    
    #sub_accuracy = correct_pred/total_wheeze
    


        

    



# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:52:54 2022

@author: Spirelab
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from numpy import loadtxt
import xgboost
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/control_mfcc/100_consecutive_random_chunk_mfcc_stat/sets'

os.chdir(dir)

sets = os.listdir(dir)

set_path = []

for i in sets:
    
    set_dir = os.path.join(dir,i)
    
    set_path.append(set_dir)
    
set_files = []

for j in set_path:
    
    files = os.listdir(j)
    
    set_files.append(files)
    
list_of_path=[]

for z in range (len(set_path)):
    
    #set_no=os.listdir(ee(z))
    set=set_files[z]
    
    for j in range (len(set_files[z])):
        
        filepath=set_path[z]+'/'+set_files[z][j]
        
        list_of_path.append(filepath)
        
set1=list(filter(lambda k: 'set1' in k, list_of_path))

set2=list(filter(lambda k: 'set2' in k, list_of_path))

set3=list(filter(lambda k: 'set3' in k, list_of_path))

set4=list(filter(lambda k: 'set4' in k, list_of_path))

set5=list(filter(lambda k: 'set5' in k, list_of_path))

value1=[]

for n in range (len(set1)):
    
    table1=pd.read_csv(set1[n],index_col=0)
    
    value1.append(table1)

value2=[]

for n in range (len(set2)):
    
    table2=pd.read_csv(set2[n],index_col=0)
    
    value2.append(table2)

value3=[]

for n in range (len(set3)):
    
    table3=pd.read_csv(set3[n],index_col=0)
    
    value3.append(table3)

value4=[]

for n in range (len(set4)):
    
    table4=pd.read_csv(set4[n],index_col=0)
    
    value4.append(table4)

value5=[]

for n in range (len(set5)):
    
    table5=pd.read_csv(set5[n],index_col=0)
    
    value5.append(table5)

mfcc1=pd.concat(value1)

mfcc2=pd.concat(value2)

mfcc3=pd.concat(value3)

mfcc4=pd.concat(value4)

mfcc5=pd.concat(value5)

mfcc1=mfcc1.sample(frac=1)

mfcc2=mfcc2.sample(frac=1)

mfcc3=mfcc3.sample(frac=1)

mfcc4=mfcc4.sample(frac=1)

mfcc5=mfcc5.sample(frac=1)

trainx_1 = mfcc1.loc[:, mfcc1.columns.drop(['Name','Gender'])]

trainy_1 = mfcc1.loc[:, mfcc1.columns.drop(['Name','F0_mean','F1_mean','F2_mean','F3_mean','F4_mean','F5_mean','F6_mean','F7_mean','F8_mean','F9_mean','F10_mean','F11_mean','F12_mean','F0_median','F1_median','F2_median','F3_median','F4_median','F5_median','F6_median','F7_median','F8_median','F9_median','F10_median','F11_median','F12_median','F0_mode','F1_mode','F2_mode','F3_mode','F4_mode','F5_mode','F6_mode','F7_mode','F8_mode','F9_mode','F10_mode','F11_mode','F12_mode','F0_std','F1_std','F2_std','F3_std','F4_std','F5_std','F6_std','F7_std','F8_std','F9_std','F10_std','F11_std','F12_std'])]

trainx_2 = mfcc2.loc[:, mfcc2.columns.drop(['Name','Gender'])]

trainy_2 = mfcc2.loc[:, mfcc2.columns.drop(['Name','F0_mean','F1_mean','F2_mean','F3_mean','F4_mean','F5_mean','F6_mean','F7_mean','F8_mean','F9_mean','F10_mean','F11_mean','F12_mean','F0_median','F1_median','F2_median','F3_median','F4_median','F5_median','F6_median','F7_median','F8_median','F9_median','F10_median','F11_median','F12_median','F0_mode','F1_mode','F2_mode','F3_mode','F4_mode','F5_mode','F6_mode','F7_mode','F8_mode','F9_mode','F10_mode','F11_mode','F12_mode','F0_std','F1_std','F2_std','F3_std','F4_std','F5_std','F6_std','F7_std','F8_std','F9_std','F10_std','F11_std','F12_std'])]

trainx_3 = mfcc3.loc[:, mfcc3.columns.drop(['Name','Gender'])]

trainy_3 = mfcc3.loc[:, mfcc3.columns.drop(['Name','F0_mean','F1_mean','F2_mean','F3_mean','F4_mean','F5_mean','F6_mean','F7_mean','F8_mean','F9_mean','F10_mean','F11_mean','F12_mean','F0_median','F1_median','F2_median','F3_median','F4_median','F5_median','F6_median','F7_median','F8_median','F9_median','F10_median','F11_median','F12_median','F0_mode','F1_mode','F2_mode','F3_mode','F4_mode','F5_mode','F6_mode','F7_mode','F8_mode','F9_mode','F10_mode','F11_mode','F12_mode','F0_std','F1_std','F2_std','F3_std','F4_std','F5_std','F6_std','F7_std','F8_std','F9_std','F10_std','F11_std','F12_std'])]

trainx_4 = mfcc4.loc[:, mfcc4.columns.drop(['Name','Gender'])]

trainy_4 = mfcc4.loc[:, mfcc4.columns.drop(['Name','F0_mean','F1_mean','F2_mean','F3_mean','F4_mean','F5_mean','F6_mean','F7_mean','F8_mean','F9_mean','F10_mean','F11_mean','F12_mean','F0_median','F1_median','F2_median','F3_median','F4_median','F5_median','F6_median','F7_median','F8_median','F9_median','F10_median','F11_median','F12_median','F0_mode','F1_mode','F2_mode','F3_mode','F4_mode','F5_mode','F6_mode','F7_mode','F8_mode','F9_mode','F10_mode','F11_mode','F12_mode','F0_std','F1_std','F2_std','F3_std','F4_std','F5_std','F6_std','F7_std','F8_std','F9_std','F10_std','F11_std','F12_std'])]

trainx_5 = mfcc5.loc[:, mfcc5.columns.drop(['Name','Gender'])]

trainy_5 = mfcc5.loc[:, mfcc5.columns.drop(['Name','F0_mean','F1_mean','F2_mean','F3_mean','F4_mean','F5_mean','F6_mean','F7_mean','F8_mean','F9_mean','F10_mean','F11_mean','F12_mean','F0_median','F1_median','F2_median','F3_median','F4_median','F5_median','F6_median','F7_median','F8_median','F9_median','F10_median','F11_median','F12_median','F0_mode','F1_mode','F2_mode','F3_mode','F4_mode','F5_mode','F6_mode','F7_mode','F8_mode','F9_mode','F10_mode','F11_mode','F12_mode','F0_std','F1_std','F2_std','F3_std','F4_std','F5_std','F6_std','F7_std','F8_std','F9_std','F10_std','F11_std','F12_std'])]




setx_1_train= pd.concat([trainx_2,trainx_3,trainx_4,trainx_5])

setx_2_train= pd.concat([trainx_1,trainx_3,trainx_4,trainx_5])

setx_3_train= pd.concat([trainx_1,trainx_2,trainx_4,trainx_5])

setx_4_train= pd.concat([trainx_1,trainx_2,trainx_3,trainx_5])

setx_5_train= pd.concat([trainx_1,trainx_2,trainx_3,trainx_4])

setx_1_test= trainx_1

setx_2_test= trainx_2

setx_3_test= trainx_3

setx_4_test= trainx_4

setx_5_test= trainx_5

sety_1_train= pd.concat([trainy_2,trainy_3,trainy_4,trainy_5])

sety_2_train= pd.concat([trainy_1,trainy_3,trainy_4,trainy_5])

sety_3_train= pd.concat([trainy_1,trainy_2,trainy_4,trainy_5])

sety_4_train= pd.concat([trainy_1,trainy_2,trainy_3,trainy_5])

sety_5_train= pd.concat([trainy_1,trainy_2,trainy_3,trainy_4])

sety_1_test= trainy_1

sety_2_test= trainy_2

sety_3_test= trainy_3

sety_4_test= trainy_4

sety_5_test= trainy_5

classifier = KNeighborsClassifier(n_neighbors=7)

#############Fold 1##############


fold1_fit = classifier.fit(setx_1_train, sety_1_train.values.ravel())

y_pred_fold_1 = classifier.predict(setx_1_test)

score_fold_1_test=fold1_fit.score(setx_1_test, sety_1_test)

score_fold_1_train = fold1_fit.score(setx_1_train, sety_1_train)

###########Fold 2################

fold2_fit = classifier.fit(setx_2_train, sety_2_train.values.ravel())

y_pred_fold_2 = classifier.predict(setx_2_test)

score_fold_2_test = fold2_fit.score(setx_2_test, sety_2_test)

score_fold_2_train = fold2_fit.score(setx_2_train, sety_2_train)



#accuracies2 = cross_val_score(estimator = classifier2, X = setx_2_train, y = sety_2_train, cv = None)

###########Fold 3################

fold3_fit = classifier.fit(setx_3_train, sety_3_train.values.ravel())

y_pred_fold_3 = classifier.predict(setx_2_test)

score_fold_3_test = fold3_fit.score(setx_3_test, sety_3_test)

score_fold_3_train = fold3_fit.score(setx_3_train, sety_3_train)

###########Fold 4################

fold4_fit = classifier.fit(setx_4_train, sety_4_train.values.ravel())

y_pred_fold_4 = classifier.predict(setx_4_test)

score_fold_4_test = fold4_fit.score(setx_4_test, sety_4_test)

score_fold_4_train = fold4_fit.score(setx_4_train, sety_4_train)


###########Fold 5################

fold5_fit = classifier.fit(setx_5_train, sety_5_train.values.ravel())

y_pred_fold_5 = classifier.predict(setx_5_test)

score_fold_5_test = fold5_fit.score(setx_5_test, sety_5_test)

score_fold_5_train = fold5_fit.score(setx_5_train, sety_5_train)

print("Test Score is = " , (score_fold_1_test+score_fold_2_test+score_fold_3_test+score_fold_4_test+score_fold_5_test)/5)

print("Train Score is = " , (score_fold_1_train+score_fold_2_train+score_fold_3_train+score_fold_4_train+score_fold_5_train)/5)
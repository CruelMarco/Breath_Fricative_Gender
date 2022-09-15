# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 15:42:49 2022

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from numpy import loadtxt
import xgboost
from keras.models import Model,Sequential
from keras import optimizers
from keras.layers import Input,Conv1D,BatchNormalization,MaxPooling1D,LSTM,Dense,Activation,Layer
#from emodata1d import load_data
from keras.utils import to_categorical
import keras.backend as K
import argparse
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
#from keras.models import load_model

dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/control_mfcc/mfcc_d_dd_sets'

os.chdir(dir)

sets = os.listdir(dir)

set_path_arr = []

for i in sets:
  set_path = os.path.join(dir,i)
  print(set_path)
  set_path_arr.append(set_path)

set1_path = set_path_arr[0]
set2_path = set_path_arr[1]
set3_path = set_path_arr[2]
set4_path = set_path_arr[3]
set5_path = set_path_arr[4]

set1_files = os.listdir(set1_path)
set2_files = os.listdir(set2_path)
set3_files = os.listdir(set3_path)
set4_files = os.listdir(set4_path)
set5_files = os.listdir(set5_path)

set1_files_path = [os.path.join(set1_path,j) for j in set1_files]
set2_files_path = [os.path.join(set2_path,j) for j in set2_files]
set3_files_path = [os.path.join(set3_path,j) for j in set3_files]
set4_files_path = [os.path.join(set4_path,j) for j in set4_files]
set5_files_path = [os.path.join(set5_path,j) for j in set5_files]

set1_mfcc = []
set1_mfcc_df = []
for j in set1_files_path:
  set1 = pd.read_csv(j,sep = ",")
  set1_mfcc.append(set1)
  set1_mfcc_df = pd.concat(set1_mfcc)
  
set2_mfcc = []
set2_mfcc_df = []
for j in set2_files_path:
  set2 = pd.read_csv(j,sep = ",")
  set2_mfcc.append(set2)
  set2_mfcc_df = pd.concat(set2_mfcc)
  
set3_mfcc = []
set3_mfcc_df = []
for j in set3_files_path:
  set3 = pd.read_csv(j,sep = ",")
  set3_mfcc.append(set3)
  set3_mfcc_df = pd.concat(set3_mfcc)
  
set4_mfcc = []
set4_mfcc_df = []
for j in set4_files_path:
  set4 = pd.read_csv(j,sep = ",")
  set4_mfcc.append(set4)
  set4_mfcc_df = pd.concat(set4_mfcc)
  
set5_mfcc = []
set5_mfcc_df = []
for j in set5_files_path:
  set5 = pd.read_csv(j,sep = ",")
  set5_mfcc.append(set5)
  set5_mfcc_df = pd.concat(set5_mfcc)
  
trainx_1 = set1_mfcc_df.loc[:, set1_mfcc_df.columns.drop(['Name','Gender', 'Unnamed: 0'])]

trainy_1 = set1_mfcc_df.loc[:, set1_mfcc_df.columns.drop(['Name','Unnamed: 0','F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F0_d','F1_d','F2_d','F3_d','F4_d','F5_d','F6_d','F7_d','F8_d','F9_d','F10_d','F11_d','F12_d','F0_dd','F1_dd','F2_dd','F3_dd','F4_dd','F5_dd','F6_dd','F7_dd','F8_dd','F9_dd','F10_dd','F11_dd','F12_dd'])]

trainx_2 = set2_mfcc_df.loc[:, set2_mfcc_df.columns.drop(['Name','Gender', 'Unnamed: 0'])]

trainy_2 = set2_mfcc_df.loc[:, set2_mfcc_df.columns.drop(['Name','Unnamed: 0','F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F0_d','F1_d','F2_d','F3_d','F4_d','F5_d','F6_d','F7_d','F8_d','F9_d','F10_d','F11_d','F12_d','F0_dd','F1_dd','F2_dd','F3_dd','F4_dd','F5_dd','F6_dd','F7_dd','F8_dd','F9_dd','F10_dd','F11_dd','F12_dd'])]

trainx_3 = set3_mfcc_df.loc[:, set3_mfcc_df.columns.drop(['Name','Gender', 'Unnamed: 0'])]

trainy_3 = set3_mfcc_df.loc[:, set3_mfcc_df.columns.drop(['Name','Unnamed: 0','F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F0_d','F1_d','F2_d','F3_d','F4_d','F5_d','F6_d','F7_d','F8_d','F9_d','F10_d','F11_d','F12_d','F0_dd','F1_dd','F2_dd','F3_dd','F4_dd','F5_dd','F6_dd','F7_dd','F8_dd','F9_dd','F10_dd','F11_dd','F12_dd'])]

trainx_4 = set4_mfcc_df.loc[:, set4_mfcc_df.columns.drop(['Name','Gender', 'Unnamed: 0'])]

trainy_4 = set4_mfcc_df.loc[:, set4_mfcc_df.columns.drop(['Name','Unnamed: 0','F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F0_d','F1_d','F2_d','F3_d','F4_d','F5_d','F6_d','F7_d','F8_d','F9_d','F10_d','F11_d','F12_d','F0_dd','F1_dd','F2_dd','F3_dd','F4_dd','F5_dd','F6_dd','F7_dd','F8_dd','F9_dd','F10_dd','F11_dd','F12_dd'])]

trainx_5 = set5_mfcc_df.loc[:, set5_mfcc_df.columns.drop(['Name','Gender', 'Unnamed: 0'])]

trainy_5 = set5_mfcc_df.loc[:, set5_mfcc_df.columns.drop(['Name','Unnamed: 0','F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F0_d','F1_d','F2_d','F3_d','F4_d','F5_d','F6_d','F7_d','F8_d','F9_d','F10_d','F11_d','F12_d','F0_dd','F1_dd','F2_dd','F3_dd','F4_dd','F5_dd','F6_dd','F7_dd','F8_dd','F9_dd','F10_dd','F11_dd','F12_dd'])]

setx_1_train= pd.concat([trainx_2,trainx_3,trainx_4,trainx_5])

setx_2_train= pd.concat([trainx_1,trainx_3,trainx_4,trainx_5])

setx_3_train= pd.concat([trainx_1,trainx_2,trainx_4,trainx_5])

setx_4_train= pd.concat([trainx_1,trainx_2,trainx_3,trainx_5])

setx_5_train= pd.concat([trainx_1,trainx_2,trainx_3,trainx_4])

# #########CNN LSTM Model##########

# model = Sequential(name = "Gender_CNN_LSTM")

# model.add(Conv1D(filters = 64, kernel_size = (3), strides = 1 , padding = "same" , data_format = "channels_last",
#                   activation="relu" ))

# model.add()
 

#value1=[]

# for n in range (len(set1)):
    
#     table1=pd.read_csv(set1[n],index_col=0)
    
#     value1.append(table1)

# value2=[]

# for n in range (len(set2)):
    
#     table2=pd.read_csv(set2[n],index_col=0)
    
#     value2.append(table2)

# value3=[]

# for n in range (len(set3)):
    
#     table3=pd.read_csv(set3[n],index_col=0)
    
#     value3.append(table3)

# value4=[]

# for n in range (len(set4)):
    
#     table4=pd.read_csv(set4[n],index_col=0)
    
#     value4.append(table4)

# value5=[]

# for n in range (len(set5)):
    
#     table5=pd.read_csv(set5[n],index_col=0)
    
#     value5.append(table5)

# mfcc1=pd.concat(value1)

# mfcc2=pd.concat(value2)

# mfcc3=pd.concat(value3)

# mfcc4=pd.concat(value4)

# mfcc5=pd.concat(value5)

# mfcc1=mfcc1.sample(frac=1)

# mfcc2=mfcc2.sample(frac=1)

# mfcc3=mfcc3.sample(frac=1)

# mfcc4=mfcc4.sample(frac=1)

# mfcc5=mfcc5.sample(frac=1)

# trainx_1 = mfcc1.loc[:, mfcc1.columns.drop(['Name','Gender'])]

# trainy_1 = mfcc1.loc[:, mfcc1.columns.drop(['Name','F0_mean','F1_mean','F2_mean','F3_mean','F4_mean','F5_mean','F6_mean','F7_mean','F8_mean','F9_mean','F10_mean','F11_mean','F12_mean','F0_median','F1_median','F2_median','F3_median','F4_median','F5_median','F6_median','F7_median','F8_median','F9_median','F10_median','F11_median','F12_median','F0_mode','F1_mode','F2_mode','F3_mode','F4_mode','F5_mode','F6_mode','F7_mode','F8_mode','F9_mode','F10_mode','F11_mode','F12_mode','F0_std','F1_std','F2_std','F3_std','F4_std','F5_std','F6_std','F7_std','F8_std','F9_std','F10_std','F11_std','F12_std'])]

# trainx_2 = mfcc2.loc[:, mfcc2.columns.drop(['Name','Gender'])]

# trainy_2 = mfcc2.loc[:, mfcc2.columns.drop(['Name','F0_mean','F1_mean','F2_mean','F3_mean','F4_mean','F5_mean','F6_mean','F7_mean','F8_mean','F9_mean','F10_mean','F11_mean','F12_mean','F0_median','F1_median','F2_median','F3_median','F4_median','F5_median','F6_median','F7_median','F8_median','F9_median','F10_median','F11_median','F12_median','F0_mode','F1_mode','F2_mode','F3_mode','F4_mode','F5_mode','F6_mode','F7_mode','F8_mode','F9_mode','F10_mode','F11_mode','F12_mode','F0_std','F1_std','F2_std','F3_std','F4_std','F5_std','F6_std','F7_std','F8_std','F9_std','F10_std','F11_std','F12_std'])]

# trainx_3 = mfcc3.loc[:, mfcc3.columns.drop(['Name','Gender'])]

# trainy_3 = mfcc3.loc[:, mfcc3.columns.drop(['Name','F0_mean','F1_mean','F2_mean','F3_mean','F4_mean','F5_mean','F6_mean','F7_mean','F8_mean','F9_mean','F10_mean','F11_mean','F12_mean','F0_median','F1_median','F2_median','F3_median','F4_median','F5_median','F6_median','F7_median','F8_median','F9_median','F10_median','F11_median','F12_median','F0_mode','F1_mode','F2_mode','F3_mode','F4_mode','F5_mode','F6_mode','F7_mode','F8_mode','F9_mode','F10_mode','F11_mode','F12_mode','F0_std','F1_std','F2_std','F3_std','F4_std','F5_std','F6_std','F7_std','F8_std','F9_std','F10_std','F11_std','F12_std'])]

# trainx_4 = mfcc4.loc[:, mfcc4.columns.drop(['Name','Gender'])]

# trainy_4 = mfcc4.loc[:, mfcc4.columns.drop(['Name','F0_mean','F1_mean','F2_mean','F3_mean','F4_mean','F5_mean','F6_mean','F7_mean','F8_mean','F9_mean','F10_mean','F11_mean','F12_mean','F0_median','F1_median','F2_median','F3_median','F4_median','F5_median','F6_median','F7_median','F8_median','F9_median','F10_median','F11_median','F12_median','F0_mode','F1_mode','F2_mode','F3_mode','F4_mode','F5_mode','F6_mode','F7_mode','F8_mode','F9_mode','F10_mode','F11_mode','F12_mode','F0_std','F1_std','F2_std','F3_std','F4_std','F5_std','F6_std','F7_std','F8_std','F9_std','F10_std','F11_std','F12_std'])]

# trainx_5 = mfcc5.loc[:, mfcc5.columns.drop(['Name','Gender'])]

# trainy_5 = mfcc5.loc[:, mfcc5.columns.drop(['Name','F0_mean','F1_mean','F2_mean','F3_mean','F4_mean','F5_mean','F6_mean','F7_mean','F8_mean','F9_mean','F10_mean','F11_mean','F12_mean','F0_median','F1_median','F2_median','F3_median','F4_median','F5_median','F6_median','F7_median','F8_median','F9_median','F10_median','F11_median','F12_median','F0_mode','F1_mode','F2_mode','F3_mode','F4_mode','F5_mode','F6_mode','F7_mode','F8_mode','F9_mode','F10_mode','F11_mode','F12_mode','F0_std','F1_std','F2_std','F3_std','F4_std','F5_std','F6_std','F7_std','F8_std','F9_std','F10_std','F11_std','F12_std'])]




# setx_1_train= pd.concat([trainx_2,trainx_3,trainx_4,trainx_5])

# setx_2_train= pd.concat([trainx_1,trainx_3,trainx_4,trainx_5])

# setx_3_train= pd.concat([trainx_1,trainx_2,trainx_4,trainx_5])

# setx_4_train= pd.concat([trainx_1,trainx_2,trainx_3,trainx_5])

# setx_5_train= pd.concat([trainx_1,trainx_2,trainx_3,trainx_4])

# setx_1_test= trainx_1

# setx_2_test= trainx_2

# setx_3_test= trainx_3

# setx_4_test= trainx_4

# setx_5_test= trainx_5

# sety_1_train= pd.concat([trainy_2,trainy_3,trainy_4,trainy_5])

# sety_2_train= pd.concat([trainy_1,trainy_3,trainy_4,trainy_5])

# sety_3_train= pd.concat([trainy_1,trainy_2,trainy_4,trainy_5])

# sety_4_train= pd.concat([trainy_1,trainy_2,trainy_3,trainy_5])

# sety_5_train= pd.concat([trainy_1,trainy_2,trainy_3,trainy_4])

# sety_1_test= trainy_1

# sety_2_test= trainy_2

# sety_3_test= trainy_3

# sety_4_test= trainy_4

# sety_5_test= trainy_5

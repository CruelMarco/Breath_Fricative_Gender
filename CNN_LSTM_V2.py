# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:35:22 2022

@author: Spirelab
"""

import IPython
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from scipy.io.wavfile import write
import scipy.signal
import librosa
import pandas as pd
from pandas import DataFrame as df
import json
import sklearn
import shutil
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
from keras.layers import Input,Conv1D,BatchNormalization,MaxPooling1D,LSTM,Dense,Activation,Layer, Flatten
from keras.utils import to_categorical
import keras.backend as K
import argparse
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn import preprocessing
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
#from keras.models import load_model

dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/control_mfcc/mfcc_d_dd_sets_w_Val_Set/Sets'

val_set_dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/control_mfcc/mfcc_d_dd_sets_w_Val_Set/val_set'

os.chdir(dir)

sets = os.listdir(dir)

set_path_arr = []

############### TRAIN_TEST_SET #################

for i in sets:
  set_path = os.path.join(dir,i)
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

label_encoder = preprocessing.LabelEncoder()

for j in set1_files_path:
  set1 = pd.read_csv(j,sep = ",")
  set1_mfcc.append(set1)
  set1_mfcc_df = pd.concat(set1_mfcc)

set1_mfcc_df["Gender"] = label_encoder.fit_transform(set1_mfcc_df["Gender"])
  
set2_mfcc = []
set2_mfcc_df = []
for j in set2_files_path:
  set2 = pd.read_csv(j,sep = ",")
  set2_mfcc.append(set2)
  set2_mfcc_df = pd.concat(set2_mfcc)
set2_mfcc_df["Gender"] = label_encoder.fit_transform(set2_mfcc_df["Gender"])
  
set3_mfcc = []
set3_mfcc_df = []
for j in set3_files_path:
  set3 = pd.read_csv(j,sep = ",")
  set3_mfcc.append(set3)
  set3_mfcc_df = pd.concat(set3_mfcc)
set3_mfcc_df["Gender"] = label_encoder.fit_transform(set3_mfcc_df["Gender"])
  
set4_mfcc = []
set4_mfcc_df = []
for j in set4_files_path:
  set4 = pd.read_csv(j,sep = ",")
  set4_mfcc.append(set4)
  set4_mfcc_df = pd.concat(set4_mfcc)
set4_mfcc_df["Gender"] = label_encoder.fit_transform(set4_mfcc_df["Gender"])
  
set5_mfcc = []
set5_mfcc_df = []
for j in set5_files_path:
  set5 = pd.read_csv(j,sep = ",")
  set5_mfcc.append(set5)
  set5_mfcc_df = pd.concat(set5_mfcc)
set5_mfcc_df["Gender"] = label_encoder.fit_transform(set5_mfcc_df["Gender"])
  
#label_encoder = preprocessing.LabelEncoder()
  
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

#################   Model   Fold 1  Preprocessing  ###########

setx_1_train_1 = np.array(setx_1_train)

setx_1_train_1 = setx_1_train_1.reshape(setx_1_train_1.shape[0], setx_1_train_1.shape[1],1)

sety_1_train_1 = np.array(sety_1_train)

setx_1_test_1 = np.array(setx_1_test)

setx_1_test_1 = setx_1_test_1.reshape(setx_1_test_1.shape[0], setx_1_test_1.shape[1],1)


# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 15:42:49 2022

@author: Shaique
"""

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
from keras.layers import Input,Conv1D,BatchNormalization,MaxPooling1D,LSTM,Dense,Activation,Layer, Flatten, Dropout,GlobalAveragePooling1D
from keras_self_attention import SeqSelfAttention
from keras.utils import to_categorical
import keras.backend as K
import argparse
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn import preprocessing
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
#from keras.models import load_model

dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/control_mfcc/mfcc_d_dd_sets'

os.chdir(dir)

sets = os.listdir(dir)

set_path_arr = []

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

val_set1_mfcc_df = set1_mfcc_df.sort_values(by = ["Gender" , "Name"], ascending=True )

#val_set1_mfcc_df = val_set1_mfcc_df.columns.drop('Unnamed: 0')

set1_names = val_set1_mfcc_df["Name"].unique()

val_set1_names = [set1_names[0] , set1_names[len(set1_names)-1]]

val_set1_F = val_set1_mfcc_df[val_set1_mfcc_df["Name"] == val_set1_names[0]]

val_set1_M = val_set1_mfcc_df[val_set1_mfcc_df["Name"] == val_set1_names[1]]

val_set1 = pd.concat([val_set1_F, val_set1_M])

#set1_mfcc_df = val_set1_mfcc_df.drop([val_set1_names[0]])

set2_mfcc = []
set2_mfcc_df = []
for j in set2_files_path:
  set2 = pd.read_csv(j,sep = ",")
  set2_mfcc.append(set2)
  set2_mfcc_df = pd.concat(set2_mfcc)
set2_mfcc_df["Gender"] = label_encoder.fit_transform(set2_mfcc_df["Gender"])

val_set2_mfcc_df = set2_mfcc_df.sort_values(by = ["Gender" , "Name"], ascending=True )

set2_names = val_set2_mfcc_df["Name"].unique()

val_set2_names = [set2_names[0] , set2_names[len(set2_names)-1]]

val_set2_F = val_set2_mfcc_df[val_set2_mfcc_df["Name"] == val_set2_names[0]]

val_set2_M = val_set2_mfcc_df[val_set2_mfcc_df["Name"] == val_set2_names[1]]

val_set2 = pd.concat([val_set2_F, val_set2_M])
  
set3_mfcc = []
set3_mfcc_df = []
for j in set3_files_path:
  set3 = pd.read_csv(j,sep = ",")
  set3_mfcc.append(set3)
  set3_mfcc_df = pd.concat(set3_mfcc)
set3_mfcc_df["Gender"] = label_encoder.fit_transform(set3_mfcc_df["Gender"])

val_set3_mfcc_df = set3_mfcc_df.sort_values(by = ["Gender" , "Name"], ascending=True )

set3_names = val_set3_mfcc_df["Name"].unique()  

val_set3_names = [set3_names[0] , set3_names[len(set3_names)-1]]

val_set3_F = val_set3_mfcc_df[val_set3_mfcc_df["Name"] == val_set3_names[0]]

val_set3_M = val_set3_mfcc_df[val_set3_mfcc_df["Name"] == val_set3_names[1]]

val_set3 = pd.concat([val_set3_F, val_set3_M])

set4_mfcc = []
set4_mfcc_df = []
for j in set4_files_path:
  set4 = pd.read_csv(j,sep = ",")
  set4_mfcc.append(set4)
  set4_mfcc_df = pd.concat(set4_mfcc)
set4_mfcc_df["Gender"] = label_encoder.fit_transform(set4_mfcc_df["Gender"])

val_set4_mfcc_df = set4_mfcc_df.sort_values(by = ["Gender" , "Name"], ascending=True )

set4_names = val_set4_mfcc_df["Name"].unique()

val_set4_names = [set4_names[0] , set4_names[len(set4_names)-1]]

val_set4_F = val_set4_mfcc_df[val_set4_mfcc_df["Name"] == val_set4_names[0]]

val_set4_M = val_set4_mfcc_df[val_set4_mfcc_df["Name"] == val_set4_names[1]]

val_set4 = pd.concat([val_set4_F, val_set4_M])
  
set5_mfcc = []
set5_mfcc_df = []
for j in set5_files_path:
  set5 = pd.read_csv(j,sep = ",")
  set5_mfcc.append(set5)
  set5_mfcc_df = pd.concat(set5_mfcc)
set5_mfcc_df["Gender"] = label_encoder.fit_transform(set5_mfcc_df["Gender"])

val_set5_mfcc_df = set5_mfcc_df.sort_values(by = ["Gender" , "Name"], ascending=True )

set5_names = val_set5_mfcc_df["Name"].unique()

val_set5_names = [set5_names[0] , set5_names[len(set5_names)-1]]

val_set5_F = val_set5_mfcc_df[val_set5_mfcc_df["Name"] == val_set5_names[0]]

val_set5_M = val_set5_mfcc_df[val_set5_mfcc_df["Name"] == val_set5_names[1]]

val_set5 = pd.concat([val_set5_F, val_set5_M])
  
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


validationx_1 = val_set1.loc[:, val_set1.columns.drop(['Name','Gender', 'Unnamed: 0'])]

validationy_1 = val_set1.loc[:, val_set1.columns.drop(['Name','Unnamed: 0','F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F0_d','F1_d','F2_d','F3_d','F4_d','F5_d','F6_d','F7_d','F8_d','F9_d','F10_d','F11_d','F12_d','F0_dd','F1_dd','F2_dd','F3_dd','F4_dd','F5_dd','F6_dd','F7_dd','F8_dd','F9_dd','F10_dd','F11_dd','F12_dd'])]


validationx_2 = val_set2.loc[:, val_set2.columns.drop(['Name','Gender', 'Unnamed: 0'])]

validationy_2 = val_set2.loc[:, val_set2.columns.drop(['Name','Unnamed: 0','F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F0_d','F1_d','F2_d','F3_d','F4_d','F5_d','F6_d','F7_d','F8_d','F9_d','F10_d','F11_d','F12_d','F0_dd','F1_dd','F2_dd','F3_dd','F4_dd','F5_dd','F6_dd','F7_dd','F8_dd','F9_dd','F10_dd','F11_dd','F12_dd'])]


validationx_3 = val_set3.loc[:, val_set3.columns.drop(['Name','Gender', 'Unnamed: 0'])]

validationy_3 = val_set3.loc[:, val_set3.columns.drop(['Name','Unnamed: 0','F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F0_d','F1_d','F2_d','F3_d','F4_d','F5_d','F6_d','F7_d','F8_d','F9_d','F10_d','F11_d','F12_d','F0_dd','F1_dd','F2_dd','F3_dd','F4_dd','F5_dd','F6_dd','F7_dd','F8_dd','F9_dd','F10_dd','F11_dd','F12_dd'])]


validationx_4 = val_set4.loc[:, val_set4.columns.drop(['Name','Gender', 'Unnamed: 0'])]

validationy_4 = val_set4.loc[:, val_set4.columns.drop(['Name','Unnamed: 0','F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F0_d','F1_d','F2_d','F3_d','F4_d','F5_d','F6_d','F7_d','F8_d','F9_d','F10_d','F11_d','F12_d','F0_dd','F1_dd','F2_dd','F3_dd','F4_dd','F5_dd','F6_dd','F7_dd','F8_dd','F9_dd','F10_dd','F11_dd','F12_dd'])]


validationx_5 = val_set5.loc[:, val_set5.columns.drop(['Name','Gender', 'Unnamed: 0'])]

validationy_5 = val_set5.loc[:, val_set5.columns.drop(['Name','Unnamed: 0','F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F0_d','F1_d','F2_d','F3_d','F4_d','F5_d','F6_d','F7_d','F8_d','F9_d','F10_d','F11_d','F12_d','F0_dd','F1_dd','F2_dd','F3_dd','F4_dd','F5_dd','F6_dd','F7_dd','F8_dd','F9_dd','F10_dd','F11_dd','F12_dd'])]


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

sety_1_test_1 = np.array(sety_1_test)
#sety_1_train_1 = sety_1_train_1.reshape(sety_1_train_1.shape[0], sety_1_train_1.shape[1],1)

#series_input = (setx_1_train.shape[1],1,)

#################   Model    ###########

model = Sequential()
model.add(Conv1D(filters = 16,kernel_size = 3,strides=1, padding='same', input_shape = (setx_1_train_1.shape[1], 1 ) , activation="relu"))
model.add(MaxPooling1D(4))
#model.add(Dropout(0.2))
model.add(BatchNormalization())
#model.add(SeqSelfAttention(attention_activation='sigmoid'))
#model.add(LSTM(20, return_sequences=True,activation='tanh'))
#model.add(LSTM(100, return_sequences=False,activation='tanh'))
#model.add(Dropout(0.2))
#model.add(Flatten())
#model.add(Dropout(0.1))
#model.add(Dense(128, activation = 'tanh'))
#model.add(Dropout(0.4))
#model.add(Dense(64, activation = 'tanh'))
#model.add(Dropout(0.2))
# #model.add(Conv1D( 16, 3, input_shape = (setx_1_train_1.shape[1], 1 ) ))
# model.add(Dropout(0.2))
# model.add(LSTM(100, return_sequences=True,activation='tanh'))
# model.add(Dropout(0.1))
# model.add(LSTM(100, return_sequences=True,activation='tanh'))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(500, activation = 'relu'))
#model.add(LSTM(100, return_sequences=True,activation='tanh'))
#model.add(Flatten())
#model.add(LSTM(64, return_sequences=False,activation='tanh'))
#model.add(Dense(1000, activation = 'relu'))
#model.add(Flatten())
#model.add(Dropout(0.2))
#model.add(Dense(500, activation = 'relu'))
model.add(Dense(1 ,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005) , metrics=['accuracy'])
model.summary()


################ Fit Fold 1 ##########################

history = model.fit(setx_1_train_1,sety_1_train_1, batch_size=256 , epochs = 20, validation_data=(setx_1_test_1,sety_1_test_1), shuffle = True)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# model_json = model.to_json()
# with open("Model_Dense_wo_Dropout.json" , "w") as json_file:
#     json_file.write(model_json)
# model.save_weights("Model_Dense_wo_Dropout_weights.h5")
# print("Model Saved to disk")

# model.evaluate(setx_1_test_1, sety_1_test_1)

# predictions_fold1 = (model.predict(setx_1_test) > 0.5).astype("int32")





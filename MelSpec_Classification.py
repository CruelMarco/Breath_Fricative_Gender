# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 14:31:53 2022

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
#from emodata1d import load_data
from keras.utils import to_categorical
import keras.backend as K
import argparse
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn import preprocessing
import tensorflow as tf
import random
import librosa.display
from IPython.display import Audio
import pyaudio  
import wave
dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/no_wheeze'

test_dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/Spectrogram_imgs'

os.chdir(dir)

files = os.listdir(dir)

wav_files = [f for f in files if f.endswith(".wav")]

txt_files = [f for f in files if f.endswith(".txt")]

male_count = 0

female_count = 0


############# MFCC Calculation ###############

for j in wav_files :
    
    print(j)
    
    audio_path = os.path.join(dir, j)
    
    audio_file, fs = librosa.load(audio_path, sr = 48000, mono = True)
    
    annot_path = audio_path[0 : -3] + 'txt'
    
    annot_file = pd.read_csv(annot_path , sep = "\t", names = ['start', 'end', 'phon'] , header = None)
    
    gender = j.split("_")[8]
    
    if gender == 'M' :
        
        male_count+=1
    else :
        
        female_count+=1
    
    name = j.split("_")[5]
    
    phon_col = annot_file['phon']
    
    st_col = annot_file['start']
    
    end_col = annot_file['end']
    
    wheeze_idx = [i for i in range(len(phon_col)) if "Wheeze" in phon_col[i]]
    
    wheeze_st_idx = st_col[wheeze_idx]
    
    wheeze_end_idx = end_col[wheeze_idx]
    
    wheeze_st_sam = list(np.ceil(wheeze_st_idx * fs))
    
    wheeze_end_sam = list(np.ceil(wheeze_end_idx*fs))
    
    wheeze_st_sam = [int(i) for i in wheeze_st_sam]
    
    wheeze_chunk = []
    
    wheeze_chunk_mfcc_df = []
    
    sub_mfcc_df = []
    
    complete_wheeze = []
    
    for i in range(len(wheeze_st_sam)) :
        
        wheeze_chunk = audio_file[int(wheeze_st_sam[i]) : int(wheeze_end_sam[i])]
        
        print(len(wheeze_chunk))
        
        S = librosa.feature.melspectrogram(wheeze_chunk, sr=fs, n_mels=128, win_length = 960,

                                    hop_length = 480, fmax=8000)
        S = librosa.power_to_db(S, ref=np.max)
                                 
        fig = plt.figure(figsize=[2.5,2.5])
        
        ax = fig.add_subplot(111)
        
        ax.axes.get_xaxis().set_visible(False)
        
        ax.axes.get_yaxis().set_visible(False)
        
        ax.set_frame_on(False)
        
        img = librosa.display.specshow(S, x_axis='time', y_axis='mel', hop_length=480,sr = fs, fmax = fs/2, ax=ax)
        
        filename  = test_dir + '/' + j[:-4] + '_' + str(i) + '.jpg'
        
        plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
        # plt.close()    
        # fig.clf()
        # plt.close(fig)
        # plt.close('all')               
        
        

        #mel_spect_dB = librosa.power_to_db(mel_spect, ref=np.max)
        
        #librosa.display.waveplot(wheeze_chunk, sr=fs, ax=ax[0])

        #img = librosa.display.specshow(S, x_axis='time',

                                       # y_axis='mel', sr=fs, hop_length= 160,

                                       #   fmax=8000, ax=ax)
        
        # ax.set(title='Linear spectrogram')
        # fig.colorbar(img, ax=ax, format="%+2.f dB")
        # fig.tight_layout()

        # plt.show()


#mel_spect = librosa.feature.melspectrogram(audio_file, sr = 16000, n_mels = 256, 
                                                   #win_length=320, hop_length= 160, fmax = 8000)    
        
# fig, ax = plt.subplots() 

# mel_spect_dB = librosa.power_to_db(mel_spect, ref=np.max)

# img = librosa.display.specshow(mel_spect_dB, x_axis='time',

#                          y_axis='mel', sr=fs,

#                          fmax=8000, ax=ax)
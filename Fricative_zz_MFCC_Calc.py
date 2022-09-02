# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 00:47:33 2022

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
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

dir = "C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/test_recs"

os.chdir(dir)

files = os.listdir(dir)

wav_files = [f for f in files if f.endswith(".wav")]

txt_files = [f for f in files if f.endswith(".txt")]

############# MFCC Calculation ###############

for j in wav_files :
    
    audio_path = os.path.join(dir, j)
    
    audio_file, fs = librosa.load(audio_path, sr = 16000, mono = True)
    
    annot_path = audio_path[0 : -3] + 'txt'
    
    annot_file = pd.read_csv(annot_path , sep = "\t", names = ['start', 'end', 'phon'] , header = None)
    
    gender = j.split("_")[8]
    
    name = j.split("_")[5]
    
    phon_col = annot_file['phon']
    
    st_col = annot_file['start']
    
    end_col = annot_file['end']
    
    #zz_idx = [i for i in range(len(phon_col)) if "Zz" or "zz" in phon_col[i]]
    
    #print(zz_idx)
    
    #zz_st_idx = st_col[zz_idx]
    
    #zz_end_idx = end_col[zz_idx]
    
    #zz_st_sam = list(np.ceil(zz_st_idx * fs))
    
    #zz_end_sam = list(np.ceil(zz_end_idx*fs))
    
    #zz_st_sam = [int(i) for i in zz_st_sam]
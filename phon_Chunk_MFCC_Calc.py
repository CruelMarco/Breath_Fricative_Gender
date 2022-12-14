# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 20:39:18 2022

Data_set = Shivani Hospital Data, Controls only

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
from operator import itemgetter

#os.chdir('C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data')

dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/test_recs'

os.chdir(dir)

mfcc_store_dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/control_mfcc/test'

files = os.listdir(dir)

wav_files = [f for f in files if f.endswith(".wav")]

txt_files = [f for f in files if f.endswith(".txt")]

male_count = 0

female_count = 0

d_columns = ['F0_d' , 'F1_d' , 'F2_d' , 'F3_d' , 'F4_d' , 'F5_d' , 'F6_d' , 'F7_d' , 'F8_d' , 'F9_d' , 
             'F10_d' ,'F11_d' , 'F12_d', 'F13_d', 'F14_d', 'F15_d', 'F16_d', 'F17_d', 'F18_d', 'F19_d', 'F20_d', 
             'F21_d', 'F22_d', 'F23_d', 'F24_d', 'F25_d', 'F26_d', 'F27_d', 'F28_d', 'F29_d', 'F30_d', 
             'F31_d', 'F32_d', 'F33_d', 'F34_d', 'F35_d', 'F36_d', 'F37_d', 'F38_d']
    
dd_columns = [j + 'd' for j in d_columns]

mfcc_columns = [j.split('_')[0] for j in d_columns]



############# MFCC Calculation ###############

for j in tqdm(wav_files) :
    
    audio_path = os.path.join(dir, j)
    
    audio_file, fs = librosa.load(audio_path, sr = 44000, mono = True)
    
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
    

    
    
    

    #print(j)
    
    for i in range(len(wheeze_st_sam)) :
        
        wheeze_chunk = audio_file[int(wheeze_st_sam[i]) : int(wheeze_end_sam[i])]
        
        mfcc = librosa.feature.mfcc(wheeze_chunk , sr = fs , n_mfcc = 39 , n_fft = 320 , win_length = 320 , hop_length = 160)
                
        wheeze_chunk_mfcc_df = np.array(np.transpose(librosa.feature.mfcc(wheeze_chunk , sr = fs , n_mfcc = 39 , n_fft = 320 , win_length = 320 , hop_length = 160)))
        
        delta = np.array(np.transpose(librosa.feature.delta(mfcc)))
        
        delta_chunk_mfcc_df = pd.DataFrame(delta , columns = d_columns)

        delta2 = np.array(np.transpose(librosa.feature.delta(mfcc, order=2)))
        
        delta2_chunk_mfcc_df = pd.DataFrame(delta2 , columns = dd_columns)
        
        wheeze_chunk_mfcc_df = pd.DataFrame(wheeze_chunk_mfcc_df , columns = mfcc_columns)
        
        mfcc_delta_delta2 = pd.concat([wheeze_chunk_mfcc_df, delta_chunk_mfcc_df, delta2_chunk_mfcc_df],axis=1, join='inner')
        
        #print(len(wheeze_chunk_mfcc_df))
        
        sub_mfcc_df.append(mfcc_delta_delta2)
        
    sub_mfcc_df = pd.concat(sub_mfcc_df)
     
    sub_mfcc_df.insert(0,'Name' , name , True)
    
    sub_mfcc_df['Gender'] = gender
    
    mfcc_df_name = j[0 : -4] + '_d_dd_39' + '.csv'
    
    mfcc_df_dir = os.path.join(mfcc_store_dir , mfcc_df_name)
    
    sub_mfcc_df.to_csv(mfcc_df_dir)
    
print("No. of Male controls = " , male_count)

print("No. of Female controls = " , female_count)
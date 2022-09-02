# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 23:21:55 2022

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

dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/test_recs'

mfcc_stat_store_dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/mfcc_stats_csv'

files = os.listdir(dir)

wav_files = [f for f in files if f.endswith(".wav")]

txt_files = [f for f in files if f.endswith(".txt")]

male_count = 0

female_count = 0
    
############# MFCC Calculation ###############

for j in tqdm(wav_files) :
    
    audio_path = os.path.join(dir, j)
    
    audio_file, fs = librosa.load(audio_path, sr = 16000, mono = True)
    
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
        
    wheeze_chunk_mfcc_df_mean = []
    
    wheeze_chunk_mfcc_df_median = []
    
    wheeze_chunk_mfcc_df_mode = []
    
    wheeze_chunk_mfcc_df_std = []
    
    sub_mfcc_df_mean = []
    
    sub_mfcc_df_median = []
    
    sub_mfcc_df_mode = []
    
    sub_mfcc_df_std = []
    
    for i in range(len(wheeze_st_sam)) :
        
        wheeze_chunk = audio_file[int(wheeze_st_sam[i]) : int(wheeze_end_sam[i])]
                
        wheeze_chunk_mfcc_df = np.array(np.transpose(librosa.feature.mfcc(wheeze_chunk , sr = fs , n_mfcc = 13 , n_fft = 320 , win_length = 320 , hop_length = 160)))
        
        ##########Complete MFCCs############
        
        wheeze_chunk_mfcc_df = pd.DataFrame(wheeze_chunk_mfcc_df , columns = ['F0' , 'F1' , 'F2' , 'F3' , 'F4' , 'F5' , 'F6' , 'F7' , 'F8' , 'F9' , 'F10' , 'F11' , 'F12'])
        
        sub_mfcc_df.append(wheeze_chunk_mfcc_df)
        
        ##########MFCCs Mean##############
        
        wheeze_chunk_mfcc_df_mean = np.array(wheeze_chunk_mfcc_df[['F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12']].mean())
        
        wheeze_chunk_mfcc_df_mean = pd.DataFrame(wheeze_chunk_mfcc_df_mean)
        
        sub_mfcc_df_mean.append(wheeze_chunk_mfcc_df_mean)
        
        ##########MFCCs Median############
        
        wheeze_chunk_mfcc_df_median = np.array(wheeze_chunk_mfcc_df[['F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12']].median())
        
        wheeze_chunk_mfcc_df_median = pd.DataFrame(wheeze_chunk_mfcc_df_median)
        
        sub_mfcc_df_median.append(wheeze_chunk_mfcc_df_median)
        
        ############MFCC Mode##############
        
        wheeze_chunk_mfcc_df_floor = np.array(wheeze_chunk_mfcc_df[['F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12']].apply(np.floor))
        
        wheeze_chunk_mfcc_df_mode = stats.mode(wheeze_chunk_mfcc_df_floor)[0].T
        
        wheeze_chunk_mfcc_df_mode = pd.DataFrame(wheeze_chunk_mfcc_df_mode)
        
        sub_mfcc_df_mode.append(wheeze_chunk_mfcc_df_mode)
        
        ############MFCC SD################
        
        wheeze_chunk_mfcc_df_std = np.array(wheeze_chunk_mfcc_df[['F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12']].std())
        
        wheeze_chunk_mfcc_df_std = pd.DataFrame(wheeze_chunk_mfcc_df_std)
        
        sub_mfcc_df_std.append(wheeze_chunk_mfcc_df_std)
        
    df_idx = [str(x) for x in range(1,len(wheeze_idx)+1)]    
    
    sub_mfcc_df = pd.concat(sub_mfcc_df)
    
    #################MFCC Mean DataFrame##################
    
    sub_mfcc_df_mean = pd.concat(sub_mfcc_df_mean, axis=1).T
    
    sub_mfcc_df_mean.columns = ['F0_mean' , 'F1_mean' , 'F2_mean' , 'F3_mean' , 'F4_mean' , 'F5_mean' , 'F6_mean' , 'F7_mean' , 'F8_mean' , 'F9_mean' , 'F10_mean' , 'F11_mean' , 'F12_mean']
    
    sub_mfcc_df_mean.index = df_idx
    
    ##################MFCC Median DataFrame#################
    
    sub_mfcc_df_median = pd.concat(sub_mfcc_df_median,axis=1).T
    
    sub_mfcc_df_median.columns = ['F0_median' , 'F1_median' , 'F2_median' , 'F3_median' , 'F4_median' , 'F5_median' , 'F6_median' , 'F7_median' , 'F8_median' , 'F9_median' , 'F10_median' , 'F11_median' , 'F12_median']
    
    sub_mfcc_df_median.index = df_idx
    
    ####################MFCC Mode DataFrame#################
    
    sub_mfcc_df_mode = pd.concat(sub_mfcc_df_mode,axis=1).T
    
    sub_mfcc_df_mode.columns = ['F0_mode' , 'F1_mode' , 'F2_mode' , 'F3_mode' , 'F4_mode' , 'F5_mode' , 'F6_mode' , 'F7_mode' , 'F8_mode' , 'F9_mode' , 'F10_mode' , 'F11_mode' , 'F12_mode']
    
    sub_mfcc_df_mode.index = df_idx
    
    ####################MFCC Standard Deviation DataFrame##################
    
    sub_mfcc_df_std = pd.concat(sub_mfcc_df_std,axis=1).T
    
    sub_mfcc_df_std.columns = ['F0_std' , 'F1_std' , 'F2_std' , 'F3_std' , 'F4_std' , 'F5_std' , 'F6_std' , 'F7_std' , 'F8_std' , 'F9_std' , 'F10_std' , 'F11_std' , 'F12_std']
    
    sub_mfcc_df_std.index = df_idx
    
    #####################MFCC Statistics DataFrame########################
    
    mfcc_stat_df = pd.concat([sub_mfcc_df_mean, sub_mfcc_df_median, sub_mfcc_df_mode, sub_mfcc_df_std ], axis = 1, join = 'inner')
    
    #####################Exporting MFCC Statistics to CSV################
    
    mfcc_stat_df.insert(0,'Name' , name , True)
    
    mfcc_stat_df['Gender'] = gender
    
    # mfcc_stat_df_name = j[0 : -4] + '_stats' + '.csv'
    
    # mfcc_df_dir = os.path.join(mfcc_stat_store_dir , mfcc_stat_df_name)
    
    # mfcc_stat_df.to_csv(mfcc_df_dir)
    
print("No. of Male controls = " , male_count)

print("No. of Female controls = " , female_count)




    
    
    

# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 19:32:51 2022

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
from operator import itemgetter

#os.chdir('C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data')

dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/control_phonation_recording'

os.chdir(dir)

mfcc_store_dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/control_mfcc/mfcc_d_dd'

files = os.listdir(dir)

wav_files = [f for f in files if f.endswith(".wav")]

txt_files = [f for f in files if f.endswith(".txt")]

male_count = 0

female_count = 0
    
############# MFCC Calculation ###############

for j in wav_files :
    
    audio_path = os.path.join(dir, j)
    
    #audio_file, fs = librosa.load(audio_path, sr = 16000, mono = True)
    
    annot_path = audio_path[0 : -3] + 'txt'
    
    annot_file = pd.read_csv(annot_path , sep = "\t", names = ['start', 'end', 'phon'] , header = None)
    
    gender = j.split("_")[8]
    
    # if gender == 'M' :
        
    #     male_count+=1
    # else :
        
    #     female_count+=1
    
    name = j.split("_")[5]
    
    phon_col = annot_file['phon']
    
    st_col = annot_file['start']
    
    end_col = annot_file['end']
    
    #wheeze_idx = [i for i in range(len(phon_col)) if "Wheeze" in phon_col[i]]
    
    if np.any(["ww" in x for x in phon_col]):
        
        print(j)
        
    else:
        
        print("Nothing")
        
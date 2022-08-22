# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 19:49:13 2022

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

mfcc_stat_store_dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/mfcc_stats_csv'

male_file_dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/control males'

os.chdir(mfcc_stat_store_dir)

files = os.listdir(mfcc_stat_store_dir)

male_count = 0

female_count = 0

male_file_count = 0

for i in files:
    
    gender = i.split("_")[8]
    
    if gender == 'M' :
        
        male_count+=1
        
        male_file = i
        
        male_file_src_path = os.path.join(mfcc_stat_store_dir,male_file)
        
        male_file_dest_path = os.path.join(male_file_dir,male_file)
        
        shutil.move(male_file_src_path, male_file_dest_path)
        

    else :
        
        female_count+=1
        
    name = i.split("_")[5]
    
print("Male count is ", male_count)

print("Feale count is ", female_count)

print(male_file_count)


    
    
    

#


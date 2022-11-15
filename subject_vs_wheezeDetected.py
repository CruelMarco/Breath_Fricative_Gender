# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 13:40:47 2022

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

dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/test_recs'

txt_files = [i for i in os.listdir(dir) if i.endswith(".txt") ]

cols =  ["name" , "wheeze_count"]

lst = []

for i in txt_files :
    
    annot_path = os.path.join(dir, i)
    
    print(annot_path)
    
    annot_file = pd.read_csv(annot_path , sep = "\t", names = ['start', 'end', 'phon'] , header = None)
    
    name = i.split("_")[5]
    
    phon_col = annot_file["phon"]
    
    wheeze_idx = [j for j in range(len(phon_col)) if "Wheeze" in phon_col[j]]
    
    wheeze_count = len(wheeze_idx)
    
    lst.append([name , wheeze_count])
    
    wheeze_count_df = pd.DataFrame(lst, columns = cols )
    
print(wheeze_count_df)
    
    


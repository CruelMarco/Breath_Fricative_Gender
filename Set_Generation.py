# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 20:31:05 2022

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

dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/control_mfcc/mfcc_d_ddC'

os.chdir(dir)

male_dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/control_mfcc/control_males'

female_dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/control_mfcc/control_females'

male_files = [os.path.join(male_dir,f) for f in os.listdir(male_dir)]

female_files = [os.path.join(female_dir,f) for f in os.listdir(female_dir)]

def chunkIt(seq, num):
    
    avg = len(seq) / float(num)
    
    out = []
    
    last = 0.0

    while last < len(seq):
        
        out.append(seq[int(last):int(last + avg)])
        
        last += avg

    return out

male_sets = chunkIt(male_files, 5)

female_sets = chunkIt(female_files,5)


    
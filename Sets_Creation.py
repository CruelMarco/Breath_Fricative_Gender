# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 20:45:47 2022

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

dir = "C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/control_mfcc/mfcc_sets"

os.chdir(dir)

sets = os.listdir(dir)

dict = {"Set_1" : [] , "Set_2" : [] , "Set_3" : [] , "Set_4" : [] , "Set_5" : []}

for i in range (len(sets)):
    
    path = os.path.join(dir,sets[i])
    
    
    


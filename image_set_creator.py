# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 02:07:13 2022

@author: Shaique
"""
import os
import pandas as pd
from tqdm import tqdm
from pandas import DataFrame as df
import glob
import shutil
import json
import sklearn
import shutil
import math




img_dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/spectrogram/Spectrogram_imgs'

os.chdir('C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/spectrogram')

imgs = os.listdir(img_dir)

set_dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/control_mfcc/mfcc_stats_sets/Set5'

set_dir_spec = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/spectrogram/set5'

fold_files = os.listdir(set_dir)

file_names = [i.split("_")[5] for i in fold_files]

out = [x for x in imgs if any([ni in x for ni in file_names])]

set_img_dir = [os.path.join(img_dir,i) for i in out]

for i in set_img_dir :
    
    print(i)

    shutil.copy(i , set_dir_spec)








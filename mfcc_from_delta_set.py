# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 02:41:07 2022

@author: Spirelab
"""

import os
import pandas as pd

dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/control_mfcc/mfcc_d_dd'

dest_dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/control_mfcc/mfcc_only'

os.chdir(dir)

mfcc_files = os.listdir(dir)

mfcc_file_path = [dir + '/' + j for j in mfcc_files]

for i in mfcc_file_path:
    
    mfcc_df = pd.read_csv(i)
    
    mfcc_df1 = mfcc_df[["Name" ,"F0","F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12" ]]
    
    mfcc_csv_name = i.split("/")[8]
    
    mfcc_csv_dir = os.path.join(dest_dir,mfcc_csv_name)
    
    mfcc_df1.to_csv(mfcc_csv_dir)
    
    
    
    
    
    

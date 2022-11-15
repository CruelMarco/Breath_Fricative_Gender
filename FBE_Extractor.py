# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:27:20 2022

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
from keras.models import Model,Sequential
from keras import optimizers
from keras.layers import Input,TimeDistributed,Conv1D,BatchNormalization, Reshape,Dropout,MaxPooling1D,MaxPooling2D,LSTM,Dense,Activation,Layer, Flatten
from keras.utils import to_categorical
import keras.backend as K
import argparse
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn import preprocessing
import tensorflow as tf
import random
import librosa.display
from IPython.display import Audio
import pyaudio  
import wave

dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/test_recs'

os.chdir(dir)

files = os.listdir(dir)

wav_files = [i for i in files if i.endswith(dir)]

txt_files = [i for i in files if i.endswith(dir)]
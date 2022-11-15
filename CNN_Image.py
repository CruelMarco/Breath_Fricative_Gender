# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 14:24:58 2022

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
from sklearn.metrics import confusion_matrix

np.random.seed(12)

main_dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/spectrogram/folds/fold1/train_set'

test_dir = 'C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data/spectrogram/folds/fold1/test_set'

os.chdir(main_dir)

classes_train = os.listdir(main_dir)

classes_test = os.listdir(test_dir)

female_imgs = os.listdir(classes_train[0])

male_imgs = os.listdir(classes_train[1])

test_female_imgs = os.listdir(classes_test[0])

test_male_imgs = os.listdir(classes_test[1])

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
BATCH_SIZE = 32
N_CHANNELS = 3
N_CLASSES = 2

train_ds = tf.keras.utils.image_dataset_from_directory(
    
    main_dir, labels = "inferred", label_mode="int", 
    
    class_names= None, color_mode = "rgb", batch_size=32, 
    
    image_size=(128,128), shuffle= True, seed= 20, subset= "training", validation_split=0.1,
    
    crop_to_aspect_ratio=False)

val_ds = tf.keras.utils.image_dataset_from_directory(
    
    main_dir, labels = "inferred", label_mode="int", 
    
    class_names= None, color_mode = "rgb", batch_size=32, 
    
    image_size=(128,128), shuffle= True, seed= 20, subset= "validation", validation_split=0.1,
    
    crop_to_aspect_ratio=False)


test_ds = tf.keras.utils.image_dataset_from_directory(
    
    test_dir, labels = "inferred", label_mode="int", 
    
    class_names= None, color_mode = "rgb", batch_size=32, 
    
    image_size=(128,128), shuffle= False, seed= 20, validation_split=0,
    
    crop_to_aspect_ratio=False)


plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
        
# Function to prepare our datasets for modelling
def prepare(ds, augment=False):
    # Define our one transformation
    rescale = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Rescaling(1./255)])
    flip_and_rotate = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
    ])
    
    # Apply rescale to both datasets and augmentation only to training
    ds = ds.map(lambda x, y: (rescale(x, training=True), y))
    if augment: ds = ds.map(lambda x, y: (flip_and_rotate(x, training=True), y))
    return ds

train_dataset = prepare(train_ds, augment=False)
valid_dataset = prepare(val_ds, augment=False)
test_dataset = prepare(test_ds, augment=False)



# Create CNN model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS)))
model.add(tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()


model.compile(
    loss='binary_crossentropy',
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005),
    metrics=['accuracy'],
)

es = EarlyStopping(monitor = "val_loss", patience = 40, restore_best_weights = True)

#model.summary()

history = model.fit(train_dataset, epochs=100, validation_data=valid_dataset, callbacks=[es])



# Plot the loss curves for training and validation.
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values)+1)

plt.figure(figsize=(8,6))
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss of fold1')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



# Plot the accuracy curves for training and validation.
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(acc_values)+1)

plt.figure(figsize=(8,6))
plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



# Compute the final loss and accuracy
final_loss, final_acc = model.evaluate(test_dataset, verbose=0)
print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_acc))

va = (model.predict(test_dataset) > 0.5).astype("int32")

#y_pred2= (model.predict(test_set_x_2_1) > 0.5).astype("int32")
iterator = iter(test_ds)
true_label = []
try:
    for i in range(10):
        true_label.append(next(iterator)[-1])
except:
    print("done")
 
print(true_label)

va_1 = list(np.concatenate(va))

tl_1 = list(np.concatenate(true_label))

cf2=confusion_matrix(tl_1, va_1)





#conf = confusion_matrix(test_ds,va)
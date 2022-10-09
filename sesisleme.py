# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 21:45:22 2022

@author: lenovo
"""
import os
import datetime as dt

import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython
from IPython.display import Audio
from tqdm.notebook import tqdm_notebook
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn import metrics
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
tqdm_notebook.pandas()

for index, filename in enumerate(os.listdir('hayvan')):
    os.rename(f'hayvan/{filename}', f'hayvan/{str(index)}.wav')
    
for index,filename in enumerate(os.listdir('insan')):
    os.rename(f'insan/{filename}', f'insan/{str(index)}.wav')
IPython.display.display(IPython.display.Audio('hayvan/0.wav'))
IPython.display.display(IPython.display.Audio('hayvan/1.wav'))

IPython.display.display(IPython.display.Audio('insan/0.wav'))
IPython.display.display(IPython.display.Audio('insan/1.wav'))

audio, sr = librosa.load('hayvan/1.wav')
plt.figure(figsize=(12,5))
librosa.display.waveplot(audio, sr)

audio, sr = librosa.load('insan/1.wav')
plt.figure(figsize=(12,5))
librosa.display.waveplot(audio, sr)

hayvan_audio_list = []
for i in range(len(os.listdir('hayvan'))):
    hayvan_audio_list.append(f'hayvan/{i}.wav')
    
insan_audio_list = []
for i in range(len(os.listdir('insan'))):
   insan_audio_list.append(f'insan/{i}.wav')
   
df = pd.DataFrame()

hayvan_audio_data = []
for filename in hayvan_audio_list:
    hayvan_audio_data.append({
        'filename': filename,
        'label': 'hayvan'
    })
df = df.append(pd.DataFrame(hayvan_audio_data), ignore_index=True)
    
insan_audio_data = []
for filename in insan_audio_list:
    insan_audio_data.append({
        'filename': filename,
        'label': 'insan'
    })
df = df.append(pd.DataFrame(insan_audio_data),  ignore_index=True)

#df

df['is_insan'] = pd.get_dummies(df['label'], drop_first=True)
df['audio'] = df['filename'].progress_apply(lambda x: librosa.load(x)[0])
df['sr'] = 22050
df['mfcc'] = df['audio'].progress_apply(lambda x: librosa.feature.mfcc(x, n_mfcc=12))
df['mfcc_mean'] = df['mfcc'].progress_apply(lambda x: np.mean(x.T, axis=0))
df.to_pickle('data.pkl')

df = pd.read_pickle('data.pkl')
X = np.vstack(df['mfcc_mean'].to_numpy())
y = df['is_insan'].to_numpy().reshape(-1,1).astype('float32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14)
print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

model = Sequential(
    [
        Input(shape=(12,)),
        Dense(100, activation='relu', name='layer_1'),
        Dropout(0.5),
        Dense(200, activation='relu', name='layer_2'),
        Dropout(0.5),
        Dense(200, activation='relu', name='layer_3'),
        Dropout(0.5),
        Dense(100, activation='relu', name='layer_4'),
        Dropout(0.5),
        Dense(1, activation='sigmoid', name='output_layer'),
    ]
)

model.summary()

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

log_dir = "logs/fit/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

%%time
num_epochs = 30
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='models/best_4.hdf5', 
                               verbose=1, save_best_only=True)
model.fit(X_train, 
          y_train, 
          batch_size=num_batch_size, 
          epochs=num_epochs, 
          validation_data=(X_test, y_test), 
          callbacks=[checkpointer, tensorboard_callback], 
          verbose=1)

%load_ext tensorboard
%tensorboard --logdir logs

model = tf.keras.models.load_model('models/best_4.hdf5')

def extract_mean_mfcc(filename):
    audio, _ = librosa.load(filename)
    mfcc = librosa.feature.mfcc(audio, n_mfcc=12)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

def predict_class(model, filename):
    test_input = extract_mean_mfcc(filename).reshape(1,-1)
    proba = model.predict(test_input)
#     return proba
    if proba <= 0.5: 
        return 'hayvan'
    else: 
        return 'insan'
    
y_pred = (model.predict(X_train) > 0.5).astype("int32")
print(model.evaluate(X_train, y_train))
confusion_matrix(y_train, y_pred)


y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(model.evaluate(X_test, y_test))
confusion_matrix(y_test, y_pred)

predict_class(model, 'hayvan_test.wav')

predict_class(model, 'insan_test.wav')

model.evaluate(y)








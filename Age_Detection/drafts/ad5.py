# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
import kagglehub
yuulind_imdb_clean_path = kagglehub.dataset_download('yuulind/imdb-clean')

print('Data source import complete.')

!pip install -q tensorflow opencv-python-headless kaggle

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('/kaggle/input/imdb-clean/imdb_train_new_1024.csv')
valid_df = pd.read_csv('/kaggle/input/imdb-clean/imdb_valid_new_1024.csv')

# Filter out invalid ages
train_df = train_df[train_df['age'].between(1, 100)]
valid_df = valid_df[valid_df['age'].between(1, 100)]

image_root = '/kaggle/input/imdb-clean/imdb-clean-1024/imdb-clean-1024'
train_df['filename'] = train_df['filename'].apply(lambda x: os.path.join(image_root, x))
valid_df['filename'] = valid_df['filename'].apply(lambda x: os.path.join(image_root, x))

class AgeDataGenerator(Sequence):
    def __init__(self, df, batch_size=32, img_size=(224, 224), shuffle=True):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_df = self.df.iloc[index * self.batch_size : (index + 1) * self.batch_size]
        X, y = [], []
        for _, row in batch_df.iterrows():
            img = cv2.imread(row['filename'])
            if img is not None:
                img = cv2.resize(img, self.img_size)
                img = img.astype(np.float32) / 255.0
                X.append(img)
                y.append(row['age'])
        return np.array(X), np.array(y)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

def build_model():
    base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(1)(x)

    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer=Adam(1e-4), loss='mse', metrics=['mae'])
    return model

model = build_model()
model.summary()


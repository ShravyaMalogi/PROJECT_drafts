import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split

# Set image size according to your model (EfficientNetB4: 380x380, EfficientNetB3: 300x300)
IMG_SIZE = 300  
BATCH_SIZE = 32

# 1. Load CSV
def load_metadata(csv_path):
    df = pd.read_csv(csv_path)
    df['filepath'] = df['filename'].apply(lambda x: os.path.join('unzipped_folder', x))  # Update with your folder name
    return df

# 2. Crop face using OpenCV and bounding box
def crop_and_resize(image_path, x_min, y_min, x_max, y_max, img_size=IMG_SIZE):
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((img_size, img_size, 3))  # handle missing images
    
    h, w, _ = img.shape
    x_min = max(0, int(x_min))
    y_min = max(0, int(y_min))
    x_max = min(w, int(x_max))
    y_max = min(h, int(y_max))

    face = img[y_min:y_max, x_min:x_max]
    if face.size == 0:
        face = img  # fallback: use entire image

    face = cv2.resize(face, (img_size, img_size))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face / 255.0  # normalize
    return face

# 3. Custom data generator (for large datasets)
class AgeDataGenerator(Sequence):
    def __init__(self, dataframe, batch_size=BATCH_SIZE, img_size=IMG_SIZE, shuffle=True):
        self.df = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_images = []
        batch_labels = []
        for _, row in batch_df.iterrows():
            image = crop_and_resize(row['filepath'], row['x_min'], row['y_min'], row['x_max'], row['y_max'], self.img_size)
            batch_images.append(image)
            batch_labels.append(row['age'])  # raw age, no normalization

        return np.array(batch_images), np.array(batch_labels)

# 4. Load datasets
train_df = load_metadata('/kaggle/input/your-dataset-folder/train.csv')
valid_df = load_metadata('/kaggle/input/your-dataset-folder/valid.csv')
test_df  = load_metadata('/kaggle/input/your-dataset-folder/test.csv')

# 5. Create generators
train_gen = AgeDataGenerator(train_df)
valid_gen = AgeDataGenerator(valid_df, shuffle=False)
test_gen  = AgeDataGenerator(test_df, shuffle=False)

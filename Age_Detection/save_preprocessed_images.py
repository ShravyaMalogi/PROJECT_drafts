import os
import cv2
import pandas as pd
from tqdm.notebook import tqdm

IMG_SIZE = 300  # match with your model's input size
SAVE_DIR = "/kaggle/working/cropped_faces"

def crop_and_save_faces(df, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        filepath = os.path.join('unzipped_folder', row['filename'])  # Update with your folder name
        image = cv2.imread(filepath)

        if image is None:
            saved_paths.append(None)
            continue

        h, w, _ = image.shape
        x_min = max(0, int(row['x_min']))
        y_min = max(0, int(row['y_min']))
        x_max = min(w, int(row['x_max']))
        y_max = min(h, int(row['y_max']))
        face = image[y_min:y_max, x_min:x_max]

        if face.size == 0:
            face = image  # fallback to full image

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Save image
        save_path = os.path.join(save_dir, f"{i}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        saved_paths.append(save_path)

    df = df.copy()
    df['cropped_path'] = saved_paths
    return df

class CroppedFaceGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, batch_size=BATCH_SIZE, img_size=IMG_SIZE, shuffle=True):
        self.df = dataframe.dropna(subset=["cropped_path"]).reset_index(drop=True)
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
            image = cv2.imread(row['cropped_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255.0  # normalize
            batch_images.append(image)
            batch_labels.append(row['age'])

        return np.array(batch_images), np.array(batch_labels)

train_df = load_metadata('/kaggle/input/your-dataset/train.csv')
train_df = crop_and_save_faces(train_df)

valid_df = load_metadata('/kaggle/input/your-dataset/valid.csv')
valid_df = crop_and_save_faces(valid_df)

test_df = load_metadata('/kaggle/input/your-dataset/test.csv')
test_df = crop_and_save_faces(test_df)

train_gen = CroppedFaceGenerator(train_df)
valid_gen = CroppedFaceGenerator(valid_df, shuffle=False)
test_gen  = CroppedFaceGenerator(test_df, shuffle=False)

train_df.to_csv("/kaggle/working/train_cropped.csv", index=False)
valid_df.to_csv("/kaggle/working/valid_cropped.csv", index=False)
test_df.to_csv("/kaggle/working/test_cropped.csv", index=False)

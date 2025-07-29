import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('best_age_model.keras')

# Define image size
IMG_SIZE = (224, 224)  # Use your model's input size

# Preprocess image
def preprocess_image(img):
    img = img.convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = img_array / 255.0  # normalize if you trained like this
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit UI
st.title("🧓 Age Detection App")
st.write("Upload a face image to predict age using your trained model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting..."):
        preprocessed = preprocess_image(image)
        predicted_age = model.predict(preprocessed)[0][0]  # [0][0] for scalar regression output

    st.success(f"Estimated Age: **{predicted_age:.2f}** years")

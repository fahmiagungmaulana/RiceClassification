import os
import requests
import gdown
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobile

# Function to download model from Google Drive
def download_model_from_drive(file_id, output_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output=output_path, quiet=False)

# Download the CNN model from Google Drive if it's not already downloaded
cnn_model_path = 'rice_model.keras'
if not os.path.exists(cnn_model_path):
    st.write("Downloading CNN model...")
    download_model_from_drive('1wgi3tYyg5oX6zbft2aE9GOHaul-oKE3j', cnn_model_path)
    st.write("CNN model downloaded successfully!")

# Load models
mobilenet_model = tf.keras.models.load_model("mobilenet.h5")
cnn_model = tf.keras.models.load_model(cnn_model_path)

# Function to predict using MobileNetV2
def predict_mobilenet(uploaded_file):
    map_dict = {0: 'Arborio', 1: 'Basmati', 2: 'Ipsala', 3: 'Jasmine', 4: 'Karacadag'}

    img = load_img(uploaded_file)
    input_arr = img_to_array(img)
    input_arr = np.expand_dims(input_arr, axis=0)
    input_arr = mobile(input_arr)
    prediction = mobilenet_model.predict(input_arr)
    result = map_dict[np.argmax(prediction)]
    return result

# Function to predict using CNN model
def predict_cnn(uploaded_file):
    map_dict = {0: 'Arborio', 1: 'Basmati', 2: 'Ipsala', 3: 'Jasmine', 4: 'Karacadag'}

    img = load_img(uploaded_file)
    input_arr = img_to_array(img)
    input_arr = np.expand_dims(input_arr, axis=0)
    input_arr = mobile(input_arr)
    prediction = cnn_model.predict(input_arr)
    result = map_dict[np.argmax(prediction)]
    return result

# Title and description
st.title("Rice Classification")
st.write(
    "This classifier categorizes images of rice into five different types: Arborio, Basmati, Ipsala, Jasmine, and Karacadag."
)

# Sidebar navigation
page = st.sidebar.selectbox("Choose a model", ("MobileNetV2", "CNN"))

uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])
generate_pred = False

if uploaded_file is not None:
    generate_pred = st.button("Generate Prediction")

if generate_pred:
    if page == "MobileNetV2" and uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

        st.image(opencv_image, channels="RGB")
        result = predict_mobilenet(uploaded_file)
        st.title("The rice is {}".format(result))

    elif page == "CNN" and uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

        st.image(opencv_image, channels="RGB")
        result = predict_cnn(uploaded_file)
        st.title("The rice is {}".format(result))

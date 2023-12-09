import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobile

model = tf.keras.models.load_model("mobilenet.keras")

uploaded_file = st.file_uploader("Choose a image file", type = ['png', 'jpg', 'jpeg'])

map_dict = {
    0: 'Arborio',
    1: 'Basmati',
    2: 'Ipsala',
    3: 'Jasmine',
    4: 'Karacadag'
}

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    st.image(opencv_image, channels="RGB")

    Genrate_pred = st.button("Generate Prediction")
    if Genrate_pred:
        img = load_img(uploaded_file)
        input_arr = img_to_array(img)
        prediction = model(np.array([input_arr]))
        st.title("Beras Tersebut adalah beras jenis {}".format(map_dict[np.argmax(prediction)]))

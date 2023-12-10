import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobile

st.markdown(
    """
    <style>
    .title-container {
        display: flex;
        align-items: center;
    }
    .title-text {
        margin-top: 20px;
        margin-left: 10px;
        text-align: left;
        
    }
    .title-img {
        width: 700px;
        height: auto;
    }
    .left-text {
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 4])

with col1:
    st.image('COVER.png', width=700)

with col2:
    st.markdown("<h1 class='title-text'>Rice Classification</h1>", unsafe_allow_html=True)

st.write("This classifier categorizes images of rice into five different types: Arborio, Basmati, Ipsala, Jasmine, and Karacadag.")

model = tf.keras.models.load_model('mobilenet.h5')

uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

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

    Generate_pred = st.button("Generate Prediction")
    if Generate_pred:
        img = load_img(uploaded_file)
        input_arr = img_to_array(img)
        prediction = model(np.array([input_arr]))
        st.write("Beras Tersebut adalah beras jenis:")
        st.write(map_dict[np.argmax(prediction)])


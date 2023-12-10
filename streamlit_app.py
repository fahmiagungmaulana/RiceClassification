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
        align-items: left;
    }
    .title-text {
        margin-top: 20px;
        margin-left: -120px;
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

model_selection = st.radio("Select Model", ('CNN', 'MobileNet'))

if model_selection == 'CNN':
    model_path = 'rice_model.keras'
elif model_selection == 'MobileNet':
    model_path = 'mobilenet.h5'

model = tf.keras.models.load_model(model_path)
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
        st.write("This type of rice is:")
        st.write(map_dict[np.argmax(prediction)])


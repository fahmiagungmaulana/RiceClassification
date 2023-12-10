import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobile

st.title("Rice Classification")
st.write("This classifier categorizes images of rice into five different types: Arborio, Basmati, Ipsala, Jasmine, and Karacadag.")

mobilenet_model = tf.keras.models.load_model('mobilenet.h5')

def predict_mobilenet(uploaded_file):
    map_dict = {0: 'Arborio', 1: 'Basmati', 2: 'Ipsala', 3: 'Jasmine', 4: 'Karacadag'}

    img = load_img(uploaded_file)
    input_arr = img_to_array(img)
    input_arr = np.expand_dims(input_arr, axis=0)
    input_arr = mobile(input_arr)
    prediction = mobilenet_model.predict(input_arr)
    result = map_dict[np.argmax(prediction)]
    return result

uploaded_file = st.file_uploader("Choose a image file", type = ['png', 'jpg', 'jpeg'])
generate_pred = False

if uploaded_file is not None:
    generate_pred = st.button("Generate Prediction")

if generate_pred:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    
    st.image(opencv_image, channels="RGB")
    result = predict_mobilenet(uploaded_file)
    st.title("The rice is {}".format(result))

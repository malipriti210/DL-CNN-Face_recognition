import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

if "open_camera" not in st.session_state:
    st.session_state.open_camera = False
st.title("Face Mask Detection")

import os
import requests

MODEL_URL = "https://github.com/MadhuraKoshti/Face_Mask_Recognition/releases/download/v1.0/mask_final.keras"

if not os.path.exists("mask_final.keras"):
    response = requests.get(MODEL_URL)
    with open("mask_final.keras", "wb") as f:
        f.write(response.content)

model = load_model("mask_final.keras")
uploaded_file = st.file_uploader("Upload an Image ",type = ["jpg","jpeg","png"])

if uploaded_file is not None:
    st.image(uploaded_file)

    img = Image.open(uploaded_file)

    img = img.resize((128,128))  # resize image to 128 , 128 as this size we ussed while training

    img_array = image.img_to_array(img) / 255.0 #create array of uploaded image and normalise it
   
    img_array = np.expand_dims(img_array,axis  = 0)  # converts to expected dimension i.e 1,128,128

    prediction = model.predict(img_array)

    prob = prediction[0][0]

    if prob > 0.5 :
        st.success("Prediction: WITHOUT Mask ❌😷")
    else :
        st.success("Prediction: WITH Mask ✅😷")

col1, col2 = st.columns(2)

with col1:
    if st.button("📸 Open Camera"):
        st.session_state.open_camera = True

with col2:
    if st.button("❌ Close Camera"):
        st.session_state.open_camera = False


if st.session_state.open_camera:

    camera_image = st.camera_input("Click Photo")

    if camera_image is not None:

        img = Image.open(camera_image)
        img = img.resize((128,128))

        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        confidence = prediction[0][0]

        if confidence > 0.5:
            st.success(f"Without Mask 😷 ({confidence:.2%})")
        else:
            st.success(f"With Mask ✅ ({(1-confidence):.2%})")

        st.session_state.open_camera = False

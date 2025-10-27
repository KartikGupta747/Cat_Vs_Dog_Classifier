
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
# import subprocess # Server chalu karne ke liye
# import time # Thoda rukne ke liye


st.title("🐱🐶 Cat vs. Dog Classifier")
st.header('CNN based Model')
st.write('Upload a photo of dog or cat, this model will tell whether it is a dog or a cat')
# @st.cache_resource
def load_model():
    model = tf.keras.models.load_model('cats_vs_dogs_v1.keras') 
    transfer_model=tf.keras.models.load_model('transfer_learning_model (2).keras')
    return model,transfer_model
model,transfer_model=load_model()
uploaded_file=st.file_uploader('Select image type',type=['JPG','PNG','JPEG'])

#img preprocessing
def preprocess(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    size=(150,150)
    image=ImageOps.fit(image,size,Image.Resampling.LANCZOS)
    img_array=np.asarray(image)
    img_array=img_array/255.0
    img_array=np.expand_dims(img_array,axis=0)

    return img_array

def preprocess_trf(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    size = (150,150)
    image=ImageOps.fit(image,size,Image.Resampling.LANCZOS)
    img_array=np.asarray(image)
    img_array=np.expand_dims(img_array,axis=0)
    img_array=preprocess_input(img_array)


    return img_array

#Prediction Logic
if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image,caption='Uploaded Image',use_column_width=True)
    preprocessed_img=preprocess(image)
    preprocessed_img_trf=preprocess_trf(image)
    prediction = model.predict(preprocessed_img)
    prediction_trf = transfer_model.predict(preprocessed_img_trf)
    st.write('OUTPUT OF MODEL 1 (WITHOUT TRANSFER_LEARNING)')
    if prediction[0][0] > 0.5:
        st.success(f"**Faisla: Yeh ek KUTTA (DOG) hai!** 🐶 (Confidence: {prediction[0][0]*100:.2f}%)")
    else:
        st.success(f"**Faisla: Yeh ek BILLI (CAT) hai!** 🐱 (Confidence: {(1-prediction[0][0])*100:.2f}%)")

    st.write('OUTPUT OF MODEL 2 (WITH TRANSFER_LEARNING)')
    if prediction_trf[0][0] > 0.5:
        st.success(f"**Faisla: Yeh ek KUTTA (DOG) hai!** 🐶 (Confidence: {prediction_trf[0][0]*100:.2f}%)")
    else:
        st.success(f"**Faisla: Yeh ek BILLI (CAT) hai!** 🐱 (Confidence: {(1-prediction_trf[0][0])*100:.2f}%)")




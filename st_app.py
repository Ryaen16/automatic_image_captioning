# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 16:57:18 2024

@author: saumya
"""

import streamlit as st 
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences



model = tf.keras.models.load_model("model_49.h5")

model_temp = tf.keras.applications.ResNet50(weights="imagenet", input_shape=(224,224,3))

# Create a new model, by removing the last layer (output layer of 1000 classes) from the resnet50
model_resnet = Model(model_temp.input, model_temp.layers[-2].output)

def preprocess_image(img):
    img = image.load_img(img, target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def encode_image(img):
    img = preprocess_image(img)
    feature_vector = model_resnet.predict(img)
    feature_vector = feature_vector.reshape(1, feature_vector.shape[1])
    return feature_vector


def preprocess_img_arr(img) :
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    feature_vector = model_resnet.predict(img)
    feature_vector = feature_vector.reshape(1, feature_vector.shape[1])
    return feature_vector

with open("word_to_idx.pkl", 'rb') as w2i:
    word_to_idx = pickle.load(w2i)
    
with open("idx_to_word.pkl", 'rb') as i2w:
    idx_to_word = pickle.load(i2w)

def predict_caption(photo):
    in_text = "startseq"
    max_len = 35
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')

        ypred =  model.predict([photo,sequence])
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        in_text+= ' ' +word
        
        if word =='endseq':
            break
        
        
    final_caption =  in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)
    
    return final_caption



def caption_this_image(image):

    enc = encode_image(image)
    caption = predict_caption(enc)
    
    return caption



st.markdown("""
<style>
body {
    background-color: #ADD8E6;
}
</style>
    """, unsafe_allow_html=True)


st.title("Automatic Image Captioning")
st.header("Upload an image to get a neural caption for it")
# To View Uploaded Image
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
if bool(image_file)==True :
    img_temp=plt.imread(image_file)
    st.image(img_temp) 
    fv=encode_image(image_file)
    caption=predict_caption(fv)
    st.write('##### PREDICTED CAPTION')
    st.write(caption)
    st.text("")
    st.text("")
    st.text("")        
    st.write("##### NOTE : This prediction is based on basic encoder-decoder model . I am currently working on it to improve it using advance encoder-decoder architecture such as Attention Models and Transformer . Please Stay Tuned . Thankyou ❤️")
    
else :
    ran_imageid=['1298295313_db1f4c6522','3203453897_6317aac6ff','3482974845_db4f16befa','3655155990_b0e201dd3c','3558370311_5734a15890']
    st.text("")
    st.text("")
    st.text("You can download some sample images by clicking on the below links :")
    st.write("[link](https://drive.google.com/file/d/19CkGmWcnXmPdabdg003vR43TFHWGZvnq/view?usp=sharing)")
    st.write("[link](https://drive.google.com/file/d/12nxEHF-DmJ5WnSudeZijkEOcxfIbNuTJ/view?usp=sharing)")
    st.text("")
    st.text("")
    st.markdown("OOPS !!!!!!!!!! You are not ready with some images 😬. Don't worry I have some images for you click on the below button and it will generate caption to a random image from a set of images. 😎")
    if st.button('Generate Caption for a random image') :
        ran_num=np.random.randint(0,len(ran_imageid))
        img_static_path=str(ran_imageid[ran_num])+'.jpg'
        img_temp=plt.imread(img_static_path)
        st.image(img_temp)
        fvs=encode_image(img_static_path)
        st.write('##### PREDICTED CAPTION')
        st.write(predict_caption(fvs))
        st.text("")
        st.text("")
        st.text("")        
        st.write("##### NOTE : This prediction is based on basic encoder-decoder model. I am currently working on it to improve it using advance encoder-decoder architecture such as Attention Models and Transformer . Please Stay Tuned . Thankyou ❤️")
            

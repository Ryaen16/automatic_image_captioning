# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 16:57:18 2024

@author: shilp
"""

import streamlit as st 
import Caption_it
import matplotlib.pyplot as plt
import numpy as np

st.title("Automatic Image Captioning")
st.header("Upload a image to get a neural caption for it")
# To View Uploaded Image
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
if bool(image_file)==True :
    img_temp=plt.imread(image_file)
    st.image(img_temp) 
    fv=Caption_it.encode_image(image_file)
    caption=Caption_it.predict_caption(fv)
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
    st.markdown("OOPS !!!!!!!!!! You are not ready with some images 😬. Don't worry i have some images for you click on the below button and it will generate caption to a random image from a set of images. 😎")
    if st.button('Generate Caption for a random image') :
        ran_num=np.random.randint(0,len(ran_imageid))
        img_static_path=str(ran_imageid[ran_num])+'.jpg'
        img_temp=plt.imread(img_static_path)
        st.image(img_temp)
        fvs=Caption_it.encode_image(img_static_path)
        st.write('##### PREDICTED CAPTION')
        st.write(Caption_it.predict_caption(fvs))
        st.text("")
        st.text("")
        st.text("")        
        st.write("##### NOTE : This prediction is based on basic encoder-decoder model. I am currently working on it to improve it using advance encoder-decoder architecture such as Attention Models and Transformer . Please Stay Tuned . Thankyou ❤️")
            

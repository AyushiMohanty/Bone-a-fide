import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
from skimage import transform
import time

# Load Models
st.header("Osteosarcoma Predictor")
model_binary = tf.keras.models.load_model('model_binary.h5')
model_multiclass = tf.keras.models.load_model('model_multiclass.h5')

#List of predictions
binary_prediction = []
multiclass_prediction = []

#Main Class
def main():
    
    #Dropdown menu for model selection
    model_chosen = st.selectbox('Choose the model you would like to use', 
                            ("Binary Model", "Multiclass Model"))
    
    if (model_chosen == 'Binary Model'):
        
        #Get uploaded file
        file_uploaded = st.file_uploader("Choose the file", type = ['jpg'])
        
        if file_uploaded is not None:
            
            #Resize img to be displayed
            image = Image.open(file_uploaded)
            resize_image = image.resize((512, 512))
            image_arr = plt.imread(file_uploaded)
            
            #Progress bar
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.1)
                progress.progress(i+1)
            
            #Get result
            result = predict_class_binary(image_arr, model_binary)
            
            #Get confidence %
            if (binary_prediction[1] > binary_prediction[0]):
                osteo_pred = "**Confidence:** {}".format(round((binary_prediction[1])*100,2))
                
            if (binary_prediction[1] < binary_prediction[0]):
                osteo_pred = "**Confidence:** {}".format(round((binary_prediction[0])*100,2))
            
            #Format page with 2 cols 
            col1, col2 = st.columns(2)
            
            #put img in first col
            with col1:
                st.image(resize_image)
            
            #put result and confidence in second col
            with col2:
                st.title(result)
                st.write(osteo_pred, "%")
    
    if (model_chosen == 'Multiclass Model'):
            
            #Get uploaded file
            file_uploaded = st.file_uploader("Choose the file", type = ['jpg'])
            
            if file_uploaded is not None:
                
                #resize img to be displayed
                image = Image.open(file_uploaded)
                resize_image = image.resize((512, 512))
                image_arr = plt.imread(file_uploaded)
                
                #progress bar
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.1)
                    progress.progress(i+1)
                
                #get result and confidences
                result = predict_class_multiclass(image_arr, model_multiclass)
                osteo_non_pred = "**Non-Tumor Confidence:** {}".format(round((multiclass_prediction[0])*100,2))
                osteo_nec_pred = "**Necrotic-Tumor Confidence:** {}".format(round((multiclass_prediction[1])*100,2))
                osteo_via_pred = "**Viable-Tumor Confidence:** {}".format(round((multiclass_prediction[2])*100,2))

                col1, col2 = st.columns(2)
                
                #Display img
                with col1:
                    st.image(resize_image)
                
                #Display result and confidences
                with col2:
                    st.title(result)
                    st.write(osteo_via_pred, "%")
                    st.write(osteo_nec_pred, "%")
                    st.write(osteo_non_pred, "%")
                
#Gets pred from binary model
def predict_class_binary(image, model):
    
    #Resize img 
    scaling_ratio = np.array([100, 100])/np.array([1024, 1024])
    anti_alias = np.any(scaling_ratio < 1)
    processed_img = transform.resize(image, [100, 100, 3], anti_aliasing=anti_alias)
    resized_img = processed_img.astype(np.float16)
    
    #make pred
    class_names = {0:'Osteosarcoma Negative', 1:'Osteosarcoma Positive'}
    reshaped_img = resized_img.reshape(1, 100, 100, 3)
    tensor_img = tf.constant(reshaped_img, dtype=tf.float16)
    prediction = model.predict(tensor_img)
    
    #put pred in list of preds
    binary_prediction.clear()
    for pred in prediction:
        for conf in pred:
            binary_prediction.append(conf)
    
    #Get class and return result
    pred = np.argmax(prediction)
    image_class = class_names[pred]

    result = image_class
    return result

#Gets pred from multiclass model
def predict_class_multiclass(image, model):
    
    #Resize img and put into tensor
    scaling_ratio = np.array([100, 100])/np.array([1024, 1024])
    anti_alias = np.any(scaling_ratio < 1)
    processed_img = transform.resize(image, [100, 100, 3], anti_aliasing=anti_alias)
    resized_img = processed_img.astype(np.float16)
    
    #make pred
    class_names = {0:'Non-Tumor', 1:'Necrotic-Tumor', 2:'Viable-Tumor'}
    reshaped_img = resized_img.reshape(1, 100, 100, 3)
    tensor_img = tf.constant(reshaped_img, dtype=tf.float16)
    prediction = model.predict(tensor_img)
    
    #put pred in list of preds
    multiclass_prediction.clear()
    for pred in prediction:
        for conf in pred:
            multiclass_prediction.append(conf)
    
    #get class and return result
    pred = np.argmax(prediction)
    image_class = class_names[pred]

    result = image_class
    return result

if __name__ == '__main__':
    main()
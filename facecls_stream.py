import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import PIL.Image as Image
import gdown
import os

os.makedirs("models", exist_ok=True)
if not os.path.isfile('models/face_cls_3.2.h5'):
    url = 'https://drive.google.com/uc?export=download&id=1V_0ywjuNS1KZTnGQ4Mu7XzHBBIOreFSl'
    output = 'models/face_cls_3.2.h5'
    gdown.download(url, output, quiet=False)
else:
    print("Model already exists")

model = load_model('models/face_cls_3.2.h5')
label_maps = {0: 'Ahegao', 1: 'Angry', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}

def preprocess_image(image):

    image = image.convert('L')
    
    image = image.resize((96, 96))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0
    return image

def predict_expression(image):
    preprocessed_img = preprocess_image(image)
    prediction = model.predict(preprocessed_img)
    return np.argmax(prediction, axis=1), prediction

st.title("Facial Expression Recognition with EfficientNet Transfer Learning")



uploaded_file = st.file_uploader("Upload an image...", type=['jpg', 'png', 'jpeg'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    if st.button('Predict'):
        expression_id, probabilities = predict_expression(image)
        expression = label_maps[expression_id[0]]
        
        st.markdown(f"<h1 style='color: red;'>{expression}</h1>", unsafe_allow_html=True)
        st.image(image, caption=f'Expression: {expression}', use_column_width=True)

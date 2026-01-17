import streamlit as st
from fastai.vision.all import *
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# to run: open terminal and type 'streamlit run streamlit_tutorial.py'
# rapid prototyping (for educational purposes)
# ctrl c to stop project

def extract_pet_breed(filename):
    parts = filename.split('_')

    filename = "_".join(parts[0:-1])
    return filename

pet_breed_model = load_learner("pet_breed_model_fastai284.pkl")

st.markdown("<h1 style='color: yellow;'>Pet Breed Classifier</h1>", unsafe_allow_html=True)
st.text("Created by Sirui Gong")

uploaded_file = st.file_uploader("Choose as image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    real_img = PILImage.create(uploaded_file)
    resized_img = real_img.resize((224, 224), Image.NEAREST)
    prediction = pet_breed_model.predict(resized_img)
    print(prediction)

    index = int(prediction[1])
    confidence_lvl = prediction[2][index] * 100
    print(f"confidence level: {confidence_lvl}")

    if confidence_lvl > 90:
        label = f"I am {confidence_lvl:.2f}% sure it is a {prediction[0]}"
    else:
        label = f"WARNING: I am not sure what this is.\n I am {confidence_lvl:.2f}% sure it is a {prediction[0]}"

    st.text(label)
    st.image(uploaded_file)
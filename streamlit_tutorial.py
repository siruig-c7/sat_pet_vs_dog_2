import streamlit as st
from fastai.vision.all import *
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# to run: open terminal and type 'streamlit run streamlit_tutorial.py'
# rapid prototyping (for educational purposes)
# ctrl c to stop project

def cat_or_dog(file_name):
    if file_name[0].isupper():
        return "CAT"
    else:
        return "DOG"

cat_vs_dog_model = load_learner("cat_vs_dog_model.pkl")

st.markdown("<h1 style='color: yellow;'>Cat or Dog Classifier</h1>", unsafe_allow_html=True)
st.text("Created by Sirui Gong")

uploaded_file = st.file_uploader("Choose as image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    real_img = PILImage.create(uploaded_file)
    resized_img = real_img.resize((224, 224), Image.NEAREST)
    prediction = cat_vs_dog_model.predict(resized_img)
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
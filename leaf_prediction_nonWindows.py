import streamlit as st
from fastai.vision.all import *

# to run: open terminal and type 'streamlit run streamlit_tutorial.py'
# rapid prototyping (for educational purposes)
# ctrl c to stop project

def extract_leaf_scientific_name(file_path):
    parts = str(file_path).split('/')
    name = str(parts[-2]).split('_')
    return name[-2].capitalize() + " " + name[-1]

leaf_prediction = load_learner("leaf_prediction_model_fastai284_b.pkl")

st.markdown("<h1 style='color: green;'>Leaf Prediction</h1>", unsafe_allow_html=True)
st.text("Created by Sirui Gong")

uploaded_file = st.file_uploader("Choose as image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    real_img = PILImage.create(uploaded_file)
    prediction = leaf_prediction.predict(real_img)
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
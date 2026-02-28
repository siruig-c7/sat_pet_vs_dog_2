import streamlit as st
import pandas as pd
import numpy as np
import pathlib
from fastai.tabular.all import *

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

titanic_model = load_learner("titanic_model.pkl")

st.markdown("<h1 style='color: yellow;'>Titanic Survival Rate</h1>", unsafe_allow_html=True)
st.text("Created by Sirui Gong")

name = st.text_input("What is your name: ")
age = st.number_input("How old are you: ", min_value=0.0, step=1.0)
sex = st.radio("What is your sex", ("male", "female"))
fare = st.number_input("Enter your ticket price: ", min_value=0.0, step=1.0)
pclass = st.radio("What is your passenger class: ", (1,2,3))
sib_sp = 0
parch = 0

if st.button("Submit"):
    new_passenger = {
        "Sex": [sex],
        "Age": [age],
        "Fare": [fare],
        "Pclass": [pclass],
        "SibSp": [sib_sp],
        "Parch": [parch]
    }
    new_df = pd.DataFrame(new_passenger)
    prediction = titanic_model.predict(new_df.iloc[0])
    print(prediction)
    st.write(f"{name} has a survival rate of {prediction[2][1].item()*100:.2f}%")
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
age = st.number_input("How old are you: ")
sex = st.radio("What is your sex", ("male", "female"))
fare = st.number_input("Enter your ticket price: ", min_value=0.0, step=1.0)
pclass = st.radio("What is your passenger class: ", (1,2,3))
sub_sp = 0
parch = 0
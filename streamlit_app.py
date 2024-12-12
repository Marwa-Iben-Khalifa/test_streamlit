import streamlit as st
import pandas as pd
st.title('Predict Categories 💶')
with st.expander('Data'):
  st.write('**Raw data**')
  df= pd.read_csv('https://raw.githubusercontent.com/Marwa-Iben-Khalifa/data/refs/heads/main/data.csv')
  df

import joblib
rf_model_loaded = joblib.load('rf_model.joblib')

import pickle

# Charger le OneHotEncoder
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

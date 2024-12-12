import streamlit as st
import pandas as pd
st.title('Predict Categories 💶')
with st.expander('Data'):
  st.write('**Raw data**')
  df= pd.read_csv('https://raw.githubusercontent.com/Marwa-Iben-Khalifa/data/refs/heads/main/data.csv')
  df

import pickle

# Charger le modèle et l'encodeur
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

# Titre de l'application
st.title('Predict Categories 💶')

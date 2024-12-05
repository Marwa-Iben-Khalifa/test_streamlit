import streamlit as st
import pandas as pd
st.title('🎈 App Name')
df= pd.read_csv('https://raw.githubusercontent.com/Marwa-Iben-Khalifa/data/refs/heads/main/data.csv')
df
st.write('Hello world!')

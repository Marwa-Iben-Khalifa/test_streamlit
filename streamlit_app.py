import streamlit as st
import pandas as pd
import joblib
import pickle
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import LabelEncoder
import string
import re

# Charger le modèle et l'encodeur
@st.cache_resource
def load_model_and_encoder():
    rf_model = joblib.load('rf_model.joblib')
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return rf_model, encoder

rf_model, encoder = load_model_and_encoder()

# Nettoyer le texte
def clean_transaction_text(text):
    unwanted_terms = ["cb", "www", "facture numero", "remittance_info", "reference", "code", "transaction_id", "tel"]
    for term in unwanted_terms:
        text = text.replace(term, "")
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
    text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '', text)
    text = re.sub(r'\b\d{2}/\d{2}/\d{4}\b', '', text)
    text = re.sub(r'\b\d{1,}\b', '', text)
    text = re.sub(r'[^a-zA-Z0-9çéèêëàâäîïôöùûüÿÇÉÈÊËÀÂÄÎÏÔÖÙÛÜŸ\s./-]', '', text)
    text = re.sub(r'\s*=\s*|\s*/\s*', ' ', text)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text if text else "Information non disponible"

# Afficher les données existantes
st.title('Predict Categories 💶')
with st.expander('Data'):
    st.write('**Raw data**')
    df = pd.read_csv('https://raw.githubusercontent.com/Marwa-Iben-Khalifa/data/refs/heads/main/data.csv')
    st.write(df)

# Charger le fichier des transactions à prédire
@st.cache_data
def load_transaction_data():
    return pd.read_csv('https://raw.githubusercontent.com/Marwa-Iben-Khalifa/data/refs/heads/main/date-non-categ.csv', on_bad_lines='skip', sep=',')

transaction_data = load_transaction_data()

# Nettoyage et prédiction
if st.button('Predict'):
    try:
        transaction_data['transaction_amount'] = transaction_data['debit'].astype(float) - transaction_data['credit'].astype(float)
        transaction_data["cleaned_document_label_translated_cleaned"] = transaction_data["label"].apply(clean_transaction_text)
        transaction_data['transaction_type'] = transaction_data['debit'].astype(float).apply(lambda x: 'debit' if x > 0 else 'credit')

        # Encodage
        le = LabelEncoder()
        for col in ['journal_code', 'transaction_type']:
            transaction_data[col + '_encoded'] = le.fit_transform(transaction_data[col].astype(str))

        label_onehot = encoder.transform(transaction_data[['cleaned_document_label_translated_cleaned']])
        features = csr_matrix(transaction_data[['transaction_amount', 'journal_code_encoded', 'transaction_type_encoded']].values)
        X_transaction = hstack([features, label_onehot])

        # Prédiction
        y_pred = rf_model.predict(X_transaction)
        transaction_data['predicted_label'] = y_pred

        st.write("**Predictions**")
        st.write(transaction_data[['label', 'predicted_label']])

    except Exception as e:
        st.error(f"Error during prediction: {e}")

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
    df = pd.read_csv('https://raw.githubusercontent.com/Marwa-Iben-Khalifa/data/refs/heads/main/date-non-categ.csv', on_bad_lines='skip', sep=',')
    st.write(df)

# Sélecteur pour choisir les lignes
st.subheader("Select Rows for Prediction")
select_all = st.checkbox("Select all rows")
selected_indices = []

if not select_all:
    selected_indices = st.multiselect(
        "Select specific rows (indices)", options=df.index, format_func=lambda x: f"Row {x}"
    )

# Filtrer les lignes à prédire
if select_all:
    transaction_data = df
else:
    if selected_indices:
        transaction_data = df.loc[selected_indices]
    else:
        st.warning("No rows selected. Please select rows to continue.")
        transaction_data = None

# Nettoyage et prédiction
if transaction_data is not None and st.button('Predict'):
    try:
        # Vérification des colonnes nécessaires
        required_columns = ['debit', 'credit', 'label', 'journal_code']
        missing_columns = [col for col in required_columns if col not in transaction_data.columns]
        if missing_columns:
            st.error(f"Les colonnes suivantes manquent dans les données : {', '.join(missing_columns)}")
            st.stop()

        # Préparation des données
        transaction_data['transaction_amount'] = transaction_data['debit'].astype(float) - transaction_data['credit'].astype(float)
        transaction_data["cleaned_document_label_translated_cleaned"] = transaction_data["label"].apply(clean_transaction_text)
        transaction_data['transaction_type'] = transaction_data['debit'].astype(float).apply(lambda x: 'debit' if x > 0 else 'credit')

        # Encode categorical variables
        le = LabelEncoder()
        categorical_cols = ['journal_code', 'transaction_type']
        for col in categorical_cols:
            transaction_data[col + '_encoded'] = le.fit_transform(transaction_data[col].astype(str))

        # Encodage
        label_onehot = encoder.transform(transaction_data[['cleaned_document_label_translated_cleaned']])
        
        # Préparation de la matrice des caractéristiques
        features = csr_matrix(transaction_data[['transaction_amount', 'journal_code_encoded', 'transaction_type_encoded']].values)
        X_transaction = hstack([features, label_onehot])

        # Prédiction
        y_pred = rf_model.predict(X_transaction)
        transaction_data['predicted_label'] = y_pred

        # Afficher les résultats dans le tableau
        st.write("**Predictions**")
        st.dataframe(transaction_data[['label', 'predicted_label']])

    except Exception as e:
        st.error(f"Erreur pendant la prédiction : {e}")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sqlalchemy import text
from datetime import datetime
from engine import engine  # Importer l'objet engine

# Fonction pour charger le modèle et le scaler
def load_model_and_scaler(model_name):
    with open(f'models/scaler_{model_name}.pkl', 'rb') as f:
        scaler = pickle.load(f)
    model = load_model(f'models/model_{model_name}.h5')  # Utiliser le fichier .h5
    return model, scaler

# Fonction pour effectuer des prédictions
def predict(model, scaler, data):
    data_scaled = scaler.transform(data)
    predictions = []
    length = 7
    n_features = 1
    first_eval_batch = data_scaled[-length:]
    current_batch = first_eval_batch.reshape((1, length, n_features))

    for i in range(len(data)):
        current_pred = model.predict(current_batch)[0]
        predictions.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[data_scaled[i]]], axis=1)
    
    true_predictions = scaler.inverse_transform(predictions)
    return true_predictions

# Fonction pour mettre à jour la base de données
def update_predictions(table_name):
    # Charger le modèle et le scaler pour chaque modèle
    models = {}
    scalers = {}
    for model_name in ['exxon', 'saudi', 'total', 'ecopetrol']:
        models[model_name], scalers[model_name] = load_model_and_scaler(model_name)
    
    # Récupérer la dernière date de la table
    query_last_date = text(f"SELECT MAX(date) FROM {table_name}")
    with engine.connect() as conn:
        result = conn.execute(query_last_date).fetchone()
        last_date = result[0]
    
    # Comparer avec la date d'aujourd'hui
    today_date = datetime.today().date()
    if last_date < today_date:
        # Lire les nouvelles données à partir de la dernière date
        query_data = text(f"SELECT date, close FROM {table_name} WHERE date > :last_date")
        df = pd.read_sql(query_data, engine, params={'last_date': last_date})
        
        # Si des nouvelles données sont présentes, faire la prédiction
        if not df.empty:
            df.set_index("date", inplace=True)
            df = df[["close"]]
            
            for model_name in ['exxon', 'saudi', 'total', 'ecopetrol']:
                predictions = predict(models[model_name], scalers[model_name], df)
                df[f'predicted_{model_name}'] = predictions
            
            # Mettre à jour la base de données avec les nouvelles prédictions
            df_to_update = df.reset_index()
            df_to_update.to_sql(table_name, engine, if_exists='append', index=False)
            st.write(f"Les prédictions ont été mises à jour jusqu'au {today_date}.")
        else:
            st.write("Pas de nouvelles données à mettre à jour.")
    else:
        st.write("Les prédictions sont déjà à jour.")

# Interface utilisateur Streamlit
st.title('Prédictions du Modèle de Prix du Pétrole')

# Ajouter une barre latérale pour la sélection de la table à mettre à jour
table_choice = st.sidebar.selectbox(
    'Choisissez une table à mettre à jour',
    ('exon', 'saudi', 'total', 'ecopetrol')
)

if st.sidebar.button('Mettre à jour les prédictions'):
    update_predictions(table_choice)

# Onglets pour chaque modèle
tabs = st.tabs(['Exxon', 'Saudi', 'Total', 'Ecopetrol'])

# Afficher les données pour chaque modèle
for tab, model_name in zip(tabs, ['exxon', 'saudi', 'total', 'ecopetrol']):
    with tab:
        st.header(f'Prédictions pour {model_name.capitalize()}')
        
        # Lire les données de la table sélectionnée
        df = pd.read_sql_table(f'table_{model_name}', engine)
        
        # Sélectionner une date
        selected_date = st.date_input(f'Sélectionnez une date pour {model_name.capitalize()}', min_value=df['date'].min(), max_value=df['date'].max())

        # Afficher les prédictions et les valeurs close pour la date sélectionnée
        if selected_date:
            df['date'] = pd.to_datetime(df['date'])
            selected_data = df[df['date'] == selected_date]
            
            if not selected_data.empty:
                st.write(selected_data[['date', 'close', f'predicted_{model_name}']])
                
                # Afficher le graphique
                plt.figure(figsize=(10, 6))
                plt.plot(selected_data['date'], selected_data['close'], label='Vérité terrain')
                plt.plot(selected_data['date'], selected_data[f'predicted_{model_name}'], label='Prédictions')
                plt.legend()
                st.pyplot(plt)
            else:
                st.write("Pas de données disponibles pour la date sélectionnée.")

# Sauvegarder le script sous le nom app.py et lancer l'application avec:
# streamlit run app.py

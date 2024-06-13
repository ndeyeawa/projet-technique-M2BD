from tensorflow.keras.models import load_model, save_model

# Liste des noms de modèles
model_names = ['exxon', 'saudi', 'total', 'ecopetrol']

for model_name in model_names:
    # Charger le modèle au format .pkl
    model = load_model(f'models/model_{model_name}.pkl')
    
    # Sauvegarder le modèle au format .h5
    model.save(f'models/model_{model_name}.h5')
    print(f"Model {model_name} converted to HDF5 format.")
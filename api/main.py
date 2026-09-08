import pandas as pd
import mlflow
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Initialisation de l'API
app = FastAPI(title="Détection de Fraude API", version="1.0")

# 2. Variable globale pour garder le modèle en mémoire (cache)
model = None

# 3. Définition du format de la requête entrante (Le "contrat" de l'API)
class Transaction(BaseModel):
    transaction_type: str
    transaction_amount: float
    origin_account_id: str
    origin_old_balance: float
    origin_new_balance: float
    destination_account_id: str
    destination_old_balance: float
    destination_new_balance: float

# 4. Au démarrage du serveur, on télécharge le cerveau (le modèle) depuis MLflow
@app.on_event("startup")
def load_model():
    global model
    import os
    print("⏳ Connexion à MLflow pour télécharger le dernier modèle...")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Détection_Fraude_Financière")
    mlflow.set_tracking_uri(mlflow_uri)
    
    try:
        # On cherche l'expérience par son nom
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            raise Exception("L'expérience MLflow est introuvable.")
            
        # On récupère automatiquement le TOUT DERNIER entraînement (Run)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id], 
            order_by=["start_time DESC"], 
            max_results=1
        )
        latest_run_id = runs.iloc[0].run_id
        print(f"✅ Modèle trouvé ! Run ID : {latest_run_id}")
        
        # On cherche le fichier model.xgb directement sur le disque partagé (plus fiable)
        import glob
        pattern = f"/mlflow/artifacts/*/{latest_run_id}/artifacts/model/model.xgb"
        matches = glob.glob(pattern)
        if not matches:
            raise Exception(f"Fichier model.xgb introuvable pour Run ID {latest_run_id}")
        model_path = matches[0]
        print(f"📁 Fichier modèle trouvé : {model_path}")
        
        # On charge le modèle directement en natif Booster XGBoost (sans passer par scikit-learn)
        # Cela évite tout conflit de version de scikit-learn
        model = xgb.Booster()
        model.load_model(model_path)
        print("🧠 Cerveau chargé avec succès. L'API est prête !")
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {str(e)}")

# 5. La route principale (Le point de contact avec le site e-commerce)
@app.post("/predict")
def predict_fraud(tx: Transaction):
    if model is None:
        raise HTTPException(status_code=500, detail="Le modèle n'est pas chargé.")
        
    # --- ETAPE CRUCIALE : REPRODUIRE LA COUCHE SILVER (Feature Engineering) ---
    # L'algorithme ne sait pas analyser un compte brut, il a besoin de nos 5 "features" magiques.
    
    # 1. is_origin_emptied
    is_origin_emptied = 1 if (tx.origin_old_balance > 0 and tx.origin_new_balance == 0) else 0
    
    # 2. origin_balance_error
    origin_balance_error = round(tx.origin_old_balance - tx.transaction_amount - tx.origin_new_balance, 2)
    
    # 3. destination_balance_error
    destination_balance_error = round(tx.destination_old_balance + tx.transaction_amount - tx.destination_new_balance, 2)
    
    # 4. amount_to_oldbalance_ratio
    amount_to_oldbalance_ratio = 0 if tx.origin_old_balance == 0 else round(tx.transaction_amount / tx.origin_old_balance, 4)
    
    # 5. is_merchant_destination
    is_merchant_destination = 1 if tx.destination_account_id.startswith('M') else 0

    # --- ETAPE CRUCIALE 2 : REPRODUIRE LA TRADUCTION BINAIRE (One-Hot Encoding) ---
    # On crée les 5 colonnes avec des zéros par défaut
    tx_types = {
        'transaction_type_CASH_IN': 0,
        'transaction_type_CASH_OUT': 0,
        'transaction_type_DEBIT': 0,
        'transaction_type_PAYMENT': 0,
        'transaction_type_TRANSFER': 0
    }
    # On met un '1' pour le type exact de notre transaction
    if f"transaction_type_{tx.transaction_type}" in tx_types:
        tx_types[f"transaction_type_{tx.transaction_type}"] = 1
        
    # --- ASSEMBLAGE FINAL ---
    # On rassemble tout dans l'ordre EXACT où XGBoost l'a appris (la couche Gold)
    features = {
        'transaction_amount': [tx.transaction_amount],
        'origin_old_balance': [tx.origin_old_balance],
        'origin_new_balance': [tx.origin_new_balance],
        'destination_old_balance': [tx.destination_old_balance],
        'destination_new_balance': [tx.destination_new_balance],
        'is_origin_emptied': [is_origin_emptied],
        'origin_balance_error': [origin_balance_error],
        'destination_balance_error': [destination_balance_error],
        'amount_to_oldbalance_ratio': [amount_to_oldbalance_ratio],
        'is_merchant_destination': [is_merchant_destination]
    }
    # On ajoute nos 5 colonnes de type de transaction
    features.update({k: [v] for k, v in tx_types.items()})
    
    # On convertit ça en tableau Pandas (le format que XGBoost comprend)
    df_features = pd.DataFrame(features)
    
    # --- PRÉDICTION ---
    # Le Booster natif attend une DMatrix
    dmatrix = xgb.DMatrix(df_features)
    probability = float(model.predict(dmatrix)[0])
    prediction = int(probability >= 0.5)
    
    return {
        "is_fraud": bool(prediction),
        "fraud_probability": round(float(probability) * 100, 2),
        "status": "Alerte Rouge 🚨" if prediction == 1 else "Autorisé ✅"
    }

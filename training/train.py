import os
import clickhouse_connect
import pandas as pd
import mlflow
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("1. Connexion à ClickHouse pour récupérer la couche GOLD...")
ch_host = os.getenv('CLICKHOUSE_HOST', 'clickhouse')
ch_port = int(os.getenv('CLICKHOUSE_HTTP_PORT', '8123'))
ch_user = os.getenv('CLICKHOUSE_USER', 'default')
ch_password = os.getenv('CLICKHOUSE_PASSWORD', 'password')

client = clickhouse_connect.get_client(host=ch_host, port=ch_port, username=ch_user, password=ch_password)

# On télécharge la table Gold (fct_transactions_features) en mémoire Pandas
# Attention : Dans la vraie vie avec des milliards de lignes, on ferait l'entraînement par lots, 
# mais avec 6 millions de lignes, Pandas/XGBoost peuvent gérer ça en RAM.
query = "SELECT * FROM fraud_detection_gold.fct_transactions_features"
df = client.query_df(query)

print(f"✅ Données récupérées : {len(df)} lignes prêtes pour l'IA.")

print("2. Préparation des données (Features vs Target)...")
# X = Nos indices (features). Y = La réponse à trouver (Fraude 1 ou 0).
# On exclut "step" (le temps n'est pas une feature causale) et "is_fraud" (c'est la réponse !)
X = df.drop(columns=['step', 'is_fraud'])

# XGBoost ne lit que des nombres. Or "transaction_type" est du texte (CASH_OUT, TRANSFER...).
# On utilise la méthode de "One-Hot Encoding" de Pandas (get_dummies) pour transformer 
# le texte en colonnes binaires (ex: type_TRANSFER = 1, type_CASH_OUT = 0).
X = pd.get_dummies(X, columns=['transaction_type'])

y = df['is_fraud']

# On coupe les données : 80% pour l'entraînement, 20% pour l'examen final (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✅ Découpage terminé : {len(X_train)} lignes pour s'entraîner, {len(X_test)} pour le test.")

print("3. Configuration de MLflow (Notre carnet de notes automatique)...")
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow_exp = os.getenv("MLFLOW_EXPERIMENT_NAME", "Détection_Fraude_Financière")
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment(mlflow_exp)

# C'est la magie de MLflow : il va tout noter tout seul (les paramètres, le modèle final, l'importance des variables)
mlflow.xgboost.autolog()

print("4. Lancement du cerveau (XGBoost) ! Ça va chauffer...")
with mlflow.start_run() as run:
    # Création du modèle. 
    # n_estimators=100 : Il va créer 100 arbres de décisions qui vont voter ensemble.
    # max_depth=6 : Chaque arbre aura une profondeur maximale de 6 questions.
    model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.1, 
        random_state=42,
        eval_metric='logloss'
    )
    
    # L'entraînement en lui-même (L'IA apprend)
    model.fit(X_train, y_train)
    
    print("5. L'entraînement est fini ! On passe à l'examen (Test)...")
    # L'IA passe l'examen sur les 20% de données qu'elle n'a jamais vu
    y_pred = model.predict(X_test)
    
    # On calcule la note de l'examen (Précision globale)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎓 Précision du modèle : {acc * 100:.2f}%")
    
    # MLflow a déjà enregistré le modèle grâce à "autolog()", mais on peut rajouter nos propres notes :
    mlflow.log_metric("accuracy_finale", acc)
    
    print("\nRapport détaillé :")
    print(classification_report(y_test, y_pred))

print(f"🎉 Terminé ! Le modèle est sauvegardé dans MLflow (Run ID: {run.info.run_id}).")

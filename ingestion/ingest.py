import os
import pandas as pd
import clickhouse_connect
import time

print("Connexion à ClickHouse...")
host = os.getenv('CLICKHOUSE_HOST', 'clickhouse')
port = int(os.getenv('CLICKHOUSE_HTTP_PORT', '8123'))
user = os.getenv('CLICKHOUSE_USER', 'default')
password = os.getenv('CLICKHOUSE_PASSWORD', 'password')

client = clickhouse_connect.get_client(host=host, port=port, username=user, password=password)
print("Connecté !")

# Définition du schéma de la table raw_transactions
create_table_query = """
CREATE TABLE IF NOT EXISTS raw_transactions (
    step UInt32,
    type String,
    amount Float64,
    nameOrig String,
    oldbalanceOrg Float64,
    newbalanceOrig Float64,
    nameDest String,
    oldbalanceDest Float64,
    newbalanceDest Float64,
    isFraud UInt8,
    isFlaggedFraud UInt8
) ENGINE = MergeTree()
ORDER BY step
"""
client.command(create_table_query)
print("Table 'raw_transactions' vérifiée/créée.")

csv_path = '/app/data/raw/paysim.csv'
print(f"Démarrage de l'ingestion depuis {csv_path}...")

# Lecture par morceaux (chunks) pour ne pas saturer la mémoire (493 Mo)
chunk_size = 100000
chunks = pd.read_csv(csv_path, chunksize=chunk_size)

total_rows = 0
start_time = time.time()

for chunk in chunks:
    # Insérer le DataFrame entier dans ClickHouse
    client.insert_df('raw_transactions', chunk)
    total_rows += len(chunk)
    print(f"--> {total_rows} lignes insérées jusqu'à présent...")

end_time = time.time()
print(f"✅ Ingestion terminée ! {total_rows} lignes insérées en {end_time - start_time:.2f} secondes.")

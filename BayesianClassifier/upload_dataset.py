import pandas as pd
from pymongo import MongoClient
import math
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Cargar el .csv en un DataFrame
df = pd.read_csv("fraud_credit_card.csv", sep=",", quotechar='"')
print(f"El dataset tiene {len(df)} registros.\n")

# Reemplazar ',' por '.' en 'amount'
df["amount"] = df["amount"].str.replace(",", ".").astype(float)

# Discretización de 'amount'
# 0 <= x < 10  : 107373 (very low)
# 10 <= x < 50 : 386499 (low)
# 50 <= x < 100: 80210  (medium)
# 100 <= x     : 20561  (high)
bins = [0, 10, 50, 100, float("inf")]
labels = ["very low", "low", "medium", "high"]
df["amount_bin"] = pd.cut(df["amount"], bins=bins,
                          labels=labels, include_lowest=True)

# Convertir 'age' a número, marcando 'U' como None
df["age"] = df["age"].apply(lambda x: 
    x.strip('\'') if x != "'U'" else "U")

# Quitar quotes de 'gender', 'U' es 'unknown'
# NOTE: Cuando 'age' es 'U', 'gender' siempre es 'E' (7 de estos casos son fraude).
df["gender"] = df["gender"].apply(
    lambda x: x.strip('\'') if x != "'U'" else "U")

# Quitar quotes de 'category', marcando 'U' como None
df["category"] = df["category"].apply(
    lambda x: x.strip('\'') if x != "'U'" else None)

# Convertir 'fraud' a 'yes' or 'no'
df["fraud"] = df["fraud"].apply(
    lambda x: 'yes' if x == 1 else 'no')

# Borrar columnas que no son relevantes para la inferencia.
# Solo hay 1 único zipcodeOri y 1 único zipMerchant.
# NOTE: Hay 50 distintos merchants, podría ser relevante.
# Hay 180 distintos steps, por lo que no es relevante.
# Hay 4112 distintos customers, por lo que no es relevante.
df = df.drop(columns=["step", "customer",
             "zipcodeOri", "merchant", "zipMerchant"])

# Ordenar columnas
df = df[['age', 'gender', 'category', 'amount_bin', 'fraud']]

print(f"Descripción del dataset:\n{df.describe()}\n")

print(f"Primeras 5 filas:\n{df[0:5]}\n")

# TODO: Guardar esto en un lugar más apropriado (see .env)

# Conectar a MongoDB Atlas
client = MongoClient(os.environ["ATLASMONGODB_CONNECTION_STRING"])
db = client["fraud_db"]
collection = db["transactions"]

# collection.delete_many({})
collection.drop()
print("Colección limpiada.", flush=True)

# collection.insert_many(df.to_dict("records"))
# Insertar en batches
BATCH_SIZE = 10000
total = len(df)
n_batches = math.ceil(total / BATCH_SIZE)


def print_progress(current, total, bar_length=40):
    percent = current / total
    filled = int(bar_length * percent)
    bar = "█" * filled + "-" * (bar_length - filled)
    print(f"\rProgreso: |{bar}| {current}/{total} docs", end="", flush=True)


for i in range(n_batches):
    start = i * BATCH_SIZE
    end = min((i + 1) * BATCH_SIZE, total)
    batch = df.iloc[start:end].to_dict("records")
    collection.insert_many(batch)
    print_progress(end, total)
print("\nCarga completa.")

# index_dataset.py

from pymongo import MongoClient
from collections import defaultdict
import math
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = MongoClient(os.environ["ATLASMONGODB_CONNECTION_STRING"])
db = client["fraud_db"]
original = db["transactions"]
indexed = db["transactions_indexed"]
meta = db["metadata"]

BATCH_SIZE = 10000


variables = ["age", "gender", "category", "amount_bin", "fraud"]


def compute_cardinalidades():
    cardinalidades = defaultdict(dict)
    reverse_maps = defaultdict(dict)
    for doc in original.find({}, {"_id": 0}):
        # for k, v in doc.items():
        #     if k == "_id":
        #         continue
        #     if v not in cardinalidades[k]:
        #         idx = len(cardinalidades[k])
        #         cardinalidades[k][v] = idx
        #         reverse_maps[k][idx] = v
        for key in variables:
            value = doc[key]
            if value not in cardinalidades[key]:
                idx = len(cardinalidades[key])
                cardinalidades[key][value] = idx
                reverse_maps[key][idx] = value
    return dict(cardinalidades), dict(reverse_maps)


def store_metadata(cardinalidades):
    meta.drop()
    for var, cardinality in cardinalidades.items():
        meta.insert_one({"variable": var, "mapping": cardinality})


def index_and_store(cardinalidades):
    total = original.count_documents({})
    indexed.drop()
    #for i in range(0, total, BATCH_SIZE):
    for i in range(math.ceil(total / BATCH_SIZE)):
        start = i * BATCH_SIZE
        end = min((i + 1) * BATCH_SIZE, total)
        print_progress(end, total)
        batch = list(original.find({}, {}, skip=start, limit=BATCH_SIZE))
        for doc in batch:
            for k in cardinalidades:
                doc[k] = cardinalidades[k][doc[k]]
        indexed.insert_many(batch)
        # print(f"{min(i + BATCH_SIZE, total)}/{total} docs indexed", flush=True)


def print_progress(current, total, bar_length=40):
    percent = current / total
    filled = int(bar_length * percent)
    bar = "█" * filled + "-" * (bar_length - filled)
    print(f"\rProgreso: |{bar}| {current}/{total} docs", end="", flush=True)


if __name__ == "__main__":
    print("Computing cardinalidades...", flush=True)
    cardinalidades, _ = compute_cardinalidades()
    print("Storing metadata...", flush=True)
    store_metadata(cardinalidades)
    print("Indexing dataset...", flush=True)
    index_and_store(cardinalidades)
    print("\nDone.", flush=True)

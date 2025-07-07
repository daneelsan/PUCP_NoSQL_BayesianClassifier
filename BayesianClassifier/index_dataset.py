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
precomputed = db["precomputed"]
cardinalities_col = db["cardinalities"]

BATCH_SIZE = 10000


variables = ["age", "gender", "category", "amount_bin", "fraud"]


def compute_cardinalities():
    cardinalities = defaultdict(dict)
    reverse_maps = defaultdict(dict)
    for doc in original.find({}, {"_id": 0}):
        # for k, v in doc.items():
        #     if k == "_id":
        #         continue
        #     if v not in cardinalities[k]:
        #         idx = len(cardinalities[k])
        #         cardinalities[k][v] = idx
        #         reverse_maps[k][idx] = v
        for key in variables:
            value = doc[key]
            if value not in cardinalities[key]:
                idx = len(cardinalities[key])
                cardinalities[key][value] = idx
                reverse_maps[key][idx] = value
    return dict(cardinalities), dict(reverse_maps)


def store_cardinalties(cardinalities):
    cardinalities_col.drop()
    for var, cardinality in cardinalities.items():
        cardinalities_col.insert_one({"variable": var, "mapping": cardinality})


def index_and_store(cardinalities):
    total = original.count_documents({})
    indexed.drop()
    # for i in range(0, total, BATCH_SIZE):
    for i in range(math.ceil(total / BATCH_SIZE)):
        start = i * BATCH_SIZE
        end = min((i + 1) * BATCH_SIZE, total)
        print_progress(end, total)
        batch = list(original.find({}, {}, skip=start, limit=BATCH_SIZE))
        for doc in batch:
            for k in cardinalities:
                doc[k] = cardinalities[k][doc[k]]
        indexed.insert_many(batch)
        # print(f"{min(i + BATCH_SIZE, total)}/{total} docs indexed", flush=True)


def precompute_counts_and_store(cardinalities):
    precomputed.drop()
    # precomputing Naive Bayes counts
    target_variable = "fraud"
    target_values = list(cardinalities[target_variable].values())
    for target_val in target_values:
        count = indexed.count_documents({target_variable: target_val})
        precomputed.insert_one({target_variable: target_val, "count": count})
    for var, cardinality in cardinalities.items():
        for val in cardinality.values():
            precomputed.insert_one({var: val, "count": indexed.count_documents({var: val})})
            for target_val in target_values:
                count = indexed.count_documents({var: val, target_variable: target_val})
                precomputed.insert_one({var: val, target_variable: target_val, "count": count})


def print_progress(current, total, bar_length=40):
    percent = current / total
    filled = int(bar_length * percent)
    bar = "█" * filled + "-" * (bar_length - filled)
    print(f"\rProgreso: |{bar}| {current}/{total} docs", end="", flush=True)


if __name__ == "__main__":
    print("Computing cardinalities...")
    cardinalities, _ = compute_cardinalities()
    print("Storing cardinalities...")
    store_cardinalties(cardinalities)
    print("Indexing dataset...")
    index_and_store(cardinalities)
    print("\nPrecomputing counts...")
    precompute_counts_and_store(cardinalities)
    print("\nDone.")

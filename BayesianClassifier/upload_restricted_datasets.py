import os
import math
import random
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

FRACTIONS = [0.1, 0.3, 0.5, 0.7, 0.9]
DB_NAME = "fraud_db"
SOURCE_COLLECTION = "transactions_indexed"

random.seed(42)

def main():
    client = MongoClient(os.environ["ATLASMONGODB_CONNECTION_STRING"])
    db = client[DB_NAME]
    source = db[SOURCE_COLLECTION]

    total = source.estimated_document_count()
    print(f"Total documents in source collection: {total}")

    for fraction in FRACTIONS:
        size = math.floor(total * fraction)
        temp_name = f"transactions_sampled_{int(fraction * 100)}"

        print(
            f"\n[{int(fraction * 100)}%] Sampling {size} documents -> {temp_name} ..."
        )

        # Drop the collection if it exists
        if temp_name in db.list_collection_names():
            db.drop_collection(temp_name)

        # Sample with allowDiskUse (using aggregation)
        pipeline = [{"$sort": {"_id": 1}}, {"$sample": {"size": size}}]
        sampled_docs = list(source.aggregate(pipeline, allowDiskUse=True))

        if sampled_docs:
            db[temp_name].insert_many(sampled_docs)
            print(
                f"Uploaded {len(sampled_docs)} documents to '{temp_name}' collection."
            )
        else:
            print("Warning: No documents sampled!")

    print("\n✅ All restricted datasets uploaded.")


if __name__ == "__main__":
    main()

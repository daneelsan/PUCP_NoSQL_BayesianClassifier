import random
import pandas as pd
import sys # Import the sys module
from bayes_classifier import BayesianClassifier
from bayes_classifier import available_hypotheses

# Parameters to test
PARAMETERS = {
    "category": [
        "es_transportation",
        "es_health",
        "es_otherservices",
        "es_food",
        "es_hotelservices",
        "es_barsandrestaurants",
        "es_tech",
        "es_sportsandtoys",
        "es_wellnessandbeauty",
        "es_hyper",
        "es_fashion",
        "es_home",
        "es_contents",
        "es_travel",
        "es_leisure",
    ],
    "gender": ["M", "F", "E", "U"],
    "age": ["0", "1", "2", "3", "4", "5", "6", "U"],
    "amount_bin": ["very low", "low", "medium", "high"],
}

# How many unique combinations to test
NUM_COMBINATIONS = 30
REPEATS_PER_COMBINATION = 10


results = []

def run_benchmark_classifier(transactions_db_name):
    classifier = BayesianClassifier(transactions_db_name=transactions_db_name, use_lru_cache=False)

    # Random but reproducible combinations
    random.seed(42)

    combinations = [
        {
            "category": random.choice(PARAMETERS["category"]),
            "gender": random.choice(PARAMETERS["gender"]),
            "age": random.choice(PARAMETERS["age"]),
            "amount_bin": random.choice(PARAMETERS["amount_bin"]),
        }
        for _ in range(NUM_COMBINATIONS)
    ]
    #print(combinations)
    for idx, evidence in enumerate(combinations):
        for i in range(REPEATS_PER_COMBINATION):
            result, prob, elapsed = classifier.classify(evidence)

            results.append(
                {
                    "test_id": idx,
                    "repeat": i,
                    "category": evidence["category"],
                    "gender": evidence["gender"],
                    "age": evidence["age"],
                    "amount_bin": evidence["amount_bin"],
                    "fraud_prediction": result,
                    "probability": prob,
                    "elapsed": elapsed,
                    "transactions_db_name": transactions_db_name,
                }
            )
            print(f"Test {idx}-{i}: {elapsed:.4f}s")

output_filename = "benchmarks/benchmark_results_fractions.csv"

run_benchmark_classifier("transactions_indexed")
run_benchmark_classifier("transactions_sampled_10")
df = pd.DataFrame(results)
df.to_csv(output_filename, index=False) # Use the dynamically generated filename

print(f"Benchmarking complete. Saved to {output_filename}.")

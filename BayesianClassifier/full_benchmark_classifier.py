# benchmark_classifier.py
import random
import time
import csv
import os
from bayes_classifier import BayesianClassifier, available_hypotheses
from dotenv import load_dotenv

# Load environment variables (if needed)
load_dotenv()

# --- Benchmark Parameters ---
REPEATS = 10  # Number of times each test is repeated
NUM_COMBINATIONS = 20  # Number of different evidence samples per dataset size
FRACTIONS = [0.1 * i for i in range(1, 11)]  # Fractions from 10% to 100%

# Define parameter values for generating evidence samples
parameters = {
    "category": [
        "es_transportation",
        "es_health",
        "es_otherservices",
        "es_food",
        "es_hotelservices",
        "es_barsandrestaurants",
        "es_travel",
        "es_leisure",
    ],
    "gender": ["M", "F", "E", "U"],
    "age": ["4", "2", "3", "5"],
    "amount_bin": ["very low", "low", "medium", "high"],
}


def generate_random_evidences(n):
    """Return n randomized evidence dictionaries using the parameters above."""
    random.seed(42)
    evidences = []
    for _ in range(n):
        evidence = {key: random.choice(values) for key, values in parameters.items()}
        evidences.append(evidence)
    return evidences


evidences = generate_random_evidences(NUM_COMBINATIONS)

# Prepare to collect benchmark results
results = []

# Benchmark each hypothesis across each dataset fraction
for hypo_name, hypo_structure in available_hypotheses.items():
    print(f"\nTesting hypothesis: {hypo_name}")

    classifier = BayesianClassifier()
    classifier.set_hypothesis(hypo_structure)

    for i, evidence in enumerate(evidences):
        total_time = 0.0
        for rep in range(REPEATS):
            try:
                fraud_prediction, prob, elapsed = classifier.classify(evidence)
            except Exception as e:
                print(f"Error on evidence {evidence}: {e}")
                continue
            total_time += elapsed

        avg_time = total_time / REPEATS
        print(f"   Evidence {i+1}: average time = {avg_time:.4f}s")
        results.append(
            {
                "hypothesis": hypo_name,
                "evidence_id": i + 1,
                "avg_time_sec": avg_time,
                "fraud_prediction": fraud_prediction,
                "probability": prob,
                "age": evidence["age"],
                "gender": evidence["gender"],
                "category": evidence["category"],
                "amount_bin": evidence["amount_bin"],
            }
        )

# Write results to CSV for later plotting
csv_filename = "benchmarks/full_benchmark_results.csv"
with open(csv_filename, "w", newline="") as f:
    fieldnames = [
        "hypothesis",
        "evidence_id",
        "avg_time_sec",
        "fraud_prediction",
        "probability",
        "age",
        "gender",
        "category",
        "amount_bin",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nBenchmark complete. Results saved to {csv_filename}.")

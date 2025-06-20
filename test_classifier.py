# test_classifier.py
from bayes_classifier import BayesianClassifier
from bayes_classifier import available_hypotheses

classifier = BayesianClassifier()


def report(evidence):
    print(f"Evidence: {evidence}")
    result, prob, time = classifier.classify(evidence)
    print(f"Fraud?: {result} (probability: {prob:.6f}, took: {time:4f}s)\n")


# No fraud
report({
    "gender": "M",
    "age": "3",
    "category": "es_health",
    "amount_bin": "medium",
})

# Fraud!
report({
    "gender": "F",
    "age": "2",
    "category": "es_travel",
    "amount_bin": "medium",
})

# Change hypothesis
classifier.set_hypothesis({"fraud": ["amount_bin"], "gender": ["age"]})

report({
    "gender": "F",
    "age": "2",
    "category": "es_travel",
    "amount_bin": "medium",
})

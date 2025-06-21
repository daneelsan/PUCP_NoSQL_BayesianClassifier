from collections import defaultdict
from pymongo import MongoClient
import time
import os
from dotenv import load_dotenv


available_hypotheses = {
    "Naive Bayes": {"fraud": ["age", "gender", "amount_bin", "category"]},
    "Structured (fraud->amount_bin, gender->age)": {
        "fraud": ["amount_bin"],
        "gender": ["age"],
    },
}


class BayesianClassifier:
    def __init__(
        self,
        db_name="fraud_db",
        alpha=1.0,
    ):
        # Load environment variables from .env file
        load_dotenv()
        self.client = MongoClient(
            os.environ["ATLASMONGODB_CONNECTION_STRING"], readConcernLevel="local"
        )
        self.db = self.client[db_name]
        self.data_collection = self.db["transactions_indexed"]
        self.meta = self.db["metadata"]
        self.precomputed = self.db["precomputed"]
        self.alpha = alpha
        self.cardinalities = self.load_cardinalities()
        # print(f'cardinalities: {self.cardinalities}')
        self.variables = list(self.cardinalities.keys())
        # print(f'variables: {self.variables}')
        self.target_variable = "fraud"
        self.parents = defaultdict(list)
        # Naive Bayes Hyphothesis
        self.set_hypothesis(available_hypotheses["Naive Bayes"])
        # for var in self.variables:
        #     if var != self.target_variable:
        #         self.parents[var].append(self.target_variable)
        # print(f'parents: {self.parents}')
        self.ensure_indexes()

    def load_cardinalities(self):
        result = {}
        for doc in self.meta.find({}):
            result[doc["variable"]] = doc["mapping"]
        return result

    def ensure_indexes(self):
        for var in self.variables:
            if var == self.target_variable:
                continue
            for parent in self.parents[var]:
                self.data_collection.create_index([(parent, 1), (var, 1)])

    def count_occurrencies(self, variable, value, context):
        context[variable] = value
        res = self.precomputed.find_one(context, { "count": 1 })
        if res is not None:
            count = res['count']
        else:
            count = self.data_collection.count_documents(context)
        return count

    def conditional_probability(self, variable, value, context, total):
        k = len(self.cardinalities[variable])
        count = self.count_occurrencies(variable, value, context)
        return (count + self.alpha) / (total + self.alpha * k)

    def compute_joint_distribution(self, evidence_indexed):
        resultados = []
        # Obtain all possible (indexed) values for the target variable
        target_values = list(self.cardinalities[self.target_variable].values())
        for target_val in target_values:
            context = evidence_indexed.copy()
            context[self.target_variable] = target_val
            prob_total = 1.0
            # Traverse all variables to compute total probability
            for var in self.variables:
                parents = self.parents.get(var, [])
                # Obtain the parent(s) context (indexed values)
                context_parents = {p: context[p] for p in parents}
                total = self.data_collection.count_documents(context_parents)
                # Compute P(var=val | parents)
                p = self.conditional_probability(
                    var, context[var], context_parents, total
                )
                prob_total *= p
            resultados.append((target_val, prob_total))
        return resultados

    def classify(self, evidence):
        # Start recording how long it took
        time_start = time.time()
        # Convert all values to their indices
        evidence_indexed = {}
        for var, val in evidence.items():
            val_indexed = self.cardinalities[var][val]
            evidence_indexed[var] = val_indexed
        # Compute the joint_distribution
        distribution = self.compute_joint_distribution(evidence_indexed)
        distribution.sort(key=lambda x: x[1], reverse=True)
        pred_clase, prob = distribution[0]
        # Record how long it took
        time_total = time.time() - time_start
        # print(f'Time: {time_total:.4f}s')
        # Return a tuple
        return (pred_clase == "1" or pred_clase == 1, prob, time_total)

    def set_hypothesis(self, hypothesis, target_variable="fraud"):
        self.target_variable = target_variable
        self.parents = defaultdict(list)
        for var in self.variables:
            children = hypothesis.get(var, [])
            for c in children:
                self.parents[c].append(var)
        print(f"Changed hypothesis to: {self.parents}")

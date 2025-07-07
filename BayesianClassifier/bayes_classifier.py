from collections import defaultdict
from itertools import combinations
from scipy.special import gammaln
from itertools import product
from pymongo import MongoClient
import time
import os
from dotenv import load_dotenv
from functools import lru_cache
import math


available_hypotheses = {
    # Baseline models
    "Naive Bayes": {"fraud": ["age", "gender", "amount_bin", "category"]},
    # "Structured (fraud->amount_bin, gender->age)": {
    #     "fraud": ["amount_bin"],
    #     "gender": ["age"]
    # },
    # New fraud-centric hypotheses
    "Fraud as Root Cause": {
        # "fraud": [],  # Root node
        "category": ["fraud", "gender"],
        "amount_bin": ["fraud", "age"],
    },
    "Fraud as Mediator": {
        "fraud": ["amount_bin", "category"],
        "amount_bin": ["age"],
        "category": ["gender"],
    },
    "Demographic-Driven Fraud": {
        "fraud": ["age", "gender"],  # Fraud depends on demographics
        "amount_bin": ["age"],
        "category": ["gender", "amount_bin"],
    },
    "High-Risk Category Focus": {
        "fraud": ["category", "amount_bin"],  # Direct category-amount link
        "category": ["gender"],
        "amount_bin": ["age"],
    },
    # Hybrid expert-knowledge model
    "Known Fraud Patterns": {
        "fraud": ["category", "amount_bin", "gender"],  # e.g., Male + es_travel + high
        "amount_bin": ["age"],
        # "category": []  # Independent
    },
}


class BayesianClassifier:
    def __init__(
        self,
        alpha=1.0,
        hyphothesis_name="Naive Bayes",
        transactions_db_name="transactions_indexed",
        use_lru_cache=True
    ):
        # Load environment variables from .env file
        load_dotenv()
        self.client = MongoClient(
            os.environ["ATLASMONGODB_CONNECTION_STRING"], readConcernLevel="local"
        )
        self.db = self.client["fraud_db"]
        self.data_collection = self.db[transactions_db_name]
        self.N = self.data_collection.estimated_document_count()
        self.precomputed = self.db["precomputed"]
        self.alpha = alpha
        self.cardinalities = self.load_cardinalities()
        # print(f'cardinalities: {self.cardinalities}')
        self.variables = list(self.cardinalities.keys())
        # print(f'variables: {self.variables}')
        self.target_variable = "fraud"
        self.parents = defaultdict(list)

        self.set_hypothesis(available_hypotheses[hyphothesis_name])

        self.use_lru_cache = use_lru_cache
        # NOTE: Uncomment to benchmark
        #self.data_collection.drop_indexes()

    def load_cardinalities(self):
        result = {}
        cardinalities_col = self.db["cardinalities"]
        for doc in cardinalities_col.find({}):
            result[doc["variable"]] = doc["mapping"]
        return result

    def ensure_indexes(self):
        for var in self.variables:
            if var == self.target_variable:
                continue
            for parent in self.parents[var]:
                self.data_collection.create_index([(parent, 1), (var, 1)])

    # The actual method decorated with lru_cache
    # It must take 'self' as the first argument, but lru_cache expects a hashable argument.
    # We will wrap the original method to handle the dict argument for caching.
    @lru_cache(maxsize=10000) # You can adjust maxsize based on expected unique queries
    def _cached_compute_counts(self, evidence_tuple):
        """
        Helper method for caching compute_counts.
        Takes a hashable tuple representation of evidence.
        """
        evidence = dict(evidence_tuple) # Convert tuple back to dict
        
        res = self.precomputed.find_one(evidence, {"count": 1})
        if res is not None:
            count = res["count"]
        else:
            count = self.data_collection.count_documents(evidence)
        return count

    def compute_counts(self, evidence):
        """
        The public interface for compute_counts.
        Converts the dictionary evidence to a hashable tuple for caching.
        """
        # NOTE: Uncomment to benchmark
        # return self.data_collection.count_documents(evidence)
        # Convert the dictionary (which is not hashable) to a sorted tuple of (key, value) pairs
        # so it can be used as a cache key.
        hashable_evidence = tuple(sorted(evidence.items()))
        if self.use_lru_cache:
            return self._cached_compute_counts(hashable_evidence)
        else:
            return BayesianClassifier._cached_compute_counts.__wrapped__(self, hashable_evidence)

    def conditional_probability(self, variable, value, context, total):
        k = len(self.cardinalities[variable])
        context[variable] = value
        count = self.compute_counts(context)
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
                # If the variable is not conditioned, then the total is the size of the dataset
                if len(context_parents) == 0:
                    total = self.N
                else:
                    total = self.compute_counts(context_parents)
                # Compute P(var=val | parents)
                p = self.conditional_probability(
                    var, context[var], context_parents, total
                )
                prob_total *= p
            resultados.append((target_val, prob_total))
        return resultados

    def classify(self, evidence, apply_index=True):
        # Start recording how long it took
        time_start = time.time()
        # Convert all values to their indices
        evidence_indexed = {}
        if apply_index:
            for var, val in evidence.items():
                val_indexed = self.cardinalities[var][val]
                evidence_indexed[var] = val_indexed
        else:
            evidence_indexed = evidence
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
        """ "
        parents should have the form {child1: [par11, par12, ...], child2: [par21, par22, ...]}
        E.g. {'age': ['fraud'], 'gender': ['fraud'], 'amount_bin': ['fraud'], 'category': ['fraud']})
        """
        self.target_variable = target_variable
        self.parents = defaultdict(list)
        for var in self.variables:
            children = hypothesis.get(var, [])
            for c in children:
                self.parents[c].append(var)
        self.ensure_indexes()
        #print(f"Changed hypothesis to: {self.parents}")

    def k2_score(self, child, parents):
        """
        Compute the K2 score for a child variable given a list of parent variables.

        K2 score formula:
        log P(D|G) = Σ_i [ log(Γ(α*r_i)) - log(Γ(N_i + α*r_i)) + Σ_j [log(Γ(N_ij + α)) - log(Γ(α))] ]

        where:
        - r_i is the number of values that variable X_i can take
        - N_i is the number of instances where the parents of X_i take their i-th configuration
        - N_ij is the number of instances where X_i takes its j-th value and parents take i-th configuration
        """
        r = len(self.cardinalities[child])  # Number of values child can take
        total_score = 0

        # Get all possible parent value combinations
        if not parents:
            parent_combinations = [{}]  # Empty context if no parents
        else:
            parent_values = [list(self.cardinalities[p].values()) for p in parents]
            parent_combinations = [
                dict(zip(parents, combo)) for combo in product(*parent_values)
            ]

        for parent_config in parent_combinations:
            # N_i: count of instances where parents take this configuration
            # print(f"        parent_config: {parent_config}")
            N_i = self.compute_counts(parent_config)
            # print(f"        N_i: {N_i}")

            # Skip configurations that don't occur in the data
            if N_i == 0:
                continue

            # First term: log(Γ(α*r)) - log(Γ(N_i + α*r))
            term1 = gammaln(self.alpha * r) - gammaln(N_i + self.alpha * r)

            # Second term: Σ_j [log(Γ(N_ij + α)) - log(Γ(α))]
            term2 = 0
            for child_val in self.cardinalities[child].values():
                context_with_child = dict(parent_config)
                context_with_child[child] = child_val
                # print(f"context_with_child: {context_with_child}")
                N_ij = self.compute_counts(context_with_child)
                term2 += gammaln(N_ij + self.alpha) - gammaln(self.alpha)

            total_score += term1 + term2

        return total_score

    def learn_k2_structure(
        self,
        u=3,
        variable_order=None,
    ):
        """
        Simplest possible K2 algorithm implementation.

        Args:
            data: list of dictionaries, each representing one data point
            variable_order: list of variable names in topological order
            max_parents: maximum number of parents per variable

        Returns:
            Dictionary {child: [list of parents]}
        """
        if variable_order is None:
            # Use arbitrary order based on first data point
            variable_order = self.variables

        print(f"Variable order: {variable_order}")

        result = {}

        for i, child in enumerate(variable_order):
            print(f"\nLearning parents for {child}...")

            # Possible parents are variables that come before in the ordering
            possible_parents = variable_order[:i]

            if not possible_parents:
                result[child] = []
                print(f"  No possible parents (first variable)")
                continue

            # Start with no parents
            current_parents = []
            current_score = self.k2_score(child, current_parents)
            print(f"  Score with no parents: {current_score:.4f}")

            # Greedily add parents while score improves
            for num_parents in range(u):
                if len(current_parents) >= len(possible_parents):
                    break

                best_candidate = None
                best_score = current_score

                # Try adding each possible parent
                for candidate in possible_parents:
                    if candidate in current_parents:
                        continue

                    candidate_parents = current_parents + [candidate]
                    try:
                        score = self.k2_score(child, candidate_parents)
                        print(f"  Score with parents {candidate_parents}: {score:.4f}")

                        if score > best_score:
                            print(f"    Δscore = {score - current_score:.4f}")
                            best_score = score
                            best_candidate = candidate
                    except Exception as e:
                        print(f"  Error with parents {candidate_parents}: {e}")
                        continue

                # Add best candidate if it improves score
                if best_candidate is not None:
                    current_parents.append(best_candidate)
                    current_score = best_score
                    print(
                        f"  Added parent {best_candidate}, new score: {current_score:.4f}"
                    )
                else:
                    print(f"  No improvement found, stopping")
                    break

            result[child] = current_parents
            print(f"  Final parents for {child}: {current_parents}")

        print(f"Best hypothesis for u={u}: {result}")
        return result

    def k2_parents_to_hypothesis(self, parents_dict):
        """
        Convert {child: [parents]} → {parent: [children]} format
        to match self.set_hypothesis expectations.
        """
        hypothesis = defaultdict(list)
        for child, parent_list in parents_dict.items():
            for parent in parent_list:
                hypothesis[parent].append(child)
        return dict(hypothesis)

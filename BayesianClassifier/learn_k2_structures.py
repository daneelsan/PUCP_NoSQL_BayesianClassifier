import json
import time
from bayes_classifier import BayesianClassifier, available_hypotheses


def learn_and_save_structures(alpha=1.0):
    """
    Learn Bayesian network structures using K2 algorithm with different u values
    and save them to available_hypotheses and a JSON file.
    """
    # Initialize classifier
    print(f"Initializing Bayesian Classifier (alpha={alpha})...")
    classifier = BayesianClassifier(alpha=alpha)

    # Print basic info about the dataset
    print(f"Alpa: {classifier.alpha}")
    print(f"Dataset size: {classifier.N:,} documents")
    print(f"Variables: {classifier.variables}")
    print(
        f"Variable cardinalities: {[(var, len(cards)) for var, cards in classifier.cardinalities.items()]}"
    )
    print()

    # Learn structures for different values of u
    learned_structures = {}

    for u in range(1, 6):
        print(f"{'='*60}")
        print(f"Learning K2 structure for u={u} (max {u} parents per variable)")
        print(f"{'='*60}")

        start_time = time.time()

        #variable_order=['fraud', 'amount_bin', 'category', 'gender', 'age']
        variable_order=['age', 'gender', 'amount_bin', 'category', 'fraud']
        try:
            # Learn the structure
            structure = classifier.learn_k2_structure(u=u, variable_order=variable_order)

            # Convert to hypothesis format
            hypothesis = classifier.k2_parents_to_hypothesis(structure)
            print(f"Converted hypothesis: {hypothesis}")

            # Store in available_hypotheses
            name = f"K2 learned (u={u})"
            available_hypotheses[name] = hypothesis
            learned_structures[name] = {
                "structure": structure,
                "hypothesis": hypothesis,
                "u": u,
                "learning_time": time.time() - start_time,
            }

            print(f"Learning completed in {time.time() - start_time:.2f} seconds")
            print(f"Structure learned: {name}")
            print("Parent-child relationships:")

            # Show learned structure in both formats
            print("  Child → Parents:")
            for child, parents in structure.items():
                print(f"    {child} ← {parents if parents else '(no parents)'}")

            print("  Parent → Children:")
            for parent, children in hypothesis.items():
                print(f"    {parent} → {children}")

            # Calculate and display some statistics
            total_edges = sum(len(parents) for parents in structure.values())
            max_parents = (
                max(len(parents) for parents in structure.values()) if structure else 0
            )
            variables_with_parents = sum(1 for parents in structure.values() if parents)

            print(f"  Statistics:")
            print(f"    Total edges: {total_edges}")
            print(f"    Max parents for any variable: {max_parents}")
            print(
                f"    Variables with parents: {variables_with_parents}/{len(structure)}"
            )
            print()

        except Exception as e:
            print(f"Error learning structure for u={u}: {e}")
            print(f"Skipping u={u}")
            continue

    # Save to JSON file
    print("Saving learned hypotheses to file...")
    try:
        # Create a clean version for JSON serialization
        json_data = {
            "metadata": {
                "dataset_size": classifier.N,
                "variables": classifier.variables,
                "cardinalities": {
                    var: len(cards) for var, cards in classifier.cardinalities.items()
                },
                "learning_timestamp": time.time(),
            },
            "hypotheses": available_hypotheses,
            "learning_details": {
                name: {
                    "structure": details["structure"],
                    "u": details["u"],
                    "learning_time": details["learning_time"],
                }
                for name, details in learned_structures.items()
            },
        }

        with open("learned_hypotheses.json", "w") as f:
            json.dump(json_data, f, indent=2)

        print(
            f"Saved {len(learned_structures)} learned structures to 'learned_hypotheses.json'"
        )

    except Exception as e:
        print(f"Error saving to JSON: {e}")

        # Fallback: save just the hypotheses
        try:
            with open("learned_hypotheses_fallback.json", "w") as f:
                json.dump(available_hypotheses, f, indent=2)
            print("Saved fallback version to 'learned_hypotheses_fallback.json'")
        except Exception as e2:
            print(f"Failed to save fallback: {e2}")

    print(f"\nFinal available_hypotheses keys: {list(available_hypotheses.keys())}")
    return learned_structures


# def validate_learned_structures():
#     """
#     Validate that learned structures are reasonable and don't contain cycles.
#     """
#     print("\nValidating learned structures...")

#     classifier = BayesianClassifier(alpha=0.1)

#     for name, hypothesis in available_hypotheses.items():
#         if not name.startswith("K2 learned"):
#             continue

#         print(f"Validating {name}:")

#         # Check for obvious issues
#         all_vars = set()
#         for parent, children in hypothesis.items():
#             all_vars.add(parent)
#             all_vars.update(children)

#         # Basic validation
#         issues = []

#         # Check if all variables are known
#         unknown_vars = (
#             all_vars - set(classifier.variables) if "classifier" in locals() else set()
#         )
#         if unknown_vars:
#             issues.append(f"Unknown variables: {unknown_vars}")

#         # Check for self-loops
#         for parent, children in hypothesis.items():
#             if parent in children:
#                 issues.append(f"Self-loop detected: {parent} → {parent}")

#         if issues:
#             print(f"  Issues found: {issues}")
#         else:
#             print(f"  Structure appears valid")


if __name__ == "__main__":
    print("K2 Structure Learning Script")
    print("=" * 40)

    # Learn structures
    learned_structures = learn_and_save_structures(alpha=0.5)

    # Validate structures
    # validate_learned_structures()

    print("\nStructure learning completed!")
    print(
        f"Total structures learned: {len([k for k in available_hypotheses.keys() if k.startswith('K2 learned')])}"
    )

import random
import pandas as pd
import sys
import json  # Import the json module to handle JSON files
import time  # Ensure time is imported for timestamps if needed in metadata

# Asumiendo que 'bayes_classifier.py' contiene la definición de BayesianClassifier
# y las dependencias de MongoDB, dotenv, etc.
# Si la clase BayesianClassifier no se encuentra en bayes_classifier.py,
# deberás pegarla aquí o asegurarte de que el módulo sea accesible.
from bayes_classifier import BayesianClassifier
from bayes_classifier import available_hypotheses

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)


# --- Función Principal de Medición de Métricas ---
def run_classification_metrics_comparison(hyphothesis_name):
    # Número de muestras de prueba para clasificar y evaluar
    num_test_samples = 5000  # Un número mayor para una evaluación más robusta

    print(
        f"Iniciando la evaluación del Clasificador {hyphothesis_name} en el dataset completo..."
    )

    classifier = BayesianClassifier(hyphothesis_name=hyphothesis_name)

    print(
        f"\n--- Clasificador para la hipotesis {hyphothesis_name} en dataset completo ({classifier.N:,} documentos) ---"
    )

    print(f"  Obteniendo {num_test_samples} muestras de prueba para evaluación...")
    test_samples = []
    try:
        # Usar $sample para obtener una muestra aleatoria de documentos del dataset completo
        # Asegurarse de que el tamaño de la muestra no exceda el tamaño de la colección
        sample_size_for_evaluation = min(num_test_samples, classifier.N)
        if sample_size_for_evaluation == 0:
            print("  No hay documentos disponibles para muestrear. Finalizando.")
            return

        test_data_cursor = classifier.data_collection.aggregate(
            [{"$sample": {"size": sample_size_for_evaluation}}], allowDiskUse=True
        )
        test_samples = list(test_data_cursor)
    except Exception as e:
        print(f"  Error al obtener muestras de prueba: {e}. Finalizando.")
        return

    if not test_samples:
        print(
            f"  No hay muestras de prueba disponibles después de muestrear. Finalizando."
        )
        return

    # Listas para almacenar las etiquetas verdaderas y las predicciones
    y_true = []
    y_pred = []
    total_classify_time = 0
    classified_count = 0

    print(f"  Clasificando {len(test_samples)} muestras...")

    # Obtener el valor indexado para 'yes' (fraude) de las cardinalidades
    indexed_fraud_yes_val = classifier.cardinalities["fraud"]["yes"]

    for sample in test_samples:
        # Preparar la evidencia para la clasificación (todas las variables excepto 'fraud')
        evidence = {
            k: v
            for k, v in sample.items()
            if k != "fraud" and k in classifier.variables
        }

        # Obtener la etiqueta verdadera (convertir de índice 0/1 a booleano False/True)
        true_label_indexed = sample["fraud"]  # Será 0 o 1
        true_label_boolean = true_label_indexed == indexed_fraud_yes_val

        try:
            # classify() retorna (es_predicción_fraude_booleano, probabilidad, tiempo_tomado)
            is_fraud_pred, prob, classify_time = classifier.classify(
                evidence, apply_index=False
            )

            y_true.append(true_label_boolean)
            y_pred.append(is_fraud_pred)
            total_classify_time += classify_time
            classified_count += 1
        except Exception as e:
            print(f"    Error al clasificar muestra: {e} - Muestra: {evidence}")
            continue

    metrics_results = {}  # Diccionario para almacenar los resultados para JSON

    if classified_count > 0:
        avg_classify_time = total_classify_time / classified_count

        # Calcular métricas de clasificación
        precision = precision_score(y_true, y_pred, pos_label=True, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=True, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=True, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        # CALCULAR MATRIZ DE CONFUSIÓN
        # labels=[False, True] asegura que el orden sea [[TN, FP], [FN, TP]]
        cm = confusion_matrix(y_true, y_pred, labels=[False, True])
        tn, fp, fn, tp = cm.ravel()  # Desempaqueta la matriz de confusión

        print(f"\n--- Resultados de la Clasificación ({hyphothesis_name}) ---")
        print(f"  Muestras clasificadas: {classified_count}")
        print(f"  Tiempo promedio por clasificación: {avg_classify_time:.6f} segundos")
        print(f"  ------------------------------------------------")
        print(f"  **Accuracy (Exactitud):** {accuracy:.4f}")
        print(f"  **Precision (Precisión):** {precision:.4f}")
        print(f"  **Recall (Exhaustividad):** {recall:.4f}")
        print(f"  **F1-Score:** {f1:.4f}")
        print(f"  ------------------------------------------------")
        print(f"  **Verdaderos Positivos (TP):** {tp}")
        print(f"  **Verdaderos Negativos (TN):** {tn}")
        print(f"  **Falsos Positivos (FP):** {fp}")
        print(f"  **Falsos Negativos (FN):** {fn}")
        print(f"  ------------------------------------------------")

        # Almacenar los resultados en el diccionario
        metrics_results = {
            "model": hyphothesis_name,
            "dataset_info": {
                "total_documents_in_db": classifier.N,
                "evaluated_samples_count": classified_count,
            },
            "metrics": {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
            },
            "performance": {"avg_classification_time_s": round(avg_classify_time, 6)},
            "timestamp": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.gmtime()
            ),  # Add a timestamp
        }

        return metrics_results
    else:
        print(
            f"  No se clasificaron muestras. No se calcularon métricas ni se guardaron resultados."
        )


# --- Ejecutar la Función Principal ---
if __name__ == "__main__":
    metrics_results = []
    for name in available_hypotheses.keys():
        metrics_results.append(run_classification_metrics_comparison(name))
    # Guardar los resultados en un archivo JSON
    output_filename = "classification_metrics.json"
    try:
        with open(output_filename, "w") as f:
            json.dump(metrics_results, f, indent=4)
        print(f"\nResultados guardados en '{output_filename}'")
    except Exception as e:
        print(f"Error al guardar los resultados en JSON: {e}")

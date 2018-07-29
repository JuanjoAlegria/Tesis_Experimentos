"""Script para entrenar un HubModuleModel, utilizando para ello la clase
auxiliar HubModelExperiment.
"""
import os
import json
import numpy as np
from ...experiments import hub_module_experiment


def train_model_with_dataset(flags, dataset_json):
    print("Obteniendo dataset desde archivo json")
    train_filenames = np.array(dataset_json["train_features"])
    train_labels = np.array(dataset_json["train_labels"])
    validation_filenames = np.array(dataset_json["validation_features"])
    validation_labels = np.array(dataset_json["validation_labels"])
    test_filenames = np.array(dataset_json["test_features"])
    test_labels = np.array(dataset_json["test_labels"])
    n_classes = len(set(train_labels))

    print("Creando HubModelExperiment")
    experiment = hub_module_experiment.HubModelExperiment(flags, n_classes)

    print("Entrenando")
    experiment.train_and_evaluate(train_filenames, train_labels,
                                  validation_filenames, validation_labels)
    print("Evaluando con el conjunto de validación")
    experiment.test_and_predict(
        validation_filenames, validation_labels, "validation")
    print("Evaluando con el conjunto de prueba")
    experiment.test_and_predict(test_filenames, test_labels, "test")
    print("Exportando")
    experiment.export_graph()


def main(flags):
    """Función de entrada. Se obtienen los datasets desde dataset_json,
    y se crea un HubModelExperiment, con el cual se entrena y evalúa el modelo,
    para finalmente ser exportado.
    """

    if os.path.isfile(flags.dataset_path):
        np.random.seed(flags.random_seed)
        with open(flags.dataset_path) as file:
            dataset_json = json.load(file)
        train_model_with_dataset(flags, dataset_json)

    else:  # si es que flags.dataset_path es un directorio
        base_experiment_name = flags.experiment_name
        for dataset_file in os.listdir(flags.dataset_path):
            name, ext = os.path.splitext(dataset_file)
            if ext != ".json":
                continue

            np.random.seed(flags.random_seed)
            # Obtenemos el índice del dataset
            idx_dataset = name.split("_")[-1]
            experiment_name = base_experiment_name + "_" + idx_dataset
            flags.experiment_name = experiment_name
            full_dataset_path = os.path.join(flags.dataset_path, dataset_file)
            print("Dataset actual", dataset_file)
            with open(full_dataset_path) as file:
                dataset_json = json.load(file)
            train_model_with_dataset(flags, dataset_json)

if __name__ == "__main__":
    PARSER = hub_module_experiment.get_parser()
    FLAGS = PARSER.parse_args()
    main(FLAGS)

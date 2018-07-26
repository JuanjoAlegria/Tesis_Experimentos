"""Script para entrenar un HubModuleModel, utilizando para ello la clase
auxiliar HubModelExperiment.
"""
import json
import numpy as np
from ...experiments import hub_module_experiment


def main(flags):
    """Función de entrada. Se obtienen los datasets desde dataset_json,
    y se crea un HubModelExperiment, con el cual se entrena y evalúa el modelo,
    para finalmente ser exportado.
    """
    np.random.seed(flags.random_seed)
    with open(flags.dataset_json) as file:
        dataset_json = json.load(file)

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

if __name__ == "__main__":
    PARSER = hub_module_experiment.get_parser()
    FLAGS = PARSER.parse_args()
    main(FLAGS)

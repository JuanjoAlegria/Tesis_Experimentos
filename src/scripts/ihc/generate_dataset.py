"""Script para generar un dataset en formato JSON"
"""

import os
import json
import argparse
import numpy as np
from ...dataset import data_utils


def main(images_dir, dataset_path, test_percentage,
         validation_percentage, random_seed,
         minimum_tissue_proportion, proportions_json):
    """Genera un archivo JSON con los nombres de los archivos que conforman
    un dataset.

    Args:
        - images_dir: str. Directorio donde se encuentran las imágenes.
        - dataset_path: str. Ubicación donde guardar el dataset en formato
        json.
        - validation_percentage: int. Porcentaje de imágenes del conjunto de
        entrenamiento que serán utilizadas como conjunto de validación.
        - test_percentage: int. Porcentaje de imágenes del conjunto de
        entrenamiento que serán utilizadas como conjunto de prueba.
        - random_seed: int. Semilla aleatoria (útil para obteer resultados
        reproducibles)
        - minimum_tissue_proportion: float. Cantidad mínima de tejido requerida
        en la imagen para que ésta sea mantenida en el dataset.
        - proportions_json: str. Ubicación del archivo json con las
        proporciones de tejido de cada imagen. Sólo necesario si
        minimum_tissue_proportion > 0.
    """
    # Obtenemos todos los archivos
    np.random.seed(random_seed)
    dataset_dir, _ = os.path.split(dataset_path)
    os.makedirs(dataset_dir, exist_ok=True)
    percentages = [100 - test_percentage - validation_percentage,
                   validation_percentage, test_percentage]
    filenames, labels, \
        labels_map = data_utils.get_filenames_and_labels(images_dir)

    if minimum_tissue_proportion > 0:
        with open(proportions_json) as file:
            proportions_dict = json.load(file)
        filenames_tmp = []
        labels_tmp = []
        for filename, label in zip(filenames, labels):
            if proportions_dict[filename] > minimum_tissue_proportion:
                filenames_tmp.append(filename)
                labels_tmp.append(label)
        filenames = np.array(filenames_tmp)
        labels = np.array(labels_tmp)

    datasets = data_utils.generate_partition(filenames, labels, percentages)
    train_filenames, train_labels = datasets[0]
    validation_filenames, validation_labels = datasets[1]
    test_filenames, test_labels = datasets[2]

    data_utils.dump_dataset(dataset_path,
                            train_filenames.tolist(),
                            train_labels.tolist(),
                            validation_filenames.tolist(),
                            validation_labels.tolist(),
                            test_filenames.tolist(),
                            test_labels.tolist())

    data_utils.check_integrity_dumped_dataset(dataset_path, labels_map)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--images_dir',
        type=str,
        required=True,
        help='Directorio donde se encuentran las imágenes.'
    )
    PARSER.add_argument(
        '--dataset_path',
        type=str,
        default="",
        help="Ubicación donde guardar el dataset en formato json.",
        required=True
    )
    PARSER.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help="""
        Porcentaje de imágenes del conjunto de entrenamiento que serán
        utilizadas como conjunto de validación.
        """
    )
    PARSER.add_argument(
        '--test_percentage',
        type=int,
        default=10,
        help="""
        Porcentaje de imágenes del conjunto de entrenamiento que serán
        utilizadas como conjunto de prueba.
        """
    )
    PARSER.add_argument(
        '--random_seed',
        type=int,
        default=None,
        help='Semilla aleatoria (útil para obteer resultados reproducibles).'
    )
    PARSER.add_argument(
        '--minimum_tissue_proportion',
        type=float,
        help="""\
        Proporción mínima de tejido que debe tener una imagen para ser
        considerada en el dataset final. Opcional, pero si se entrega un valor
        debe ser entre 0 y 1 (ambos inclusive), y además se debe entregar una
        ruta para proportions_json.\
        """,
        default=-1
    )
    PARSER.add_argument(
        '--proportions_json',
        type=str,
        help=""""\
        Ruta al archivo json, que contiene un diccionario que tiene como
        llaves los nombres de las imágenes, y como valores asociados la
        proporción de tejido correspondiente a cada imagen. Sólo es necesario
        si es que minimum_tissue_proportion > 0.\
        """
    )
    FLAGS = PARSER.parse_args()
    main(FLAGS.images_dir, FLAGS.dataset_path,
         FLAGS.test_percentage, FLAGS.validation_percentage,
         FLAGS.random_seed, FLAGS.minimum_tissue_proportion,
         FLAGS.proportions_json)

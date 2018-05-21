"""Script para crear un dataset en formato json basado en las imágenes
de mnist
"""

import os
import argparse
import numpy as np
from ..dataset import data_utils


def check_filesystem(flags):
    """Crea variables si es que su valor es un string vacío, y crea los
    directorios correspondientes.

    Args:
        flags: argparse.Namespace
    """
    root_path = os.getcwd()
    _, images_dir_name = os.path.split(flags.images_dir)
    dict_partitions_path = os.path.join(root_path, "data",
                                        "partitions_json",
                                        images_dir_name)
    if flags.output_labels == "":
        flags.output_labels = os.path.join(dict_partitions_path,
                                           "output_labels.txt")
    if flags.dataset_path == "":
        flags.dataset_path = os.path.join(dict_partitions_path,
                                          "dataset_dict.json")
    os.makedirs(dict_partitions_path, exist_ok=True)


def main(flags):
    """Crea un archivo json con las particiones de mnist.

    Args:
        - flags: argparse.Namespace
    """
    np.random.seed(flags.random_seed)
    check_filesystem(flags)
    train_filenames, train_labels, labels_map = data_utils.\
        get_filenames_and_labels(
            flags.images_dir, "train", labels_to_indexes=False,
            labels_map=None, max_files=flags.max_files_train)
    test_filenames, test_labels, _ = data_utils.\
        get_filenames_and_labels(
            flags.images_dir, "test", labels_to_indexes=False,
            labels_map=labels_map, max_files=flags.max_files_test)

    (train_filenames, train_labels), \
        (validation_filenames, validation_labels) = data_utils.\
        generate_partition(train_filenames, train_labels,
                           flags.validation_proportion)

    data_utils.dump_dataset(flags.dataset_path,
                            train_filenames.tolist(),
                            train_labels.tolist(),
                            validation_filenames.tolist(),
                            validation_labels.tolist(),
                            test_filenames.tolist(),
                            test_labels.tolist())
    data_utils.write_labels_map(labels_map, flags.output_labels)
    data_utils.check_integrity_dumped_dataset(flags.dataset_path, labels_map)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--images_dir',
        type=str,
        required=True,
        help='Directorio donde se encuentran las imágenes.'
    )
    PARSER.add_argument(
        '--random_seed',
        type=int,
        default=None,
        help='Semilla aleatoria (útil para obteer resultados reproducibles).'
    )
    PARSER.add_argument(
        '--output_labels',
        type=str,
        default="",
        help='Ubicación donde guardar el mapeo de etiquetas.'
    )
    PARSER.add_argument(
        '--dataset_path',
        type=str,
        default="",
        help="""\
        Ubicación donde guardar el dataset en formato json.'\
        """
    )
    PARSER.add_argument(
        '--validation_proportion',
        type=int,
        default=10,
        help="""\
        Porcentaje de imágenes del conjunto de entrenamiento que serán
        utilizadas como conjunto de validación.
        """
    )
    PARSER.add_argument(
        '--max_files_train',
        type=int,
        default=-1,
        help="""\
        Cantidad máxima de archivos a usar para entrenar. Útil para depuración.
        """,
    )
    PARSER.add_argument(
        '--max_files_test',
        type=int,
        default=-1,
        help="""\
        Cantidad máxima de archivos a usar para evaluar. Útil para depuración.
        """,
    )
    FLAGS, _ = PARSER.parse_known_args()
    main(FLAGS)

import os
import argparse
import numpy as np
from ...dataset import data_utils


def main(images_dir, dataset_path,
         test_percentage, validation_percentage, random_seed):
    # Obtenemos todos los archivos
    np.random.seed(random_seed)
    dataset_dir, _ = os.path.split(dataset_path)
    os.makedirs(dataset_dir, exist_ok=True)
    percentages = [100 - test_percentage - validation_percentage,
                   validation_percentage, test_percentage]
    filenames, labels, labels_map = data_utils.get_filenames_and_labels(
        images_dir)
    datasets = data_utils.generate_partition(filenames, labels, percentages)
    import pdb
    pdb.set_trace()  # breakpoint 20ca7c1f //

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
        help="""\
        Porcentaje de imágenes del conjunto de entrenamiento que serán
        utilizadas como conjunto de validación.
        """
    )
    PARSER.add_argument(
        '--test_percentage',
        type=int,
        default=10,
        help="""\
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
    FLAGS = PARSER.parse_args()
    main(FLAGS.images_dir, FLAGS.dataset_path,
         FLAGS.test_percentage, FLAGS.validation_percentage,
         FLAGS.random_seed)

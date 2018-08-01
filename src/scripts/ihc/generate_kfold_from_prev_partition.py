"""Script para generar varias particiones del dataset, utilizando una
partición de ids de biopsias ya generada y almacenada en formato json.
"""

import os
import json
import argparse
from ...dataset import data_utils


def main(ids_partition_json, train_dir, test_dir, dataset_dst_dir,
         train_proportions_json, test_proportions_json, proportion_threshold):
    """Crea n_folds particiones del dataset, a partir de una partición de ids
    de slides ya generada.
    """
    os.makedirs(dataset_dst_dir, exist_ok=True)

    with open(ids_partition_json) as file:
        partitions = json.load(file)

    negative_part = partitions["negative_part"]
    equivocal_part = partitions["equivocal_part"]
    positive_part = partitions["positive_part"]

    labels_map = {'0': 0, '1': 0, '2': 1, '3': 2}

    with open(train_proportions_json) as file:
        train_proportions = json.load(file)
    with open(test_proportions_json) as file:
        test_proportions = json.load(file)

    data_utils.generate_kfold_with_previous_partition(
        negative_part, equivocal_part, positive_part, train_dir, test_dir,
        dataset_dst_dir, train_proportions, test_proportions,
        proportion_threshold, labels_map)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--ids_partition_json',
        type=str,
        required=True,
        help="""\
        Ruta al archivo json con la partición de ids generada previamente.
        """
    )
    PARSER.add_argument(
        '--train_dir',
        type=str,
        help="""\
        Directorio donde se encuentran los parches que deben ser utilizados
        para entrenamiento y validación. En caso de no entregar un valor, se
        asume que los parches están en la carpeta
        data/processed/ihc_patches_from_roi_x40 .\
        """,
        default=os.path.join(os.getcwd(), "data",
                             "processed", "ihc_patches_from_roi_x40")
    )
    PARSER.add_argument(
        '--test_dir',
        type=str,
        help="""\
        Directorio donde se encuentran los parches que deben ser utilizados
        para prueba. En caso de no entregar un valor, se asume que los parches
        están en la carpeta data/processed/ihc_all_patches_x40 .\
        """,
        default=os.path.join(os.getcwd(), "data",
                             "processed", "ihc_all_patches_x40")
    )
    PARSER.add_argument(
        '--dataset_dst_dir',
        type=str,
        help="""\
        Directorio donde se guardarán los distintos archivos json con los
        datasets creados. Si no se entrega ningún valor, los archivos serán
        guardados en data/partitions_json/ihc_patches_x40/k_fold .\
        """,
        default=os.path.join(os.getcwd(), "data", "partitions_json",
                             "ihc_patches_x40", "k_fold")
    )
    PARSER.add_argument(
        '--train_proportions_json',
        type=str,
        help=""""\
        Ruta al archivo json, que contiene un diccionario que tiene como
        llaves los nombres de las imágenes de entrenamiento, y como valores
        asociados la proporción de tejido correspondiente a cada imagen. Si no
        se entrega un valor, se asume que la ruta es
        data/extras/ihc_slides/tissue_proportion_ihc_patches_from_rois_x40.json
        . """,
        default=os.path.join(os.getcwd(), "data", "extras", "ihc_slides",
                             "tissue_proportion_ihc_patches_from_rois_x40")
    )
    PARSER.add_argument(
        '--test_proportions_json',
        type=str,
        help=""""\
        Ruta al archivo json, que contiene un diccionario que tiene como
        llaves los nombres de las imágenes de prueba, y como valores
        asociados la proporción de tejido correspondiente a cada imagen. Si no
        se entrega un valor, se asume que la ruta es
        data/extras/ihc_slides/tissue_proportion_ihc_all_patches_x40.json
        . """,
        default=os.path.join(os.getcwd(), "data", "extras", "ihc_slides",
                             "tissue_proportion_ihc_all_patches_x40")
    )
    PARSER.add_argument(
        '--proportion_threshold',
        type=float,
        help="""\
        Proporción mínima de tejido que debe tener una imagen para ser 
        considerada en el dataset final. \
        """,
        default=0.1
    )

    FLAGS = PARSER.parse_args()
    main(FLAGS.ids_partition_json, FLAGS.train_dir, FLAGS.test_dir,
         FLAGS.dataset_dst_dir, FLAGS.train_proportions_json,
         FLAGS.test_proportions_json, FLAGS.proportion_threshold)

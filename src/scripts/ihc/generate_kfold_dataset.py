"""Script para generar varias particiones del dataset, y así validar usando
k-fold.
"""

import os
import json
import argparse
import numpy as np
from ...utils import excel
from ...dataset import data_utils


def main(excel_file, n_folds, train_dir, test_dir, dataset_dst_dir,
         train_proportions_json, test_proportions_json, proportion_threshold):
    """Crea una partición del dataset.
    """
    ids, labels = excel.get_valid_slides_ids(excel_file)
    ids, labels = np.array(ids), np.array(labels)
    negative = ids[np.where((labels == '0') | (labels == '1'))]
    equivocal = ids[np.where(labels == '2')]
    positive = ids[np.where(labels == '3')]

    with open(train_proportions_json) as file:
        train_proportions = json.load(file)
    with open(test_proportions_json) as file:
        test_proportions = json.load(file)

    data_utils.generate_kfold(n_folds, negative, equivocal, positive,
                              train_dir, test_dir, dataset_dst_dir,
                              train_proportions, test_proportions,
                              proportion_threshold)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--excel_file',
        type=str,
        help="""\
        Ruta al archivo excel con la información de las slides y anotaciones.\
        """,
        default=os.path.join(os.getcwd(), "data", "extras",
                             "ihc_slides", "HER2.xlsx")
    )
    PARSER.add_argument(
        '--n_folds',
        type=int,
        help="""\
        Número de particiones que se desea generar. Si es que no se entrega un
        valor, se crearán k particiones, donde
        k = min(len(negative_slides), len(equivocal_slides),
        len(positive_slides))
        """,
        default=-1
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
    main(FLAGS.excel_file, FLAGS.n_folds, FLAGS.train_dir, FLAGS.test_dir,
         FLAGS.dataset_dst_dir, FLAGS.train_proportions_json,
         FLAGS.test_proportions_json, FLAGS.proportion_threshold)

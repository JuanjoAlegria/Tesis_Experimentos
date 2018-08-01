"""Script para generar particiones de las ids de las slides. Esto permitirá
reutilizar dichas particiones entre distintos experimentos con diferentes
magnificaciones.
"""

import os
import json
import argparse
import numpy as np
from ...utils import excel
from ...dataset import data_utils


def main(excel_file, n_folds, ids_partition_dst):
    """Crea n_folds particiones de las ids de las slides del dataset.
    """
    ids, labels = excel.get_valid_slides_ids(excel_file)
    ids, labels = np.array(ids), np.array(labels)
    negative = ids[np.where((labels == '0') | (labels == '1'))]
    equivocal = ids[np.where(labels == '2')]
    positive = ids[np.where(labels == '3')]

    ids_partition = data_utils.generate_partitions_slide_ids(
        n_folds, negative, equivocal, positive)
    with open(ids_partition_dst, "w") as file:
        json.dump(ids_partition, file)

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
        '--ids_partition_dst',
        type=str,
        help="""\
        Ubicación donde se escribirá un archivo json con las particiones de 
        ids generadas. Si no se entrega ningún valor, los archivos serán
        guardados en data/partitions_json/slides_ids_kfold.json .\
        """,
        default=os.path.join(os.getcwd(), "data", "partitions_json",
                             "slides_ids_kfold.json")
    )

    FLAGS = PARSER.parse_args()
    main(FLAGS.excel_file, FLAGS.n_folds, FLAGS.ids_partition_dst)

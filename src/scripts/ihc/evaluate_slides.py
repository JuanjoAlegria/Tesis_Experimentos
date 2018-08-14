"""Extrae los parches desde los ROIs y genera un dataset.
Este script está hecho para funcionar con el archivo excel con el que he
estado trabajando estas últimas semanas, no es generalizable a otros archivos.
"""

import json
import argparse
import numpy as np
from ...utils import excel, outputs
from ...experiments import evaluation


def main(excel_file, predictions_path, tissue_proportions_path):
    """Clasifica biopsias intentando seguir las guías clínicas.

    Args:
        - excel_file: str. Ruta al archivo excel con la información de las
        slides y anotaciones.
        - predictions_path: str. Ruta al archivo .txt con las predicciones
        realizadas por la red.
        - tissue_proportions_path: str. Ruta al archivo .json con las
        proporciones de tejido en cada imagen.
    """
    slides_ids, labels = excel.get_valid_slides_ids(excel_file)
    slides_ids, labels = np.array(slides_ids), np.array(labels)
    with open(predictions_path) as file:
        predictions_list = file.readlines()
    with open(tissue_proportions_path) as file:
        tissue_proportions_dict = json.load(file)
    predictions_dict = outputs.transform_to_dict(
        predictions_list)
    test_ids = outputs.get_all_slides_ids(predictions_dict)
    for test_id in test_ids:
        her2_score = evaluation.classify_slide_her2(
            test_id, predictions_dict, tissue_proportions_dict)
        print(test_id, her2_score)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--excel_file',
        type=str,
        help="""\
        Ruta al archivo excel con la información de las slides y anotaciones.
        """,
        required=True
    )
    PARSER.add_argument(
        '--predictions_path',
        type=str,
        help="""\
        Ruta al archivo .txt con las predicciones realizadas por la red.
        """,
        required=True
    )
    PARSER.add_argument(
        '--tissue_proportions_path',
        type=str,
        help="""\
        Ruta al archivo .json con las proporciones de tejido en cada imagen.
        """,
        required=True
    )
    FLAGS = PARSER.parse_args()
    main(FLAGS.excel_file, FLAGS.predictions_path,
         FLAGS.tissue_proportions_path)

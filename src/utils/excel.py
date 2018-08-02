"""Módulo con funciones utilitarias para trabajar con planillas de Excel.
"""

import re
import pandas as pd

ANNOTATIONS_INFO_SHEET = 1
CROSSED_INFO_SHEET = 2

FINAL_EVALUATION_COLUMN = "Evaluación final (consenso)"
SLIDES_NAMES_COLUMN = "Nombre slide"
ANNOTATION_ID_COLUMN = "Id anotación"
ANNOTATION_TYPE_COLUMN = "Tipo anotación"
ANNOTATION_OWNER_COLUMN = "Autor anotación"
ANNOTATION_LABEL_COLUMN = "Detalles anotación"
SLIDES_NAME_PATTERN = r"229-UCH-(\d*)-IDA"


def get_valid_slides_ids(excel_file):
    """Retorna una lista con los slides de las ids válidas; es decir, aquellas
    donde existe un consenso entre patólogos. Además, retorna una lista con
    la clasificación HER2 de cada slide.

    Args:
        - excel_file: str. Ruta del archivo excel con las clasificaciones.

    Returns:
        2-tuple(list[str]), donde en la primera lista cada item es una
        id de slide (11, 4, 116, etc), mientras que en la segunda lista cada
        item corresponde a la clasificación HER2 de la slide correlativa.
    """
    crossed_info = pd.read_excel(excel_file, sheetname=CROSSED_INFO_SHEET)
    not_null = ~crossed_info[FINAL_EVALUATION_COLUMN].isnull()
    valid_rows = crossed_info[not_null]

    valid_slides_names = valid_rows[SLIDES_NAMES_COLUMN].tolist()
    valid_slides_ids = map(lambda s: re.match(SLIDES_NAME_PATTERN, s).group(1),
                           valid_slides_names)

    slides_evaluation = valid_rows[FINAL_EVALUATION_COLUMN].tolist()
    slides_evaluation = map(lambda x: str(int(x)), slides_evaluation)

    return list(valid_slides_ids), list(slides_evaluation)


def get_valid_annotations_labels(excel_file, valid_owners):
    """Obtiene un diccionario con las ids de las anotaciones válidas (es
    decir, aquellas que hayan sido creadas por un autor en valid_owners) y la
    clase a la cual pertenecen.

    Args:
        - excel_file: str. Ruta del archivo excel con las clasificaciones.
        - valid_owners: list[str]. Lista con los autores de anotaciones
        válidas.

    Returns:
        dict[str: str], diccionario que tiene como llaves las ids de las
        anotaciones válidas, y como valores asociados las clases
        correpondientes.
    """
    annotations_info = pd.read_excel(
        excel_file, sheetname=ANNOTATIONS_INFO_SHEET)
    annotations_info.set_index(ANNOTATION_ID_COLUMN, inplace=True)
    valid_annotations = {}
    for index, row in annotations_info.iterrows():
        if row[ANNOTATION_OWNER_COLUMN] in valid_owners and \
                row[ANNOTATION_TYPE_COLUMN] in ["freehand", "circle"]:
            valid_annotations[str(index)] = str(row[ANNOTATION_LABEL_COLUMN])
    return valid_annotations

"""Módulo con funciones utilitarias para trabajar con planillas de Excel.
"""

import re
import pandas as pd

ANNOTATIONS_INFO_SHEET = "Información anotaciones"
SUMMARY_INFO_SHEET = "Resumen"

FINAL_EVALUATION_COLUMN = "Evaluación final (consenso)"
PATHOLOGIST_0_COLUMN = "Patólogo 0"
PATHOLOGIST_1_COLUMN = "Patólogo 1"
PATHOLOGIST_2_COLUMN = "Patólogo 2"
SLIDES_NAMES_COLUMN = "Nombre slide (ndp.microscopiavirtual.cl)"
BIOPSY_TYPE_COLUMN = "Tipo de biopsia"
ANNOTATION_ID_COLUMN = "Id anotación"
ANNOTATION_TYPE_COLUMN = "Tipo anotación"
ANNOTATION_OWNER_COLUMN = "Autor anotación"
ANNOTATION_LABEL_COLUMN = "Detalles anotación"
MIN_X_COORD_COLUMN = "Coordenada X inicial (pixeles)"
SLIDES_NAME_PATTERN = r"229-UCH-(\d*)-IDA"


RESECTION_CODE = "Resección"
ENDOSCOPY_CODE = "Endoscopia"
ALL_BIOPSIES_CODE = "Todas"


def str_of_int_or_none(num):
    """Retorna str(int(x)) si es que x puede ser representado como un entero,
    None en caso contrario.

    Args:
        - num: any. Elemento que se quiere convertir a entero

    Returns:
        int(num) si num puede ser representado como entero, None en caso
        contrario.
    """
    try:
        return str(int(num))
    except (ValueError, TypeError):
        return None


def get_slides_ids_full_eval(excel_file, biopsy_tipe=ALL_BIOPSIES_CODE):
    """Retorna una lista con las ids de todas las slides, y otra lista con
    todas sus evaluaciones asociadas(patólogo 0, patólogo 1, patólogo 2 y
    evaluación final de consenso). También puede filtrar por tipo de biopsia:
    resección, endoscopia o todas.

    Args:
        - excel_file: str. Ruta del archivo excel con las clasificaciones.
        - biopsy_tipe: str. Tipo de biopsia a filtrar. Opciones son
        "Resección", "Endoscopia" o "Todas"

    Returns:
        (list[str], list[(str, str, str, str)]), donde en la primera lista
        cada item es una id de slide(11, 4, 116, etc), mientras que en la
        segunda lista cada item corresponde a cuatro clasificaciones HER2; en
        orden, la clasificación del patólogo 0, patólogo 1, patólogo 2,
        evaluación final(ej: (3, 2, 3, 3)).
    """
    crossed_info = pd.read_excel(excel_file, sheetname=SUMMARY_INFO_SHEET"Resumen"
    if biopsy_tipe not in [ALL_BIOPSIES_CODE, ENDOSCOPY_CODE, RESECTION_CODE]:
        raise ValueError("Tipo de biopsia debe ser: " +
                         "excel.ALL_BIOPSIES_CODE, excel.ENDOSCOPY_CODE o " +
                         "excel.RESECTION_CODE")
    # Filtramos por tipo de biopsia
    if biopsy_tipe != ALL_BIOPSIES_CODE:
        crossed_info=crossed_info.loc[
            crossed_info[BIOPSY_TYPE_COLUMN] == biopsy_tipe]
    slides_names=crossed_info[SLIDES_NAMES_COLUMN].tolist()
    slides_ids=map(lambda s: re.match(SLIDES_NAME_PATTERN, s).group(1),
                     slides_names)

    eval_0=crossed_info[PATHOLOGIST_0_COLUMN].tolist()
    eval_0=map(str_of_int_or_none, eval_0)

    eval_1=crossed_info[PATHOLOGIST_1_COLUMN].tolist()
    eval_1=map(str_of_int_or_none, eval_1)

    eval_2=crossed_info[PATHOLOGIST_2_COLUMN].tolist()
    eval_2=map(str_of_int_or_none, eval_2)

    eval_final=crossed_info[FINAL_EVALUATION_COLUMN].tolist()
    eval_final=map(str_of_int_or_none, eval_final)

    all_evaluations=zip(eval_0, eval_1, eval_2, eval_final)
    return list(slides_ids), list(all_evaluations)


def get_valid_slides_ids(excel_file):
    """Retorna una lista con los ids de las slides válidas; es decir, aquellas
    donde existe un consenso entre patólogos. Además, retorna una lista con
    la clasificación HER2 respectiva de cada slide.

    Args:
        - excel_file: str. Ruta del archivo excel con las clasificaciones.

    Returns:
        (list[str], list[str]), donde en la primera lista cada item es una
        id de slide(11, 4, 116, etc), mientras que en la segunda lista cada
        item corresponde a la clasificación HER2 de la slide correlativa.
    """
    crossed_info=pd.read_excel(excel_file, sheetname=SUMMARY_INFO_SHEET"Resumen"
    not_null=~crossed_info[FINAL_EVALUATION_COLUMN].isnull()
    valid_rows=crossed_info[not_null]

    valid_slides_names=valid_rows[SLIDES_NAMES_COLUMN].tolist()
    valid_slides_ids=map(lambda s: re.match(SLIDES_NAME_PATTERN, s).group(1),
                           valid_slides_names)

    slides_eval=valid_rows[FINAL_EVALUATION_COLUMN].tolist()
    slides_eval=map(str_of_int_or_none, slides_eval)

    return list(valid_slides_ids), list(slides_eval)


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
    annotations_info=pd.read_excel(
        excel_file, sheetname=ANNOTATIONS_INFO_SHEET)
    annotations_info.set_index(ANNOTATION_ID_COLUMN, inplace=True)
    valid_annotations={}
    for index, row in annotations_info.iterrows():
        if row[ANNOTATION_OWNER_COLUMN] in valid_owners and \
                row[ANNOTATION_TYPE_COLUMN] in ["freehand", "circle"]:
            valid_annotations[str(index)]=str(row[ANNOTATION_LABEL_COLUMN])
    return valid_annotations


def get_min_x_coords(excel_file):
    """Para cada slide válida, bbtiene la coordenada mínima en el eje x que
    debe tener un parche extraído de dicha slide para ser considerado como
    parte de la biopsia y no como parte del tejido control.

    Args:
        - excel_file: str. Ruta del archivo excel con las clasificaciones.

    Returns:
        dict[str->int]. Diccionario que a cada id de slide le asocia la
        coordenada mínima en el eje x para que un parche sea considerado parte
        de la biopsia y no parte del tejido control.
    """
    crossed_info=pd.read_excel(excel_file, sheetname=SUMMARY_INFO_SHEET"Resumen"
    not_null=~crossed_info[FINAL_EVALUATION_COLUMN].isnull()
    valid_rows=crossed_info[not_null]

    valid_slides_names=valid_rows[SLIDES_NAMES_COLUMN].tolist()
    valid_slides_ids=map(lambda s: re.match(SLIDES_NAME_PATTERN, s).group(1),
                           valid_slides_names)

    min_x_coords=valid_rows[MIN_X_COORD_COLUMN].tolist()
    min_x_coords=map(int, min_x_coords)

    return dict(zip(valid_slides_ids, min_x_coords))

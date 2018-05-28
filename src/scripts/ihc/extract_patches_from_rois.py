"""Extrae los parches desde los ROIs y genera un dataset.
Este script está hecho para funcionar con el archivo excel con el que he
estado trabajando estas últimas semanas, no es generalizable a otros archivos.
"""

import os
import re
import argparse
import pandas as pd
from ...utils import image

ANNOTATIONS_INFO_SHEET = 1
CROSSED_INFO_SHEET = 2

FINAL_EVALUATION_COLUMN = "Evaluación final (consenso)"
SLIDES_NAMES_COLUMN = "Nombre slide"
ANNOTATION_ID_COLUMN = "Id anotación"
ANNOTATION_TYPE_COLUMN = "Tipo anotación"
ANNOTATION_OWNER_COLUMN = "Autor anotación"
ANNOTATION_LABEL_COLUMN = "Detalles anotación"

SLIDES_NAME_PATTERN = r"229-UCH-(\d*)-IDA"
ANNOTATIONS_NAME_PATTERN = r"(\d*)_x(\d*)_z0_(\d*)"


def get_valid_slides_ids(excel_file):
    """Retorna una lista con los slides de las ids válidas; es decir, aquellas
    donde existe un consenso entre patólogos.

    Args:
        - excel_file: str. Ruta del archivo excel con las clasificaciones.

    Returns:
        list[str], donde cada item es una id de slide (11, 4, 116, etc).
    """
    crossed_info = pd.read_excel(excel_file, sheetname=CROSSED_INFO_SHEET)
    not_null = ~crossed_info[FINAL_EVALUATION_COLUMN].isnull()
    valid_rows = crossed_info[not_null]
    valid_slides_names = valid_rows[SLIDES_NAMES_COLUMN].tolist()
    valid_slides_ids = map(lambda s: re.match(SLIDES_NAME_PATTERN, s).group(1),
                           valid_slides_names)
    return list(valid_slides_ids)


def get_info_from_roi_name(roi_name):
    """Obtiene el slide_id y el annotation_id desde el nombre de un ROI,
    utlizando expresiones regulares

    Args:
        - roi_name: str. Nombre del archivo que contiene al ROI
    """
    re_object = re.match(ANNOTATIONS_NAME_PATTERN, roi_name)
    slide_id = re_object.group(1)
    # El grupo 2 corresponde a la magnificación
    annotation_id = re_object.group(3)
    return slide_id, annotation_id


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


def main(excel_file, rois_dir, patches_dir, valid_owners,
         patches_height, patches_width, stride_rows, stride_columns,
         threshold_gray_pixels):
    """Extrae parches desde ROIs almacenados en rois_dir, utilizando la
    información contenida en excel_file. Luego, guarda los parches en
    patches_dir, en un subdirectorio correspondiente a su clase (0,1,2,3).

    Args:
        - excel_file: str. Ruta al archivo excel con la información de las
        slides y anotaciones.
        - rois_dir: str. Directorio donde se encuentran los ROIs extraídos
        desde las slides ndpi, usando las anotaciones provistas por patólogos.
        - patches_dir: str. Directorio donde se guardarán los parches extraídos
        desde los ROIs.
        - valid_owners: list[str]. Nombres de los autores de anotaciones
        válidas, desde las cuales serán extraídas los parches. Opciones son
        UI.Patologo1 y UI.Patologo2.
        - patches_height: int. Altura (pixeles) de los parches
        - patches_width: int. Ancho (pixeles) de los parches.
        - stride_rows: int. Intervalo, en pixeles, según el cual extraer los
        parches en el eje de las filas.
        - stride_columns: int. Intervalo, en pixeles, según el cual extraer los
        parches en el eje de las columnas.

    """
    valid_slides_ids = get_valid_slides_ids(excel_file)
    valid_annotations = get_valid_annotations_labels(excel_file, valid_owners)

    labels = set(valid_annotations.values())

    for label in labels:
        os.makedirs(os.path.join(patches_dir, str(label)), exist_ok=True)

    n_rois = 0
    n_patches = 0

    for roi_name_and_ext in os.listdir(rois_dir):
        roi_name, roi_ext = os.path.splitext(roi_name_and_ext)

        if roi_ext != ".jpg":
            continue
        slide_id, annotation_id = get_info_from_roi_name(roi_name)
        if slide_id not in valid_slides_ids:
            continue
        if annotation_id not in valid_annotations:
            continue
        label = valid_annotations[annotation_id]
        current_dst_dir = os.path.join(patches_dir, label)
        roi_path = os.path.join(rois_dir, roi_name_and_ext)
        patches = image.extract_patches(roi_path, patches_height,
                                        patches_width, stride_rows,
                                        stride_columns)
        patches = {name: patch for name, patch in patches.items()
                   if image.is_useful_patch(patch, threshold_gray_pixels)}
        image.save_patches(patches, roi_name, current_dst_dir)
        print(roi_name, len(patches), label)
        n_rois += 1
        n_patches += len(patches)
    print("N rois", n_rois)
    print("N patches", n_patches)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--excel_file',
        type=str,
        help="""\
        Ruta al archivo excel con la información de las slides y anotaciones.
        """,
        default=os.path.join(os.getcwd(), "data", "extras",
                             "ihc_slides", "HER2.xlsx")
    )
    PARSER.add_argument(
        '--rois_dir',
        type=str,
        help="""\
        Directorio donde se encuentran los ROIs extraídos desde las slides
        ndpi, usando las anotaciones provistas por patólogos. En caso de no
        entregar un valor, se asume que las slides están en la carpeta
        data/interim/ihc_rois_x40.\
        """,
        default=os.path.join(os.getcwd(), "data", "interim", "ihc_rois_x40")
    )
    PARSER.add_argument(
        '--patches_dir',
        type=str,
        help="""\
        Directorio donde se guardarán los parches extraídos desde los ROIs. 
        En caso de no entregar un valor, los parches serán guardado en la
        carpeta data/processed/ihc_patches_x40.\
        """,
        default=os.path.join(os.getcwd(), "data",
                             "processed", "ihc_patches_x40")
    )
    PARSER.add_argument(
        '--valid_owners',
        type=str,
        nargs="*",
        help="""\
        Nombres de los autores de anotaciones válidas, desde las cuales serán
        extraídas los parches. Opciones son UI.Patologo1, UI.Patologo2.\
        """,
        default=["UI.Patologo2"]
    )
    PARSER.add_argument(
        '--patches_height',
        type=int,
        help="Altura (pixeles) de los parches",
        default=300
    )
    PARSER.add_argument(
        '--patches_width',
        type=int,
        help="Ancho (pixeles) de los parches",
        default=300
    )
    PARSER.add_argument(
        '--stride_rows',
        type=int,
        help="""\
        Intervalo, en pixeles, según el cual extraer los parches en el eje de 
        las filas\
        """,
        default=50
    )
    PARSER.add_argument(
        '--stride_columns',
        type=int,
        help="""\
        Intervalo, en pixeles, según el cual extraer los parches en el eje de 
        las filas\
        """,
        default=50
    )
    PARSER.add_argument(
        '--threshold_gray_pixels',
        type=float,
        help="""\
        Proporción máxima de pixeles grises soportada. Debe ser un valor entre
        0 y 1.\
        """,
        default=0.8
    )

    FLAGS = PARSER.parse_args()
    main(FLAGS.excel_file, FLAGS.rois_dir,
         FLAGS.patches_dir, FLAGS.valid_owners,
         FLAGS.patches_height, FLAGS.patches_width,
         FLAGS.stride_rows, FLAGS.stride_columns,
         FLAGS.threshold_gray_pixels)

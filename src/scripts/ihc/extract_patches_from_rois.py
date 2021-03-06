"""Extrae los parches desde los ROIs y genera un dataset.
Este script está hecho para funcionar con el archivo excel con el que he
estado trabajando estas últimas semanas, no es generalizable a otros archivos.
"""

import os
import re
import argparse
from ...utils import image, excel

ANNOTATIONS_NAME_PATTERN = r"(\d*)_x(\d*)_z0_(\d*)"


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


def main(excel_file, rois_dir, patches_dir, valid_owners,
         patches_height, patches_width, stride_rows, stride_columns):
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
    valid_slides_ids, _ = excel.get_valid_slides_ids(excel_file)
    valid_annotations = excel.get_valid_annotations_labels(excel_file,
                                                           valid_owners)

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
        carpeta data/processed/ihc_patches_from_rois_x40.\
        """,
        default=os.path.join(os.getcwd(), "data",
                             "processed", "ihc_patches_from_rois_x40")
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

    FLAGS = PARSER.parse_args()
    main(FLAGS.excel_file, FLAGS.rois_dir,
         FLAGS.patches_dir, FLAGS.valid_owners,
         FLAGS.patches_height, FLAGS.patches_width,
         FLAGS.stride_rows, FLAGS.stride_columns)

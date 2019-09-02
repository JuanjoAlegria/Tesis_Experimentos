"""Script para extraer los ROIs asociados a anotaciones serializadas desde la
slide ndpi correspondiente.
"""

import os
import re
import glob
import pickle
import argparse
from ...utils import image


def move_and_convert_rois(src_dir, dst_dir):
    """Convierte todos los rois en formato .tif a formato .jpeg y los mueve
    a otro directorio.

    Args:
        - src_dir: str. Directorio donde residen los ROIs.
        - dst_dir: str. Directorio a donde se quiere mover los ROIs.
    """
    os.makedirs(dst_dir, exist_ok=True)
    all_rois = glob.glob(os.path.join(src_dir, "*.tif"))
    print("Transformando a jpeg")
    for roi_path in all_rois:
        image.tif_to_jpeg(roi_path, dst_dir)
        print("Convertida imagen:", roi_path)
        os.remove(roi_path)


def get_slide_path(annotation, slides_dir):
    """Obtiene la ubicación de la slide a la cual corresponde una anotación.

    Se asume que slides_dir es la carpeta donde residen todas las slides, las
    cuales tiene nombres de tipo xyz.ndpi, donde xyz es un número.
    Originalmente, las slides tienen nombres de tipo 229-UCH-xyz-IDA (%).ndpi,
    pero por simplicidad, se les cambió el nombre a sólo xyz.ndpi.

    ACTUALIZACIÓN: en nueva versión de experimento, es posible que
    annotation.slide_name ya esté en formato xyz.ndpi, por lo cual en ese caso,
    sólo se retorna la concatenación de slides_dir y el nombre de la slide.

    Args:
        - annotation: ndp.annotation.Annotation. Anotación que contiene la
        region que se quiere extraer.
        - slides_dir: str. Directorio con las slides ndpi.

    Returns:
        str, con la ruta de la slide correspondiente.

    """
    if re.match(r'(\d*).ndpi', annotation.slide_name):
        return os.path.join(slides_dir, annotation.slide_name)
    pattern = r'229-UCH-(\d*)-IDA'
    slide_id = re.match(pattern, annotation.slide_name).group(1)
    return os.path.join(slides_dir, slide_id + ".ndpi")


def main(serialized_annotations_dir, slides_dir, rois_dir, magnification):
    """Extrae los ROIs asociados a anotaciones serializadas desde la slide
    ndpi correspondiente.

    Args:
        - serialized_annotations_dir: str. Directorio con las anotaciones
        serializadas.
        - slides_dir: str. Directorio con las slides ndpi.
        - valid_owners: list[str]. Lista con los autores de anotaciones que
        serán considerados al extraer regiones.
        - magnification: str. Magnificación a la cual se quieren extraer
        las regiones (x5, x10, x20, x40).
    """
    for s_annotation in os.listdir(serialized_annotations_dir):
        if not ".pkl" in s_annotation:
            continue
        annotation_path = os.path.join(serialized_annotations_dir,
                                       s_annotation)
        with open(annotation_path, "rb") as file:
            annotation = pickle.load(file)
        if annotation.annotation_type not in ["circle", "freehand"]:
            continue

        slide_path = get_slide_path(annotation, slides_dir)
        if os.path.exists(slide_path):
            annotation.extract_region_from_ndpi(slide_path, magnification)
    move_and_convert_rois(slides_dir, rois_dir)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--serialized_annotations_dir',
        type=str,
        help="""\
        Directorio donde se encuentran las anotaciones serializadas en formato
        pickle. En caso de no entregar un valor, se asume que las anotaciones
        están en la carpeta data/extras/ihc_slides/serialized_annotations.\
        """,
        default=os.path.join(os.getcwd(), "data", "extras",
                             "ihc_slides", "serialized_annotations")
    )
    PARSER.add_argument(
        '--slides_dir',
        type=str,
        help="""\
        Directorio donde se encuentran las slides ndpi. En caso de no entregar
        un valor, se asume que las slides están en la carpeta
        data/raw/ihc_slides.\
        """,
        default=os.path.join(os.getcwd(), "data", "raw", "ihc_slides")
    )
    PARSER.add_argument(
        '--rois_dir',
        type=str,
        help="""\
        Directorio donde se guardarán los ROIs. En caso de no entregar
        un valor, los ROIs serán guardado en la carpeta
        data/interim/ihc_rois.\
        """,
        default=os.path.join(os.getcwd(), "data", "interim", "ihc_rois")
    )
    PARSER.add_argument(
        '--magnification',
        type=str,
        help="""\
        Magnificación a la cual se quieren extraer las regiones (x5, x10,
        x20, x40).\
        """,
        default="x40"
    )
    FLAGS = PARSER.parse_args()
    main(FLAGS.serialized_annotations_dir, FLAGS.slides_dir,
         FLAGS.rois_dir, FLAGS.magnification)

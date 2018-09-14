"""Script para calcular la proporción de tejido en los parches extraídos de
las slides.
"""
import os
import json
import glob
import argparse
import concurrent.futures
import cv2
from ...utils import image


def get_tissue_proportion_from_path(image_path):
    """Función auxiliar para calcular la proporción de tejido en una imagen,
    conociendo sólo su ubicación en disco.

    Args:
        - image_path: str. Ubicación de la imagen.

    Returns:
        float en el rango [0,1], representando la proporción de tejido en la
        imagen.
    """
    patch = cv2.imread(image_path)
    return image.calculate_tissue_proportion(patch)


def main(patches_dir, json_path):
    """Calcula las proporciones de tejido en un conjunto de imágenes contenidas
    en un directorio, y guarda el resultado en un archivo json.

    Args:
        - patches_dir: str. Directorio con parches.
        - json_path: str. Ubicación del archivo json donde se guardarán las
        proporciones de tejido de cada imagen.
    """
    proportions = {}
    for subdir in os.listdir(patches_dir):
        subdir_full = os.path.join(patches_dir, subdir)
        if not os.path.isdir(subdir_full):
            continue
        print(subdir_full)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            filenames = glob.glob(os.path.join(subdir_full, "*.jpg"))
            for whole_image_name, tissue_proportion in zip(
                    filenames, executor.map(get_tissue_proportion_from_path,
                                            filenames)):
                head, image_name = os.path.split(whole_image_name)
                _, label = os.path.split(head)
                label_and_name = os.path.join(label, image_name)
                proportions[label_and_name] = tissue_proportion

    with open(json_path, "w") as json_file:
        json.dump(proportions, json_file)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--patches_dir',
        type=str,
        help="""\
        Directorio donde están guardados los parches extraídos desde 
        las slides. En caso de no entregar un valor, se asumirá que los 
        parches fueron guardados en la carpeta 
        data/processed/ihc_all_patches_x40 .\
        """,
        default=os.path.join(os.getcwd(), "data",
                             "processed", "ihc_all_patches_x40")
    )
    PARSER.add_argument(
        '--json_path',
        type=str,
        help="""\
        Ubicación del archivo json donde se guardará la información de las
        proporciones de tejido en cada imagen. Si no se entrega un valor, se
        asumirá que la ubicación es 
        data/extras/ihc_slides/tissue_proportion_ihc_all_patches_x40.json ."
        \
        """,
        default=os.path.join(os.getcwd(), "data", "extras", "ihc_slides",
                             "tissue_proportion_ihc_all_patches_x40.json")
    )

    FLAGS = PARSER.parse_args()
    main(FLAGS.patches_dir, FLAGS.json_path)

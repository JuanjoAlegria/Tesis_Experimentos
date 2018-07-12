"""Módulo con funciones utilitarias para trabajar con imágenes.
"""

import os
import numpy as np
import cv2


def calculate_tissue_proportion(image):
    """Calcula la proporción de tejido que hay en una imagen con respecto al
    total de pixeles en la imagen.

    Para realizar esto, primero se calcula la cantidad de pixeles grises (
    aquellos que no contienen tejido) usando simplemente un inRange en la
    imagen RGB. Luego, se calcula la diferencia (es decir, los pixeles que
    sí tienen tejido) y con ello se calcula su proporción respecto al total de
    pixeles.

    Args:
        - image: np.array(), imagen con tres canales

    Returns:
        float en el rango [0,1] que representa la proporción de tejido en la 
        imagen con respecto al total de pixeles en ésta.
    """
    lower = np.array([225, 225, 225])
    upper = np.array([255, 255, 255])

    non_tissue_pixels = cv2.inRange(image, lower, upper)
    n_non_tissue_pixels = np.count_nonzero(non_tissue_pixels)
    n_pixels = image.shape[0] * image.shape[1]
    n_tissue_pixels = n_pixels - n_non_tissue_pixels

    return n_tissue_pixels / n_pixels


def is_useful_patch(image, threshold_proportion=0.7):
    """Retorna True si es que la cantidad de pixeles grises en la imagen
    es menor a cierta proporción de pixeles en la imagen.

    Args:
        - image: np.array, imagen.
        - threshold_proportion: float. Proporción máxima de pixeles grises
        soportada.

    Returns:
        bool. True, si es que la cantidad de pixeles grises en image es menor
        a threshold_proportion multiplicado por la cantidad total de pixeles,
        False en caso contrario.
    """

    unuseful_proportion = 1 - calculate_tissue_proportion(image)
    return unuseful_proportion < threshold_proportion


def extract_patches(image_path, patch_height, patch_width,
                    stride_rows, stride_columns):
    """Extrae parches desde una imagen, con determinado tamaño y cada cierta
    cantidad de pixeles.

    Args:
        - image_path: str. Ruta a la imagen de la cual se quieren extraer los
        parches.
        - patch_height: int. Altura en pixeles requerida de los parches.
        - patch_width: int. Ancho en pixeles  requerido de los parches.
        - stride_rows: int. Intervalo, en pixeles, según el cual avanzar en la
        dirección de las filas.
        - stride_columns: int. Intervalo, en pixeles, según el cual avanzar en
        la dirección de las columnas.

    Returns:
        - dict[str: np.array], donde cada llave es un string que indica la fila
        y la columna de donde fue extraido el parche (en formato
        "i(\\d*)_j(\\d*)"), y cada valor asociado es l parche correspondiente.

    """
    img = cv2.imread(image_path)
    roi_height, roi_width, _ = img.shape
    name_template = "i{row}_j{column}"
    patches_dict = {}
    for row in range(0, roi_height, stride_rows):
        for column in range(0, roi_width, stride_columns):
            patch = img[row: row + patch_height,
                        column: column + patch_width]
            if patch.shape[:2] == (patch_height, patch_width):
                patch_name = name_template.format(row=row, column=column)
                patches_dict[patch_name] = patch

    return patches_dict


def save_patches(patches_dict, prefix_name, dst_dir):
    """Guarda un diccionario con imágenes a dst_dir.

    Todas las imágenes serán guardadas en dst_dir, con el nombre
    {prefix_name}_{image_key}.jpg, donde image_key es la llave asociada en el
    diccionario a cada imagen.

    Args:
        - patches_dict: dict[str: np.array], diccionario con imágenes como
        valores y con los sufijos para el nombre de cada imagen como llaves.
        - prefix_name: str. Nombre común que llevarán todas las imágenes.
        - dst_dir: str. Directorio donde se guardarán las imágenes.
    """

    template = prefix_name + "_{suffix}.jpg"
    for suffix_key in patches_dict:
        image_name = template.format(suffix=suffix_key)
        image_path = os.path.join(dst_dir, image_name)
        image = patches_dict[suffix_key]
        cv2.imwrite(image_path, image)


def tif_to_jpeg(tif_image_path, dst_dir, jpeg_quality=100):
    """Convierte una imagen en formato tif a formato jpeg.

    Args:
        - tif_image_path: str. Ubicación de la imagen en formato tif.
        - dst_dir: Directorio donde se quiere guardar la imagen en formato
        jpeg.
        - jpeg_quality. int. Calidad esperada de la imagen jpeg resultante.
    """
    _, image_name_and_ext = os.path.split(tif_image_path)
    image_name, _ = os.path.splitext(image_name_and_ext)

    image = cv2.imread(tif_image_path)
    new_file = os.path.join(dst_dir, image_name + ".jpg")
    cv2.imwrite(new_file, image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

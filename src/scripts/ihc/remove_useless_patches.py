"""Limpia los parches inútiles, basándose en el tamaño de la imagen.
"""
import os
import argparse


def maybe_delete_file(file_path, min_size):
    """Decide si es que debe o no eliminar un archivo en base a su tamaño en
    bytes, y en caso de ser así, procede a eliminarla.

    Args:
        - file_path: str. Ruta al archivo.
        - min_size: int. Tamaño en bytes.

    Returns:
        bool: True en caso de que se haya eliminado el archivo, False en caso
        contrario.
    """
    bytes_size = os.path.getsize(file_path)
    if bytes_size < min_size:
        os.remove(file_path)
        return True
    return False


def main(patches_base_dir, kib_min_size):
    """Recorre patches_base_dir, y por cada imagen que encuentre, decide
    si es que debe eliminarla o no.

    Args:
        - patches_base_dir: str. Directorio con los parches. Su estructura debe
        ser como sigue:
            patches_base_dir/
                label_1/
                    images...
                    images...
                label_2/
                    more_images
                    etc
                etc.
        - kib_min_size: float. Tamaño mínino en KiB. Su valor será multiplicado
        por 2**10 (1024) para transformarlo a una cantidad en bytes.
    """
    bytes_min_size = kib_min_size * 1024
    n_deleted = 0
    for label in os.listdir(patches_base_dir):
        label_dir = os.path.join(patches_base_dir, str(label))
        if not os.path.isdir(label_dir):
            continue
        for image_name in os.listdir(label_dir):
            if ".jpg" not in image_name:
                continue
            image_path = os.path.join(label_dir, image_name)
            deleted = maybe_delete_file(image_path, bytes_min_size)
            if deleted:
                n_deleted += 1
    print("Eliminadas", n_deleted, "imágenes")

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--patches_dir',
        type=str,
        help="""\
        Directorio donde están almacenados los parches extraídos desde los
        ROIs. En caso de no entregar un valor, los parches serán buscados en la
        carpeta data/processed/ihc_patches_x40.\
        """,
        default=os.path.join(os.getcwd(), "data",
                             "processed", "ihc_patches_x40")
    )
    PARSER.add_argument(
        '--kib_min_size',
        type=float,
        help="""\
        Tamaño mínino en KiB. Si una imagen tiene un tamaño menor a este, será
        eliminada.\
        """,
        default=9.5
    )

    FLAGS = PARSER.parse_args()
    main(FLAGS.patches_dir, FLAGS.kib_min_size)

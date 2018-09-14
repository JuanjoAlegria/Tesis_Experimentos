"""Script para extraer imágenes JPEG desde las slides NDPI. Si bien es posible
extraer imágenes con magnificaciones mayores (x40, x20, x10), se recomienda
usar magnificaciones menores, para que así sea posible abrir la imagen y que
ésta quepa en RAM."""

import os
import glob
import argparse
import subprocess

COMMAND_TEMPLATE = "ndpi_converter {options} {slide_path} {output_dir}"


def find_image_number(slide_path, magnification):
    """Encuentra el número de la imagen con la magnificación deseada para una
    slide ndpi.

    Generalmente, el número es el mismo (x40 => 1, x20 => 2, x10 => 3), pero
    existen algunas slides donde no existe la imagen con magnificación x20.

    Args:
        - slide_path: str. Ruta a la slide ndpi.
        - magnification: str. Magnificación (x40, x20, x10, x5, x2.5)

    Returns:
        int, que representa el número de la imagen con la magnificación
        deseada, o -1 si es que no se encuentra la magnificación.
    """
    command_list = COMMAND_TEMPLATE.format(
        options="-f", slide_path=slide_path,
        output_dir="").split()
    execution = subprocess.run(command_list, stdout=subprocess.PIPE)
    result = execution.stdout.decode('utf-8').split("\n")
    for line in result:
        if magnification in line:
            return int(line.split()[0])
    return -1


def export_image(slide_path, magnification, output_dir):
    """Exporta una slide ndpi a jpeg con la magnificación deseada.

    Args:
        - slide_path: str. Ruta a la slide ndpi.
        - magnification: str. Magnificación (x40, x20, x10, x5, x2.5)
        - output_dir: str. Directorio donde se guardará la imagen extraída.
    """
    image_number = find_image_number(slide_path, magnification)
    if image_number < 0:
        print("No se pudo extraer la imagen", slide_path)
        return
    command_list = COMMAND_TEMPLATE.format(
        options="-i {number} -e".format(number=image_number),
        slide_path=slide_path, output_dir=output_dir).split(" ")
    subprocess.run(command_list)


def main(src_dir, magnification, output_dir):
    """Busca todas las slides ndpi en src_dir y las exporta a output_dir con
    la magnificación deseada.

    Args:
        - src_dir: str. Directorio con slides ndpi.
        - magnification: str. Magnificación (x40, x20, x10, x5, x2.5)
        - output_dir: str. Directorio donde se guardarán las imagenes
        extraídas.

    """
    ndpi_slides = glob.glob(os.path.join(src_dir, "*.ndpi"))
    for slide in ndpi_slides:
        print(slide)
        export_image(slide, magnification, output_dir)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--src_dir',
        type=str,
        help="Directorio con slides ndpi.",
        required=True
    )
    PARSER.add_argument(
        '--magnification',
        type=str,
        help="Magnificación (x40, x20, x10, x5, x2.5)",
        required=True
    )
    PARSER.add_argument(
        '--output_dir',
        type=str,
        help="""\
        Directorio donde se guardarán las imagenes extraídas. Si no se ingresa
        un valor, las imágenes serán guardadas en src_dir/exported.
        """,
        default=""
    )
    FLAGS = PARSER.parse_args()
    main(FLAGS.src_dir, FLAGS.magnification, FLAGS.output_dir)

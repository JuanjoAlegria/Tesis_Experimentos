"""Extrae todos los parches de una slide, usando la información provista
en el archivo excel HER2.xlsx.
"""

import os
import glob
import argparse
import openslide
from ...utils import image, excel
from ...ndp import ndpisplit_wrapper


def move_patches(slide_id, src_dir, dst_dir):
    """Mueve a otra ubicación todos los parches extraídos desde una slide.

    Args:
        - slide_id: str. Id de la slide cuyos parches se desea mover.
        - src_dir: str. Directorio donde están ubicados los parches.
        - dst_dir: str. Directorio hacia donde se moverán los parches.
    """
    patches = glob.glob(os.path.join(src_dir, "{}_*.tif".format(slide_id)))
    print("Transformando a jpeg parches de slide", slide_id)
    for patch in patches:
        image.tif_to_jpeg(patch, dst_dir)
        os.remove(patch)


def main(excel_file, slides_dir, patches_dir,
         patches_height, patches_width, magnification):
    """Extrae parches desde ROIs almacenados en rois_dir, utilizando la
    información contenida en excel_file. Luego, guarda los parches en
    patches_dir, en un subdirectorio correspondiente a su clase (0,1,2,3).

    Args:
        - excel_file: str. Ruta al archivo excel con la información de las
        slides y anotaciones.
        - slides_dir: str. Directorio donde se encuentran las slides NDPI.
        - patches_dir: str. Directorio donde se guardarán los parches extraídos
        desde los ROIs.
        - patches_height: int. Altura (pixeles) de los parches
        - patches_width: int. Ancho (pixeles) de los parches.
        - magnification: str. Magnificación a la cual se quieren extraer
        las regiones (x5, x10, x20, x40).
    """
    slides_ids, slides_labels = excel.get_valid_slides_ids(excel_file)
    for slide_id, slide_label in zip(slides_ids, slides_labels):
        ndpi_path = os.path.join(slides_dir, slide_id + ".ndpi")
        print(ndpi_path)
        slide = openslide.OpenSlide(ndpi_path)
        width_l0 = int(slide.properties['openslide.level[0].width'])
        import pdb
        pdb.set_trace()  # breakpoint f073c3ab //

        height_l0 = int(slide.properties['openslide.level[0].height'])
        current_dst_dir = os.path.join(patches_dir, slide_label)
        os.makedirs(current_dst_dir, exist_ok=True)
        for row in range(0, height_l0 - (height_l0 % patches_height),
                         patches_height):
            for column in range(0, width_l0 - (width_l0 % patches_width),
                                patches_width):
                row_column = "x{column}_y{row}".format(row=row, column=column)
                ndpisplit_wrapper.extract_region(ndpi_path, column, row,
                                                 patches_width, patches_height,
                                                 magnification,
                                                 label=row_column)
        move_patches(slide_id, slides_dir, current_dst_dir)


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
        '--patches_dir',
        type=str,
        help="""\
        Directorio donde se guardarán los parches extraídos desde los ROIs. 
        En caso de no entregar un valor, los parches serán guardado en la
        carpeta data/processed/ihc_all_patches_x40 .\
        """,
        default=os.path.join(os.getcwd(), "data",
                             "processed", "ihc_all_patches_x40")
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
        '--magnification',
        type=str,
        help="""\
        Magnificación a la cual se quieren extraer las regiones (x5, x10,
        x20, x40).\
        """,
        default="x40"
    )

    FLAGS = PARSER.parse_args()
    main(FLAGS.excel_file, FLAGS.slides_dir, FLAGS.patches_dir,
         FLAGS.patches_height, FLAGS.patches_width,
         FLAGS.magnification)

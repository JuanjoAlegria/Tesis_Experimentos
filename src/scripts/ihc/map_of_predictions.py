"""Extrae los parches desde los ROIs y genera un dataset.
Este script está hecho para funcionar con el archivo excel con el que he
estado trabajando estas últimas semanas, no es generalizable a otros archivos.
"""

import os
import argparse
from PIL import Image
from ...experiments import evaluation
from ...utils import outputs


def main(predictions_path, dst_dir, patches_height, patches_width):
    """Clasifica biopsias intentando seguir las guías clínicas.

    Args:
        - predictions_path: str. Ruta al archivo .txt con las predicciones
        realizadas por la red.
        - dst_dir: str. Directorio donde se guardarán las imágenes generadas.
    """
    with open(predictions_path) as file:
        predictions_list = file.readlines()
    if dst_dir is None:
        dst_dir = os.path.split(predictions_path)[0]
    predictions_dict = outputs.transform_to_dict(predictions_list)
    all_slides_ids = outputs.get_all_slides_ids(predictions_dict)
    for slide_id in all_slides_ids:
        map_image = evaluation.generate_map_of_predictions(
            slide_id, predictions_dict,
            patches_height=patches_height,
            patches_width=patches_width)
        pil_image = Image.fromarray(map_image)
        save_path = os.path.join(dst_dir, "{}.png".format(slide_id))
        pil_image.save(save_path)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--predictions_path',
        type=str,
        help="""\
        Ruta al archivo .txt con las predicciones realizadas por la red.
        """,
        required=True
    )
    PARSER.add_argument(
        '--dst_dir',
        type=str,
        help="""\
        Directorio donde se guardarán las imágenes. Si no se entrega un valor,
        la carpeta será deducida a partir de predictions_path.""",
        default=None
    )
    PARSER.add_argument(
        '--patches_height',
        type=int,
        help="Altura de los parches utilizados",
        default=300
    )
    PARSER.add_argument(
        '--patches_width',
        type=int,
        help="Ancho de los parches utilizados",
        default=300
    )

    FLAGS = PARSER.parse_args()
    main(FLAGS.predictions_path, FLAGS.dst_dir,
         FLAGS.patches_height, FLAGS.patches_width)

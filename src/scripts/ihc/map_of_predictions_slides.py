"""Script para generar mapas de predicciones para slides. La principal
diferencia con maps_of_predictions_slides_full, es que el presente script no
requiere de una imagen con la biopsia en alguna magnificación y que el tamaño
de la ventana que representa cada parche puede ser de tamaño arbitrario, por
lo cual puede generar imágenes más pequeñas y fáciles de manejar.
"""

import os
import argparse
from PIL import Image
from ...experiments import evaluation
from ...utils import outputs


def main(predictions_path, dst_dir, patches_height,
         patches_width, map_window_height, map_window_width):
    """Genera mapas de predicciones para slides.

    Args:
        - predictions_path: str. Ruta al archivo .txt con las predicciones
        realizadas por la red.
        - dst_dir: str. Directorio donde se guardarán las imágenes generadas.
        - patches_height: int. Altura de los parches utilizados.
        - patches_width: int. Ancho de los parches utilizados.
        - map_window_height: int. Altura que debe tener la ventana usada para
        pintar cada parche.
        - map_window_width: int. Altura que debe tener la ventana usada para
        pintar cada parche.
    """
    with open(predictions_path) as file:
        predictions_list = file.readlines()
    if dst_dir is None:
        dst_dir = os.path.split(predictions_path)[0]
    else:
        os.makedirs(dst_dir, exist_ok=True)
    predictions_dict = outputs.transform_to_dict(predictions_list)
    all_slides_ids = outputs.get_all_slides_ids(predictions_dict)
    for slide_id in all_slides_ids:
        map_image = evaluation.generate_map_of_predictions_slides(
            slide_id, predictions_dict,
            patches_height=patches_height,
            patches_width=patches_width,
            map_height=map_window_height,
            map_width=map_window_width)
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
    PARSER.add_argument(
        '--map_window_height',
        type=int,
        help="Altura que debe tener la ventana usada para pintar cada parche",
        default=8
    )
    PARSER.add_argument(
        '--map_window_width',
        type=int,
        help="Altura que debe tener la ventana usada para pintar cada parche",
        default=8
    )

    FLAGS = PARSER.parse_args()
    main(FLAGS.predictions_path, FLAGS.dst_dir,
         FLAGS.patches_height, FLAGS.patches_width,
         FLAGS.map_window_height, FLAGS.map_window_width)

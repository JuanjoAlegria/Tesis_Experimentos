"""Script para generar mapas de predicciones para slides completas. Este script
permite obtener slides con una máscara de predicciones sobrepuesta, o sólo la
máscara de predicciones (utilizando alpha_value=255). Además, permite generar
imágenes con las predicciones diferenciadas o no para el tejido control.
"""
import os
import glob
import argparse
from PIL import Image
from ...utils import excel, outputs
from ...experiments import evaluation


def main(predictions_path, images_dir, magnification,
         alpha_value, dst_dir, differentiate_control_tissue=False,
         excel_file=None):
    """Función para generar mapas de predicciones para slides completas. Este
    script permite generar imágenes de slides con una máscara de predicciones
    sobrepuesta, o sólo la máscara de predicciones (utilizando
    alpha_value=255,). Además, permite generar imágenes con las predicciones
    diferenciadas o no para el tejido control.


    Args:
        - predictions_path: str. Ruta al archivo .txt con las predicciones
        realizadas por la red.
        - images_dir: str. Directorio donde están ubicados las slides en
        formato jpeg.
        - magnification: str. Magnificación de las imágenes con que se
        trabajará. Esto, por si en la misma carpeta hay imágenes de la misma
        slide, pero adistinta magnificación.
        - alpha_value: int. Valor del canal alpha; debe estar en el intervalo
        [0, 255].
        - dst_dir: str. Directorio donde se guardarán las imágenes. Si no se
        entrega un valor, la carpeta será deducida a partir de
        predictions_path.
        - differentiate_control_tissue: bool. Indica si es que el tejido
        control de cada slide debe ser pintado con colores diferentes,
        graficando así que los parches extraídos de dicha zona son tratados
        de forma distinta al resto de los parches.
        - excel_file: str. Ruta al archivo excel con la información de las
        slides y anotaciones. Si es que differentiate_control_tissue es True,
        el valor de excel_file debe ser distinto de None.
    """
    if dst_dir is None:
        dst_dir = os.path.join(os.path.split(predictions_path)[0],
                               "test_maps")
    Image.MAX_IMAGE_PIXELS = 2**31
    with open(predictions_path) as file:
        predictions_lines = file.readlines()
    predictions_dict = outputs.transform_to_dict(predictions_lines)
    slides_ids = outputs.get_all_slides_ids(predictions_dict)
    colors_alpha = {key: (*val[0:3], alpha_value)
                    for key, val in evaluation.COLORS_RGBA.items()}

    if differentiate_control_tissue:
        biopsies_min_x_coords = excel.get_min_x_coords(excel_file)
        preds_biopsies, preds_control = outputs.split_predictions_dict(
            predictions_dict, biopsies_min_x_coords)
        colors_control_alpha = {key: (*val[0:3], alpha_value)
                                for key, val in
                                evaluation.COLORS_CONTROL_RGBA.items()}
        save_path_pattern = os.path.join(
            dst_dir, "{slide_id}_{magn}_alpha{alpha}_control.jpg")
    else:
        save_path_pattern = os.path.join(
            dst_dir, "{slide_id}_{magn}_alpha{alpha}.jpg")

    for slide_id in slides_ids:
        img_path = glob.glob(os.path.join(
            images_dir, "{}_{}_*".format(slide_id, magnification)))[0]
        img = Image.open(img_path)
        print("Trabajando en imagen", img_path)
        if differentiate_control_tissue:
            mask = evaluation.get_mask_of_predictions_control_tissue(
                slide_id, preds_biopsies, preds_control, img.size,
                magnification=magnification, colors=colors_alpha,
                colors_control=colors_control_alpha)
        else:
            mask = evaluation.get_mask_of_predictions(
                slide_id, predictions_dict, img.size,
                magnification=magnification, colors=colors_alpha)

        if alpha_value == 255:
            rgb = mask.convert('RGB')
        else:
            alpha = Image.new('RGBA', img.size)
            alpha.paste(img)
            alpha.paste(mask, mask=mask)
            rgb = alpha.convert('RGB')

        rgb.save(save_path_pattern.format(slide_id=slide_id,
                                          magn=magnification,
                                          alpha=alpha_value))

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
        '--images_dir',
        type=str,
        help="Directorio donde están ubicados las slides en formato jpeg.",
        required=True
    )
    PARSER.add_argument(
        '--magnification',
        type=str,
        help="""\
        Magnificación de las imágenes con que se trabajará. Esto, por si en la
        misma carpeta hay imágenes de la misma slide, pero a distinta
        magnificación""",
        required=True
    )
    PARSER.add_argument(
        '--alpha_value',
        type=int,
        help="Valor del canal alpha; debe estar en el intervalo [0, 255]",
        default=45
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
        '--differentiate_control_tissue',
        help="""\
        Indica si es que el tejido control de cada slide debe ser pintado con
        colores diferentes, graficando así que los parches extraídos de dicha
        zona son tratados de forma distinta al resto de los parches""",
        default=False,
        action="store_true"
    )

    PARSER.add_argument(
        '--excel_file',
        type=str,
        help="""\
        Ruta al archivo excel con la información de las slides y anotaciones.
        Si es que differentiate_control_tissue es True, el valor de excel_file
        debe ser distinto de None.
        """,
        default=None
    )
    FLAGS = PARSER.parse_args()
    main(FLAGS.predictions_path, FLAGS.images_dir, FLAGS.magnification,
         FLAGS.alpha_value, FLAGS.dst_dir, FLAGS.differentiate_control_tissue,
         FLAGS.excel_file)

import os
import glob
import argparse
from PIL import Image
from ...utils import outputs
from ...experiments import evaluation


def main(predictions_path, images_dir, magnification,
         alpha_value, dst_dir, just_masks=False):
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
    for slide_id in slides_ids:
        img_path = glob.glob(os.path.join(
            images_dir, "{}_{}_*".format(slide_id, magnification)))[0]
        print("Trabajando en imagen", img_path)
        if just_masks:
            img = Image.open(img_path)
            result = evaluation.get_mask_of_predictions(
                slide_id, predictions_dict, img.size,
                magnification=magnification, colors=colors_alpha)
            save_path = os.path.join(
                dst_dir, "{}_{}_mask.jpg".format(slide_id, magnification))
            if alpha_value == 255:
                result = result.convert("RGB")
        else:
            result = evaluation.generate_map_of_predictions(
                img_path, predictions_dict, slide_id, roi_id=None,
                magnification=magnification, colors=colors_alpha)
            save_path = os.path.join(
                dst_dir, "{}_{}_alpha{}.jpg".format(
                    slide_id, magnification, alpha_value))
        result.save(save_path)

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
        '--just_masks',
        help="""\
        True para generar sólo las máscaras, sin la imagen de fondo""",
        default=False,
        action="store_true"
    )

    FLAGS = PARSER.parse_args()
    main(FLAGS.predictions_path, FLAGS.images_dir, FLAGS.magnification,
         FLAGS.alpha_value, FLAGS.dst_dir, FLAGS.just_masks)

"""Script para generar gráficos de barra para cada modelo entrenado,
disgregados por slide y clase predicha.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ...utils import outputs, excel


def get_model_plot(images_names, images_predicted_classes,
                   slides_real_classes, title):
    """Genera un gráfico de barras para el modelo.

    Args:
        - images_names: list[str]. Lista con los nombres de las imágenes, en la
        forma {label}/{slide_id}_rest.jpg.
        - images_predicted_classes: list[str]. Lista con las clases predichas
        por el algoritmo, correlativa a images_names.
        - slides_real_classes: dict[str->str]. Diccionario donde cada llave es
        la id de una slide, teniendo como valor asociado la clase real a la
        que pertenece esa slide
        - title: str. Título del gráfico.

    Returns:
        matplotlib.figure.Figure, gráfico de barras disgregado por slides y
        clase predicha.
    """
    slides_results = outputs.get_slides_aggregated_results(
        images_names, images_predicted_classes)
    n_slides = len(slides_results)
    indexes_plot = np.arange(n_slides)
    bar_width = 0.2

    slides_ids = sorted(slides_results.keys())
    colors = ["r", "y", "g"]
    labels = ["Negative (0+ or 1+)",
              "Equivocal (2+)",
              "Positive (3+)"]

    fig, axes = plt.subplots(figsize=(10, 8))
    for idx, pred_class in enumerate(["0", "1", "2"]):
        n_pred_images = [slides_results[slide_id][pred_class] /
                         slides_results[slide_id]["total"]
                         for slide_id in slides_ids]
        axes.bar(indexes_plot + bar_width * idx,
                 n_pred_images, bar_width,
                 color=colors[idx], label=labels[idx])

    ticks = ["{} ({}+)".format(slide_id, slides_real_classes[slide_id])
             for slide_id in slides_ids]
    axes.set_xlabel('Slides')
    axes.set_ylabel('% of patches classified with that label')
    axes.set_title(title)
    axes.set_xticks(indexes_plot + bar_width)
    axes.set_xticklabels(ticks)
    axes.legend()
    fig.tight_layout()
    return fig


def get_plot_title(predictions_path):
    """Infiere el título del gráfico a partir de la ruta del archivo de
    predicciones.

    Args:
        - predictions_path: str. Ruta al archivo con predicciones. Ejemplo:
        ihc_patches_kfold_fixed_ids_x10_fine_tuning_experiment_5/pred.txt

    Returns:
        str, título del gráfico.
    """
    title_template = "Inception V3{extra}, magnification {magnification}, " + \
        "fold {n_fold}"
    prefix, _ = os.path.split(predictions_path)
    _, experiment_name = os.path.split(prefix)
    substrings = experiment_name.split("_")
    magnification = substrings[5]
    n_fold = substrings[-1]
    if "random" in substrings and "fine" in substrings:
        return title_template.format(
            magnification=magnification, n_fold=n_fold,
            extra=" with data augmentation and fine tuning")
    elif "random" in substrings:
        return title_template.format(
            magnification=magnification, n_fold=n_fold,
            extra=" with data augmentation")
    elif "fine" in substrings:
        return title_template.format(
            magnification=magnification, n_fold=n_fold,
            extra=" with fine tuning")
    else:
        return title_template.format(
            magnification=magnification, n_fold=n_fold,
            extra="")


def main(predictions_path, excel_file, plot_title, dst_dir):
    """Inicializa las variables, genera el gráfico y lo guarda

    Args:
        - predictions_path: str. Ruta al archivo .txt con las predicciones
        realizadas por la red.
        - excel_file: str. Ruta al archivo excel con la información de las
        slides y anotaciones.
        - plot_title: str. Título del gráfico. Si no se entrega un valor, el
        título será deducido a partir de predictions_path.
        - dst_dir: str. Directorio donde se guardará el gráfico. Si no se
        entrega un valor, la carpeta será deducida a partir de
        predictions_path.
    """
    if plot_title is None:
        plot_title = get_plot_title(predictions_path)
    if dst_dir is None:
        dst_dir = os.path.split(predictions_path)[0]

    with open(predictions_path) as file:
        output_lines = file.readlines()
    slides_ids, slides_evaluation = excel.get_valid_slides_ids(excel_file)
    evaluation_dict = dict(zip(slides_ids, slides_evaluation))
    filenames, _, predicted = outputs.transform_to_lists(output_lines)
    fig = get_model_plot(filenames, predicted, evaluation_dict, plot_title)
    fig.savefig(os.path.join(dst_dir, "predictions_plot.png"))

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
        '--excel_file',
        type=str,
        help="""\
        Ruta al archivo excel con la información de las slides y anotaciones.\
        """,
        default=os.path.join(os.getcwd(), "data", "extras",
                             "ihc_slides", "HER2.xlsx")
    )
    PARSER.add_argument(
        '--plot_title',
        type=str,
        help="""\
        Título del gráfico. Si no se entrega un valor, el título será deducido
        a partir de predictions_path.""",
        default=None
    )
    PARSER.add_argument(
        '--dst_dir',
        type=str,
        help="""\
        Directorio donde se guardará el gráfico. Si no se entrega un valor,
        la carpeta será deducida a partir de predictions_path.""",
        default=None
    )

    FLAGS = PARSER.parse_args()
    main(FLAGS.predictions_path, FLAGS.excel_file,
         FLAGS.plot_title, FLAGS.dst_dir)

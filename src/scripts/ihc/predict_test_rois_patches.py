"""Script para generar y guardar las predicciones para las imágenes extraídas
de ROIs que pertenecen a las biopsias de entrenamiento. Esto, porque previamente
la única evaluación que se realizaba sobre esas biopsias era a nivel global, mas
no se aplicaban métricas de ML en ese conjunto"""

import os
import glob
import json
import argparse
from ...experiments import predictions
from ...utils import outputs


def main(dataset_path, images_dir, experiment_saved_model_dir,
         dst_file, batch_size):
    """Genera y guarda las predicciones para las imágenes extraídas de ROIs que
    pertenecen a las biopsias de entrenamiento.

    Args:
        - dataset_path: str. Ruta al dataset en formato json.
        El dataset en formato json debe contener las siguientes claves:
            - test_rois_features: list[str]. Parches extraídos de ROIs que
            pertenecen al conjunto de evaluación, donde cada
            elemento corresponde al nombre de un archivo en formato
            label_original / filename.jpg, relativo a images_dir.
            - test_rois_labels: list[int]. Etiquetas de parches extraídos de
            ROIs que pertenecen al conjunto de evaluación,
            donde cada elemento es un entero en el rango[0, n_classes - 1].

        - images_dir: str. Directorio donde se encuentran las imágenes.

        - experiment_saved_model_dir: str. Directorio donde se encuentra
        guardado el saved_model. Se espera que sea el directorio con el nombre
        del experimento, donde adentro se ubicará una carpeta con el nombre
        del modelo(red utilizada) y un nivelmás abajo se encuentre una
        carpeta cuyo nombre sea un timestamp.
        Ejemplo: si la ruta completa es
        ~/experiments/saved_models/experiment_name/inception_v3/1533272247,
        entonces la ruta entregada acá debe ser:
        ~/experiments/saved_models/experiment_name.
        Se usará siempre el modelo guardado más reciente(es decir, el que
        tenga el timestamp más alto)

        - dst_file: str. Ubicación donde se quiere guardar el archivo con los
        resultados

        - batch_size: int. Tamaño del batch al predecir las clasificaciones de
        las imágenes.
    """
    experiment_saved_model_pattern = os.path.join(
        experiment_saved_model_dir, "*", "*")
    saved_models = glob.glob(experiment_saved_model_pattern)
    # en caso de haber más de un modelo, se escoge el más reciente
    saved_models.sort()
    saved_model_dir = saved_models[-1]
    with open(dataset_path) as file:
        dataset_json = json.load(file)
    filenames = dataset_json["test_rois_features"]
    labels = dataset_json["test_rois_labels"]
    paths = list(map(lambda f: os.path.join(images_dir, f),
                     filenames))
    network_predictions = predictions.get_predictions(
        saved_model_dir, paths, batch_size)
    outputs.write_output(filenames, network_predictions,
                         real_labels=labels, output_file=dst_file)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help="""
        Ruta al dataset en formato json.
        El dataset en formato json debe contener las siguientes claves:
            - test_rois_features: list[str]. Parches extraídos de ROIs que
            pertenecen al conjunto de evaluación, donde cada
            elemento corresponde al nombre de un archivo en formato
            label_original / filename.jpg, relativo a images_dir.
            - test_rois_labels: list[int]. Etiquetas de parches extraídos de
            ROIs que pertenecen al conjunto de evaluación,
            donde cada elemento es un entero en el rango[0, n_classes - 1].
        """
    )
    PARSER.add_argument(
        '--images_dir',
        type=str,
        required=True,
        help='Directorio donde se encuentran las imágenes.'
    )
    PARSER.add_argument(
        '--experiment_saved_model_dir',
        type=str,
        help="""
        Directorio donde se encuentra guardado el saved_model. Se espera
        que sea el directorio con el nombre del experimento, donde adentro se
        ubicará una carpeta con el nombre del modelo(red utilizada) y un nivel
        más abajo se encuentre una carpeta cuyo nombre sea un timestamp.
        Ejemplo: si la ruta completa es
        ~/experiments/saved_models/experiment_name/inception_v3/1533272247,
        entonces la ruta entregada acá debe ser:
        ~/experiments/saved_models/experiment_name.

        Se usará siempre el modelo guardado más reciente(es decir, el que
        tenga el timestamp más alto)
        """,
        required=True
    )
    PARSER.add_argument(
        '--dst_file',
        type=str,
        help="Ubicación donde se quiere guardar el archivo con los resultados",
        required=True
    )
    PARSER.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help="""
        Tamaño del batch al predecir las clasificaciones de las imágenes.
        """
    )
    FLAGS = PARSER.parse_args()
    main(FLAGS.dataset_path, FLAGS.images_dir,
         FLAGS.experiment_saved_model_dir, FLAGS.dst_file, FLAGS.batch_size)

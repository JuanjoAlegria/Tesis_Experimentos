"""Script para generar y guardar las predicciones para el conjunto de
entrenamiento utilizado al entrenar el algoritmo"""

import os
import glob
import json
import argparse
from ...experiments import predictions
from ...utils import outputs


def main(dataset_path, train_images_dir, experiment_saved_model_dir,
         dst_file, batch_size):
    """Genera y guarda las predicciones para el conjunto de entrenamiento
    utilizado al entrenar el algoritmo

    Args:
        - dataset_path: str. Ruta al dataset en formato json.
        El dataset en formato json debe contener las siguientes claves:
            - train_filenames: list[str]. Conjunto de entrenamiento, donde cada
            elemento corresponde al nombre de un archivo en formato
            label_original / filename.jpg, relativo a train_images_dir.
            - train_labels: list[int]. Etiquetas del conjunto de entrenamiento,
            donde cada elemento es un entero en el rango[0, n_classes - 1].

        - train_images_dir: str. Directorio donde se encuentran las imágenes
        de entrenamiento

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
    train_filenames = dataset_json["train_features"]
    train_labels = dataset_json["train_labels"]
    train_paths = list(map(lambda f: os.path.join(train_images_dir, f),
                           train_filenames))
    train_predictions = predictions.get_predictions(
        saved_model_dir, train_paths, batch_size)
    outputs.write_output(train_filenames, train_predictions,
                         real_labels=train_labels, output_file=dst_file)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help="""
        Ruta al dataset en formato json.

        El dataset en formato json debe contener las siguientes claves:
        - train_filenames: list[str]. Conjunto de entrenamiento, donde cada
        elemento corresponde al nombre de un archivo en formato
        label_original / filename.jpg, relativo a train_images_dir.
        - train_labels: list[int]. Etiquetas del conjunto de entrenamiento,
        donde cada elemento es un entero en el rango[0, n_classes - 1].
        """
    )
    PARSER.add_argument(
        '--train_images_dir',
        type=str,
        required=True,
        help='Directorio donde se encuentran las imágenes de entrenamiento.'
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
    main(FLAGS.dataset_path, FLAGS.train_images_dir,
         FLAGS.experiment_saved_model_dir, FLAGS.dst_file, FLAGS.batch_size)

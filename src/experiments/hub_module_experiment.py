"""Módulo con funciones para llevar a cabo un experimento utilizando un
hub_module_model:

El módulo contiene está dividido de la siguiente forma:

hub_model_experiment_parser:
    Entrega un parser con los parámetros que se espera sean entregados por
    línea de comando.

class HubModelExperiment:
    Clase que se encarga de crear un tf.Estimator con la definición de un
    hub_model_estimator y entrenar y evaluar el modelo, utilizando para ello
    los parámetros entregados por línea de comando (flags). Además, posee
    funciones utilitarias para crear input_fn de entrenamiento, validación y
    prueba a partir de los FLAGS entregados, usando para ello las funciones
    genéricas del módulo dataset.
"""

import os
import sys
import glob
import json
import shutil
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from ..dataset import data_utils, tf_data_utils
from ..models.hub_module_model import hub_bottleneck_model_fn


class HubModelExperiment:
    """Clase para llevar a cabo un experimento con un hub_module_estimator.

    Esta clase está pensada para ser usada tras recibir parámetros desde la
    línea de comandos, por lo cual recibe un objeto llamado flags con dicho
    parámetros. Además, recibe el número de clases sobre las cuales se
    realizará la clasificación (este parámetro se infiere desde el dataset y no
    se entrega vía línea de comandos, por eso la diferenciación entre ambos
    parámetros).

    Args:
        - flags: argparse.Namespace. Objeto con parámetros necesarios para el
        modelo. Se genera tras llamar a parser.parse_know_args().
        - n_classes: int. Número de clases a clasificar.

    Returns:
        HubModelExperiment con un tf.Estimator basado en hub_module_estimator.
    """

    def __init__(self, flags, n_classes):
        self.n_classes = n_classes

        # Pasamos las variables relacionadas a la estructura del experimento
        self.experiment_name = flags.experiment_name
        self.model_name = flags.model_name
        self.logs_and_checkpoints_dir = flags.logs_and_checkpoints_dir
        self.export_model_dir = flags.export_model_dir
        self.bottlenecks_dir = flags.bottlenecks_dir
        self.results_dir = flags.results_dir
        self.remove_prev_ckpts_and_logs = flags.remove_prev_ckpts_and_logs
        self.random_seed = flags.random_seed

        # Pasamos las variables relacionadas al dataset
        # images_dir: hacemos esto para asegurar que el directorio esté
        # en formato absoluto, y además para que tenga un slash al final
        # siempre
        self.images_dir = os.path.join(os.path.abspath(flags.images_dir), "")
        self.flip_left_right = flags.flip_left_right
        self.random_crop = flags.random_crop
        self.random_scale = flags.random_scale
        self.random_brightness = flags.random_brightness

        # Pasamos las variables relacionadas al entrenamiento
        self.train_batch_size = flags.train_batch_size
        self.test_batch_size = flags.test_batch_size
        self.num_epochs = flags.num_epochs
        self.learning_rate = flags.learning_rate
        self.tensors_to_log_train = flags.tensors_to_log_train
        self.tensors_to_log_val = flags.tensors_to_log_val
        self.save_checkpoints_steps = flags.save_checkpoints_steps
        self.eval_frequency = flags.eval_frequency
        self.fine_tuning = flags.fine_tuning

        # Otras variables importantes
        self.cache_bottlenecks = not self.fine_tuning and \
            not tf_data_utils.should_distort_images(
                self.flip_left_right, self.random_crop,
                self.random_scale, self.random_brightness)
        # Obtenemos el module_spec correspondiente
        module_url = get_module_url(self.model_name)
        self.module_spec = hub.load_module_spec(module_url)
        self.module_image_shape = hub.get_expected_image_size(self.module_spec)
        self.module_image_depth = hub.get_num_image_channels(self.module_spec)

        self.__init_log_and_random_seeds()
        self.__prepare_filesystem()
        self.__save_config_file(flags)
        self.estimator = self.__build_estimator()

    def __init_log_and_random_seeds(self):
        """Inicializa semillas aleatorias para asegurar reproducibilidad, y
        configura TensorFlow para loggear mensajes con prioridad INFO.
        """
        if self.random_seed is not None:
            tf.set_random_seed(self.random_seed)
        tf.logging.set_verbosity(tf.logging.INFO)

    def __prepare_filesystem(self):
        """Inicializa variables que no fueron entregadas, elimina logs y
        checkpoints anteriores en caso de se requerido y crea las carpetas
        necesarias para ejecutar el experimento en caso de que no existan.
        """
        root_path = os.getcwd()
        # root_path = os.path.abspath(os.path.join(src_code_path, ".."))
        # Revisamos si es que el experimento tiene nombre, y en caso de no ser
        # así, le creamos uno
        if self.experiment_name == "":
            prev_unnamed_experiments = glob.glob(os.path.join(
                root_path, "unnamed_experiment_*"))
            self.experiment_name = "unnamed_experiment_{n}".format(
                n=len(prev_unnamed_experiments) + 1)
        # Logs y checkpoints son dependientes del experimento específico
        if self.logs_and_checkpoints_dir == "":
            self.logs_and_checkpoints_dir = os.path.join(
                root_path, "logs_and_checkpoints", self.experiment_name)
        # Bottlenecks, en cambio, dependen sólo del modelo y de las imágenes a
        # entrenar
        if self.bottlenecks_dir == "":
            # Nos aseguramos de que fixed_images_dir NO tenga un slash al final
            fixed_images_dir = self.images_dir[:-1] \
                if self.images_dir[-1] == "/" else self.images_dir
            # Nos aseguramos de que bottlenecks_dir SÍ tenga un slash al final
            _, images_dir_name = os.path.split(fixed_images_dir)
            self.bottlenecks_dir = os.path.join(
                root_path, "data", "bottlenecks",
                self.model_name, images_dir_name, "")
        # Export_dir depende del experimento y del modelo
        if self.export_model_dir == "":
            self.export_model_dir = os.path.join(
                root_path, "saved_models", self.experiment_name,
                self.model_name)
        # Results_dir depende sólo del experimento
        if self.results_dir == "":
            self.results_dir = os.path.join(
                root_path, "results", self.experiment_name)

        if self.remove_prev_ckpts_and_logs and \
                os.path.exists(self.logs_and_checkpoints_dir):
            shutil.rmtree(self.logs_and_checkpoints_dir)

        os.makedirs(self.logs_and_checkpoints_dir, exist_ok=True)
        os.makedirs(self.export_model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        if self.cache_bottlenecks:
            os.makedirs(self.bottlenecks_dir, exist_ok=True)
            for label_index in range(self.n_classes):
                os.makedirs(
                    os.path.join(self.bottlenecks_dir, str(label_index)),
                    exist_ok=True)

    def __save_config_file(self, flags):
        """ Guarda un archivo json en self.results_dir con la configuración
        utilizada

        Args:
            - flags: argparse.Namespace. Objeto con los parámetros del
            modelo. Se genera tras llamar a parser.parse_known_args()
            """
        flags_as_dict = vars(flags)
        config_file_path = os.path.join(self.results_dir, "config.json")
        with open(config_file_path, "w") as config_json:
            json.dump(flags_as_dict, config_json)

    def __build_estimator(self, export=False):
        """Construye un tf.Estimator basado en hub_module_estimator.

        Args:
            - export: bool. Indica si se construirá un tf.estimator para ser
            exportado; en tal caso, cache_bottlenecks = False, independiente
            de si los bottlenecks son o no cacheados durante el entrenamiento.
            - warm_start: WarmStartSettings. Configuración para iniciar "en
            caliente" el estimador.
        Returns:
            tf.Estimator, con model_fn igual a hub_module_estimator_model_fn.
        """

        cache_bottlenecks = False if export else self.cache_bottlenecks

        params = {"module_spec": self.module_spec,
                  "model_name": self.model_name,
                  "n_classes": self.n_classes,
                  "cache_bottlenecks": cache_bottlenecks,
                  "bottlenecks_dir": self.bottlenecks_dir,
                  "learning_rate": self.learning_rate,
                  "fine_tuning": self.fine_tuning}

        classifier = tf.estimator.Estimator(
            model_fn=hub_bottleneck_model_fn,
            model_dir=self.logs_and_checkpoints_dir,
            config=tf.estimator.RunConfig(
                tf_random_seed=self.random_seed,
                save_checkpoints_steps=self.save_checkpoints_steps,
                keep_checkpoint_max=20,
                save_summary_steps=2),
            params=params)

        return classifier

    def __train_input_fn(self, train_filenames, train_labels):
        """Construye un dataset para ser utilizado como input_fn en la fase de
        entrenamiento.

        La principal diferencia con __eval_input_fn es que aquí si importan las
        distorsiones aleatorias, las cuales no son utilizadas en el modo de
        evaluación.

        Args:
            - train_filenames: [str]. Nombres de los archivos que serán
            utilizados en el entrenamiento. Su formato es
            label_original/filename.jpg, y la ubicación es relativa a
            self.images_dir.
            - train_labels: [int]. Etiquetas numéricas con las cuales se
            realizará el entrenamiento.

        Returns:
            tf.data.Dataset, con mapeo de filenames a imágenes y distorsiones
            aleatorias en caso de ser requeridas. Además, se le aplican las
            operaciones shuffle, repeat y batch.
        """
        return tf_data_utils.create_images_dataset(
            train_filenames,
            train_labels,
            image_shape=self.module_image_shape,
            image_depth=self.module_image_depth,
            src_dir=self.images_dir,
            shuffle=True,
            num_epochs=self.num_epochs,
            batch_size=self.train_batch_size,
            flip_left_right=self.flip_left_right,
            random_crop=self.random_crop,
            random_scale=self.random_scale,
            random_brightness=self.random_brightness)

    def __test_input_fn(self, test_filenames, test_labels):
        """Construye un dataset para ser utilizado como input_fn en la fase de
        pruebas.

        La principal diferencia con __train_input_fn es que aquí no importan
        las distorsiones aleatorias, las cuales sí son utilizadas en el modo de
        entrenamiento. Además, esta función es para el modo de prueba posterior
        al entrenamiento; es decir, con un conjunto de validación distinto al
        conjunto de validación. Por ello, num_epochs = 1, y no se permuta
        aleatoriamente el dataset. Además, batch_size = 100 (por ahora).

        Args:
            - test_filenames: [str]. Nombres de los archivos que serán
            utilizados en la validación. Su formato es
            label_original/filename.jpg, y la ubicación es relativa a
            self.images_dir.
            - test_labels: [int]. Etiquetas numéricas con las cuales se
            realizará la validación.
            - final_test: bool. Si es True, batch_size será igual a
            len(test_filenames), ya que se evaluará con todo el conjunto.
            En caso contrario, batch_size será igual a
            self.test_batch_size.

        Returns:
            tf.data.Dataset, con mapeo de filenames a imágenes. Además, se
            aplican las operaciones shuffle, repeat y batch.
        """

        return tf_data_utils.create_images_dataset(
            test_filenames, test_labels, image_shape=self.module_image_shape,
            image_depth=self.module_image_depth,
            src_dir=self.images_dir, shuffle=False, num_epochs=1,
            batch_size=self.test_batch_size)

    def __serving_input_receiver_fn(self):
        """Construye una input_fn para ser utilizada al exportar el modelo.

        Returns:
           ServingInputReceiver, listo para ser usado en conjunto con
           estimator.export_model
        """
        feature_spec = {
            'image': tf.FixedLenFeature([], dtype=tf.string)
        }

        default_batch_size = 1
        serialized_tf_example = tf.placeholder(
            dtype=tf.string, shape=[default_batch_size],
            name='input_image_tensor')

        received_tensors = {'images': serialized_tf_example}
        features = tf.parse_example(serialized_tf_example, feature_spec)

        map_fn = lambda img: tf_data_utils.decode_image_from_string(
            img, self.module_image_shape, self.module_image_depth)

        features['image'] = tf.map_fn(
            map_fn, features['image'], dtype=tf.float32)

        return tf.estimator.export.ServingInputReceiver(features,
                                                        received_tensors)

    def train(self, train_filenames, train_labels):
        """Entrena el tf.Estimator

        Primero construye la input_fn con que se alimentará el modelo, y luego
        lo entrena.

        Args:
            - train_filenames: [str]. Nombres de los archivos que serán
            utilizados en el entrenamiento. Su formato es
            label_original/filename.jpg, y la ubicación es relativa a
            self.images_dir.
            - train_labels: [int]. Etiquetas numéricas con las cuales se
            realizará el entrenamiento.
            - hooks: [SessionRunHooks]. Lista de hooks que deben ser ejecutados
            durante el entrenamiento.
        """
        train_input_fn = lambda: self.__train_input_fn(
            train_filenames, train_labels)
        train_log = build_logging_tensor_hook(self.tensors_to_log_train)
        self.estimator.train(input_fn=train_input_fn,
                             steps=self.num_epochs, hooks=[train_log])

    def test_and_predict(self, filenames, labels, partition_name=""):
        """Evalúa el tf.Estimator

        Primero construye la input_fn con que se alimentará el modelo, y luego
        lo evalúa.

        Args:
            - filenames: [str]. Nombres de los archivos que serán
            utilizados en la evaluación. Su formato es
            label_original/filename.jpg, y la ubicación es relativa a
            self.images_dir.
            - labels: [int]. Etiquetas numéricas con las cuales se
            realizará la evaluación.
        """
        test_input_fn = lambda: self.__test_input_fn(filenames,
                                                     labels)

        eval_results = self.estimator.evaluate(input_fn=test_input_fn)
        print(eval_results)

        predictions = self.estimator.predict(input_fn=test_input_fn)
        header = "Filename / Real Class / Predicted Class\n"
        lines = [header]
        for index, pred in enumerate(predictions):
            filename = filenames[index]
            real_class = labels[index]
            predicted_class = pred['classes']
            line = "{fn} {real} {predicted}\n".format(
                fn=filename, real=real_class, predicted=predicted_class)
            lines.append(line)
        output = os.path.join(
            self.results_dir, partition_name + "_predictions.txt")
        with open(output, "w") as file:
            file.writelines(lines)

    def train_and_evaluate(self, train_filenames, train_labels,
                           validation_filenames, validation_labels):
        """Entrena y evalúa un modelo durante el mismo loop

        Args:
            - train_filenames: [str]. Nombres de los archivos que serán
            utilizados en el entrenamiento. Su formato es
            label_original/filename.jpg, y la ubicación es relativa a
            self.images_dir.
            - train_labels: [int]. Etiquetas numéricas con las cuales se
            realizará el entrenamiento.
            - validation_filenames: [str]. Nombres de los archivos que serán
            utilizados en la evaluación. Su formato es
            label_original/filename.jpg, y la ubicación es relativa a
            self.images_dir.
            - validation_labels: [int]. Etiquetas numéricas con las cuales se
            realizará la evaluación.
            - train_hooks: [SessionRunHooks]. Lista de hooks que deben ser
            ejecutados durante el entrenamiento.
            - validation_ooks: [SessionRunHooks]. Lista de hooks que deben ser
            ejecutados durante la evaluación.
        """

        iterations = self.num_epochs // self.eval_frequency
        print("# iterations:", iterations)

        train_log = build_logging_tensor_hook(self.tensors_to_log_train)
        val_log = build_logging_tensor_hook(self.tensors_to_log_val)

        train_input_fn = lambda: self.__train_input_fn(
            train_filenames, train_labels)
        eval_input_fn = lambda: self.__test_input_fn(
            validation_filenames, validation_labels)

        for __ in range(iterations):
            train_filenames, train_labels = data_utils.shuffle_dataset(
                train_filenames, train_labels)
            self.estimator.train(input_fn=train_input_fn,
                                 steps=self.eval_frequency,
                                 hooks=[train_log])
            self.estimator.evaluate(eval_input_fn, hooks=[val_log])

    def export_graph(self):
        """Exporta el grafo como un SavedModel estándar para ser utilizado
        posteriormente
        """
        estimator_for_export = self.__build_estimator(export=True)
        estimator_for_export.export_savedmodel(
            self.export_model_dir, self.__serving_input_receiver_fn,
            as_text=True)

URLS_MODEL = {
    "inception_v3":
        "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1"
}


def get_module_url(model_name):
    """Retorna la URL desde donde descargar el modelo correspondiente.

    Args:
        - model_name: str. Nombre del modelo deseado.

    Returns:
        - model_url: str. URL desde donde descargar el modelo.

    Raises:
        ValueError, si es que el nombre del modelo no corresponde a alguna
        arquitectura conocida.
    """
    if model_name in URLS_MODEL:
        return URLS_MODEL[model_name]
    keys_string = ", ".join(URLS_MODEL.keys())
    error_template = "{model_name} inválido. Los nombres soportados" + \
        "son: {valid_names}"
    raise ValueError(error_template.format(
        model_name=model_name, valid_names=keys_string))


def build_logging_tensor_hook(tensors_names):
    """Contruye un LoggingTensorHook a partir de los nombres de tensores
    entregados.

    Args:
        - tensors_names: list[str]. Nombre de los tensores que se desea
        loggear. Opciones válidas son:
            - loss
            - global_step
            - filenames
            - images
            - has_prev_bottlenecks
            - write_bottlenecks

    Returns:
        tf.train.LoggingTensorHook, configura para loggear los tensores
        pedidos en cada interación.
    """
    tensors_real_names = [name + ":0" for name in tensors_names]
    tensors_dict = {display_name: real_name
                    for display_name, real_name in zip(tensors_names,
                                                       tensors_real_names)}
    # Patch para tensor "loss"
    if "loss" in tensors_dict:
        tensors_dict["loss"] = "my_loss/value:0"
    return tf.train.LoggingTensorHook(tensors=tensors_dict, every_n_iter=1)


def get_parser():
    """Retorna un parser de línea de comandos con los parámetros que el
    usuario puede entregar mediante línea de comandos.

    Returns:
        argparse.ArgumentParser ya configurado con los argumentos que se
        pedirán por línea de comandos.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_json',
        type=str,
        required=True,
        help="""\
        Dataset en formato json, con las siguientes claves:
        - train_filenames: list[str]. Conjunto de entrenamiento, donde cada
        elemento corresponde al nombre de un archivo en formato
        label_original/filename.jpg, relativo a images_dir.
        - train_labels: list[int]. Etiquetas del conjunto de entrenamiento,
        donde cada elemento es un entero en el rango [0, n_classes - 1].
        - validation_filenames: list[str]. Conjunto de validación, análogo a 
        train_filenaames.
        - validation_labels: list[int]. Etiquetas de validación, análogo a 
        train_labels.
        - test_filenames: list[str]. Conjunto de prueba, análogo a 
        train_filenaames.
        - test_labels: list[int]. Etiquetas de prueba, análogo a 
        train_labels.\
        """
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default="",
        help="Nombre del experimento"
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='inception_v3',
        help="""\
        Nombre del módulo del Hub de Tensorflow a utilizar. En base a esto, se
        descargará el módulo correspondiente desde el Hub.\
        """
    )
    parser.add_argument(
        '--logs_and_checkpoints_dir',
        type=str,
        default="",
        help="Ubicación donde se almacenarán los checkpoints y logs del modelo"
    )
    parser.add_argument(
        '--export_model_dir',
        type=str,
        default="",
        help="Ubicación donde se exportará el modelo ya entrenado"
    )

    parser.add_argument(
        '--bottlenecks_dir',
        type=str,
        default="",
        help='Directorio donde se almacenarán los bottlenecks'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default="",
        help='Directorio donde se almacenarán los resultados'
    )
    parser.add_argument(
        '--remove_prev_ckpts_and_logs',
        default=False,
        help="""\
        Indica si es que se debe remover o no checkpoints y logs previos que
        puedan estar en logs_and_checkpoints_dir.\
        """,
        action='store_true'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=0,
        help='Semilla aleatoria (útil para obteer resultados reproducibles)',
    )
    parser.add_argument(
        '--images_dir',
        type=str,
        required=True,
        help='Directorio donde se encuentran las imágenes.'
    )
    parser.add_argument(
        '--flip_left_right',
        default=False,
        help="""\
        Indica si es que se debe o no rotar horizontalmente la imagen. Sólo
        tiene efecto en las imágenes de entrenamiento.\
        """,
        action='store_true'
    )
    parser.add_argument(
        '--random_crop',
        type=int,
        default=0,
        help="""\
        Porcentaje que indica el margen total a utilizar alrededor de la
        caja de recorte (crop box). Sólo tiene efecto en las imágenes de
        entrenamiento.\
        """
    )
    parser.add_argument(
        '--random_scale',
        type=int,
        default=0,
        help="""\
        Porcentaje que indica el rango de cuánto se debe variar la escala
        de la imagen. Sólo tiene efecto en las imágenes de entrenamiento.\
        """
    )
    parser.add_argument(
        '--random_brightness',
        type=int,
        default=0,
        help="""\
        Rango por el cual multiplicar aleatoriamente los pixeles de la imagen.
        Sólo tiene efecto en las imágenes de entrenamiento.\
        """
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=100,
        help='Cuántas imágenes entrenar en cada paso.'
    )
    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=100,
        help="""\
        Tamaño del batch a utilizar al evaluar el conjunto. Sin importar el
        valor entregado, se evaluará todo el conjunto de prueba, sólo que no
        todo al mismo tiempo.\
        """
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=4000,
        help='Cuántos pasos de entrenamiento se deben realizar.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Tasa de aprendizaje a utilizar durante el entrenamiento.'
    )
    parser.add_argument(
        '--tensors_to_log_train',
        type=str,
        nargs='*',
        help="""\
        Tensores que se deben loggear al entrenar. Opciones válidas son: 
            - loss
            - global_step
            - filenames
            - images
            - has_prev_bottlenecks
            - write_bottlenecks
        """,
        default=["loss", "global_step"]
    )
    parser.add_argument(
        '--tensors_to_log_val',
        type=str,
        nargs='*',
        help="""\
        Tensores que se deben loggear al evaluar. Opciones válidas son: 
            - loss
            - global_step
            - filenames
            - images
            - has_prev_bottlenecks
            - write_bottlenecks
        """,
        default=["loss"]
    )
    parser.add_argument(
        '--save_checkpoints_steps',
        type=int,
        default=10,
        help='Cada cuántos pasos se deben guardar los checkpoints.'
    )
    parser.add_argument(
        '--eval_frequency',
        type=int,
        default=1200,
        help='Cada cuántos pasos se debe evaluar el experimento.'
    )
    parser.add_argument(
        '--fine_tuning',
        default=False,
        help='True para reentrenar y ajustar capas internas del modelo',
        action='store_true'
    )
    return parser


def main(_):
    """Función de entrada. Se obtienen los datasets desde dataset_json,
    y se crea un HubModelExperiment, con el cual se entrena y evalúa el modelo,
    para finalmente ser exportado.
    """
    np.random.seed(FLAGS.random_seed)
    with open(FLAGS.dataset_json) as file:
        dataset_json = json.load(file)

    print("Obteniendo dataset desde archivo json")
    train_filenames = np.array(dataset_json["train_features"])
    train_labels = np.array(dataset_json["train_labels"])
    validation_filenames = np.array(dataset_json["validation_features"])
    validation_labels = np.array(dataset_json["validation_labels"])
    test_filenames = np.array(dataset_json["test_features"])
    test_labels = np.array(dataset_json["test_labels"])
    n_classes = len(set(train_labels))

    print("Creando HubModelExperiment")
    hub_experiment = HubModelExperiment(FLAGS, n_classes)

    print("Entrenando")
    hub_experiment.train_and_evaluate(train_filenames, train_labels,
                                      validation_filenames, validation_labels)
    # hub_experiment.train(train_filenames, train_labels)
    print("Evaluando con el conjunto de validación")
    hub_experiment.test_and_predict(
        validation_filenames, validation_labels, "test")
    print("Evaluando con el conjunto de prueba")
    hub_experiment.test_and_predict(test_filenames, test_labels, "test")
    print("Exportando")
    hub_experiment.export_graph()

if __name__ == "__main__":
    PARSER = get_parser()
    FLAGS, UNPARSED = PARSER.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + UNPARSED)

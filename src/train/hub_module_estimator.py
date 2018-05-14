"""Módulo con funciones y clases para realizar transfer learning, utilizando un
módulo del Hub de TensorFlow.

El módulo contiene está dividido de la siguiente forma:

hub_model_estimator:
    Métodos para crear un grafo de TensorFlow, en la forma de tf.Estimator,
    utilizando como base un módulo del Hub de Tensorflow. Lo que se hace es
    añadir una nueva capa final de clasificación a una CNN (Convolutional
    Neural Network) ya entrenada. Además, se pueden cachear los bottlenecks,
    para evitar recalcularlos en cada iteración y así reducir el tiempo de
    entrenamiento.

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
import glob
import json
import shutil
import argparse
import tensorflow as tf
import tensorflow_hub as hub
from . import dataset

URLS_MODEL = {
    "inception_v3":
        "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1"
}


def has_bottlenecks_map_fn(bottleneck_path):
    """Dado un nombre de un archivo de bottleneck, revisa si es que está en
    disco. Función diseñada para ser usada con tf.map_fn(...).

    Args:
        - bottleneck_path: Tensor de tipo tf.string. Ubicación del archivo en
        cuestión.

    Returns:
        Tensor de tipo tf.bool, con valor True si es que el archivo está en
        disco, False en caso contrario.
    """
    def _check_exists(file_path_array):
        return tf.gfile.Exists(file_path_array)
    return tf.py_func(_check_exists, [bottleneck_path], tf.bool)


def get_bottlenecks_map_fn(has_bottleneck, filename,
                           image, bottlenecks_extractor):
    """Obtiene el bottleneck correspondiente a una imagen. Función diseñada
    para ser usada con tf.map_fn(...).

    La función decide si es que debe recuperar el bottleneck desde el disco
    (si es que has_bottleneck == True) o si es que debe calcularlo en el
    instante, usando para ello el bottlenecks_extractor.

    Args:
        - has_bottleneck: Tensor de tipo tf.bool. Indica si es que hay que ir a
        buscar el archivo a disco (en caso de ser True) o si es que se debe
        calcular "al voleo".
        - filename: Tensor de tipo tf.string. Ubicación donde se debería
        encontrar el bottleneck cacheado en disco.
        - image: Tensor de tipo tf.float32 y shape = [height, width, depth]
        acordes a lo esperado por bottleneck_extractor. Imagen sobre la cual
        se debe calcular el bottleneck.
        - bottleneck_extractor: tf.hub.Module. Módulo que se usará para extraer
        los bottlenecks.

    Returns:
        - bottleneck: Tensor de tipo tf.float32 y shape = (1, n_outputs), donde
        n_outputs es el número de valores producidos por bottleneck_extractor.
    """
    def _decode_bottleneck(filename):
        raw = tf.read_file(filename)
        splitted = tf.string_split([raw], ",")
        numeric = tf.string_to_number(splitted.values)
        return numeric

    def _run_bottleneck(image):
        feature = bottlenecks_extractor(tf.expand_dims(image, 0))
        squeezed = tf.squeeze(feature)
        return squeezed

    bottleneck = tf.cond(has_bottleneck,
                         lambda: _decode_bottleneck(filename),
                         lambda: _run_bottleneck(image))
    return bottleneck


def write_bottleneck(file_path, bottleneck):
    """Escribe un bottleneck a disco. Función diseñada para ser usada con
    tf.map_fn(...).

    Args:
        - file_path: Tensor de tipo tf.string. Ubicación donde se debe escribir
        el bottleneck.
        - bottleneck: Tensor de tipo tf.float32 y rank = 1. Bottleneck que debe
        ser serializado en disco.

    Returns:
        - file_path: El mismo file_path que fue entregado como argumento. Esto
        se hace porque la función debe retornar algo, aún cuando en este caso
        no sea necesario.
    """
    def _write_bottleneck(file_path, bottleneck):
        if tf.gfile.Exists(file_path):
            return file_path

        bottleneck_as_str_array = bottleneck.astype('str')
        bottleneck_as_str = ",".join(bottleneck_as_str_array)
        with tf.gfile.Open(file_path, "w") as bottleneck_file:
            bottleneck_file.write(bottleneck_as_str)
        return file_path

    return tf.py_func(_write_bottleneck, [file_path, bottleneck], tf.string)


def hub_bottleneck_model_fn(features, labels, mode, params):
    """Función para ser entregada como model_fn a un Estimator de TF.

    Este modelo utiliza como base una CNN previamente entrenada (en la forma
    de un hub.Module) y añade encima una capa de clasificación para el problema
    que se busca trabajar. Además, permite cachear bottlenecks en disco para
    disminuir considerablemente el tiempo de entrenamiento.

    Args:
        - features: dict[str: Tensor] Features en batch entregadas por el
        iterador del dataset. Se asume que tiene dos llaves: "filename", con un
        tensor asociado con los nombres de las imágenes (en formato
        label_original/filename.jpg), y otra llave llamada "image" con las
        imágenes correspondientes (correlativas a "filename")
        - labels: Tensor de tipo tf.int. Labels correspondientes a las features
        entregadas por el iterador del dataset.
        - mode: tf.estimator.ModeKeys, determina si se está en modo de
        entrenamiento, evaluación o prediccíón.
        - params: dict[str: any]. Diccionario con parámetros de configuración
        necesarios para el clasificador. En este caso, son necesarios los
        siguientes parámetros:
            - module_spec: tf.ModuleSpec. Especificaciones del modelo a
            entrenar
            - n_classes: int. Número de clases que se desea clasificar
            - cache_bottlenecks: bool, opcional. Indica si es que se desea o no
            cachear los bottlenecks en disco (default: True)
            - bottlenecks_dir: string, opcional pero requerido si es que
            cache_bottlenecks es True. Directorio donde se almacenarán los
            bottlenecks.

    Returns:
        - tf.estimator.EstimatorSpec, la configuración dependerá del valor de
        mode:
            - Si mode==PREDICT, se entregará un EstimatorSpec con predicciones,
            un diccionario con llaves "classes" y "probabilities" y tensores
            asociados correspondientes.
            - Si mode==TRAIN, se entregará un EstimatorSpec con la función de
            pérdida y el optimizador.
            - Si mode==EVAL, se entregará un EstimatorSpec con la función de
            pérdida y la métrica de evaluación (un diccionario con llave
            "accuracy" y el tensor asociado correspondiente)
    """

    # Cargamos el módulo
    module_spec = params["module_spec"]
    bottlenecks_extractor = hub.Module(module_spec)
    output_shape = module_spec.get_output_info_dict(
        signature=None)["default"].get_shape()

    # Features que serán utilizadas
    images = features["image"]
    n_classes = params["n_classes"]

    # Le asignamos un nombre para poder recuperar ese tensor
    images = tf.identity(images, "images")
    # Otros parámetros
    cache_bottlenecks = params[
        "cache_bottlenecks"] if "cache_bottlenecks" in params else True
    learning_rate = params["learning_rate"]
    if cache_bottlenecks:
        filenames = features["filename"]
        # Si es que se deben cachear los bottlenecks, se debe añadir una serie
        # de operaciones al grafo, que permitan leer y guardar en disco
        bottlenecks_dir = params["bottlenecks_dir"]
        # Obtenemos los nombres de los bottlenecks cacheados, sin importar
        # si es que existen o no aún
        bottlenecks_filenames = tf.map_fn(
            lambda fname: tf.string_join(
                [bottlenecks_dir, fname, ".txt"]),
            filenames, back_prop=False, name="get_bottlenecks_file_map")
        # Revisamos qué archivos ya tienen sus bottlenecks cacheados
        has_prev_bottlenecks = tf.map_fn(has_bottlenecks_map_fn,
                                         bottlenecks_filenames,
                                         dtype=tf.bool,
                                         back_prop=False,
                                         name="has_prev_bottlenecks_map")
        # Añadimos un tensor identidad con el nombre "has_prev_bottlenecks"
        # por si es que es requerido para ser loggeado
        has_prev_bottlenecks = tf.identity(
            has_prev_bottlenecks, "has_prev_bottlenecks")
        # Obtenemos los bottlenecks
        bottlenecks = tf.map_fn(
            lambda x: get_bottlenecks_map_fn(
                x[0], x[1], x[2], bottlenecks_extractor),
            [has_prev_bottlenecks, bottlenecks_filenames, images],
            dtype=tf.float32, back_prop=False, name="bottlenecks_map")
        # Tensor que se encarga de escribir los bottlenecks a disco
        write_bottlenecks = tf.map_fn(lambda x: write_bottleneck(x[0], x[1]),
                                      [bottlenecks_filenames, bottlenecks],
                                      dtype=tf.string, back_prop=False,
                                      name="write_map")
        # Otro tensor identidad, con el nombre "write_bottlenecks"
        write_bottlenecks = tf.identity(
            write_bottlenecks, "write_bottlenecks")
        # Añadimos una dependencia, para asegurarnos que
        # write_bottlenecks_tensor sea ejecutado
        with tf.control_dependencies([write_bottlenecks]):
            # Tensor identidad con el nombre "bottlenecks"
            bottlenecks = tf.identity(bottlenecks, "bottlenecks")
            # Con esto le aseguramos al compilador que bottlenecks tiene una
            # forma que puede ser usada en tf.layers.dense
            bottlenecks.set_shape(output_shape)

    else:
        # En caso de que no se desee cachear las imágenes, basta con
        # calcularlas en cada iteración.
        bottlenecks = bottlenecks_extractor(images)
        # Tensor identidad con nombre "bottlenecks", para ser recuperado
        # fácilmente por un logger
        bottlenecks = tf.identity(bottlenecks, "bottlenecks")

    logits = tf.layers.dense(inputs=bottlenecks, units=n_classes)
    probabilities = tf.nn.softmax(logits, name="softmax_tensor")
    classes = tf.argmax(input=logits, axis=1)

    tf.summary.histogram('activations', probabilities)
    tf.summary.histogram('classes', classes)

    predictions = {
        # Generamos predicciones (para modos PREDICT y EVAL)
        "classes": classes,
        # Añadimos al grafo un tensor softmax (con nombre "softmax_tensor")
        # Será utilizado en el modo PREDICT y además puede ser requerido para
        # ser loggeado
        "probabilities": probabilities
    }
    export_outputs = \
        {
            'predict_output':
            tf.estimator.export.PredictOutput
            (
                {
                    "pred_output_classes": classes,
                    'probabilities': probabilities
                }
            )
        }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions,
                                          export_outputs=export_outputs)
    predictions["real_classes"] = labels
    # Calculamos la función de pérdida (para modos TRAIN y EVAL)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss = tf.identity(loss, "loss")
    # Añadimos métricas de evaluación (para modo EVAL)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=classes)
    eval_metric_ops = {"accuracy": accuracy}
    # Nos aseguramos de agregarlas al reporte de TensorBoard
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy[0])

    # Configuramos las operaciones de entrenamiento (para el modo TRAIN)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op,
                                          eval_metric_ops=eval_metric_ops)

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


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


def get_parser():
    """Retorna un parser de línea de comandos con los parámetros que el
    usuario puede entregar mediante línea de comandos.

    Returns:
        argparse.ArgumentParser ya configurado con los argumentos que se
        pedirán por línea de comandos.
    """

    parser = argparse.ArgumentParser()

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
        default='',
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
        caja de recorte (crop box).Sólo tiene efecto en las imágenes de
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
        '--validation_batch_size',
        type=int,
        default=100,
        help="""\
        Cuántas imágenes evaluar en cada paso. Si es que el valor es -1, se
        utilizará todo el conjunto de validación para evaluar en cada paso.\
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

    # No usados aún
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=10,
        help='Cada cuánto se debe evaluar el entrenamiento.'
    )
    return parser


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
        self.validation_batch_size = flags.validation_batch_size
        self.num_epochs = flags.num_epochs
        self.learning_rate = flags.learning_rate

        # Otras variables importantes
        self.cache_bottlenecks = not dataset.should_distort_images(
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

        src_code_path = os.getcwd()
        root_path = os.path.abspath(os.path.join(src_code_path, ".."))
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
                root_path, "logs_and_checkpoints_dir", self.experiment_name)
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
        os.makedirs(self.bottlenecks_dir, exist_ok=True)
        os.makedirs(self.export_model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        for label_index in range(self.n_classes):
            os.makedirs(os.path.join(self.bottlenecks_dir, str(label_index)),
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

        if export:
            cache_bottlenecks = False
            warm_start_from = self.logs_and_checkpoints_dir
        else:
            cache_bottlenecks = self.cache_bottlenecks
            warm_start_from = None

        params = {"module_spec": self.module_spec,
                  "model_name": self.model_name,
                  "n_classes": self.n_classes,
                  "cache_bottlenecks": cache_bottlenecks,
                  "bottlenecks_dir": self.bottlenecks_dir,
                  "learning_rate": self.learning_rate}

        classifier = tf.estimator.Estimator(
            model_fn=hub_bottleneck_model_fn,
            model_dir=self.logs_and_checkpoints_dir,
            config=tf.estimator.RunConfig(
                tf_random_seed=self.random_seed,
                log_step_count_steps=1,
                save_summary_steps=2),
            params=params, warm_start_from=warm_start_from)

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
        return dataset.create_images_dataset(
            train_filenames, train_labels, image_shape=self.module_image_shape,
            image_depth=self.module_image_depth, src_dir=self.images_dir,
            shuffle=True, num_epochs=self.num_epochs,
            batch_size=self.train_batch_size,
            flip_left_right=self.flip_left_right,
            random_crop=self.random_crop, random_scale=self.random_scale,
            random_brightness=self.random_brightness)

    def __eval_input_fn(self, validation_filenames, validation_labels):
        """Construye un dataset para ser utilizado como input_fn en la fase de
        validación.

        La principal diferencia con __train_input_fn es que aquí no importan
        las distorsiones aleatorias, las cuales sí son utilizadas en el modo de
        entrenamiento. Además, esta función es para el modo de evaluación
        dentro del entrenamiento; es decir, con un conjunto de validación
        distinto al conjunto de pruebas; por ende, se aleatoriza y repite.

        Args:
            - validation_filenames: [str]. Nombres de los archivos que serán
            utilizados en la validación. Su formato es
            label_original/filename.jpg, y la ubicación es relativa a
            self.images_dir.
            - validation_labels: [int]. Etiquetas numéricas con las cuales se
            realizará la validación.

        Returns:
            tf.data.Dataset, con mapeo de filenames a imágenes y distorsiones
            aleatorias en caso de ser requeridas. Además, se le aplican las
            operaciones shuffle, repeat y batch.
        """
        return dataset.create_images_dataset(
            validation_filenames, validation_labels,
            image_shape=self.module_image_shape,
            image_depth=self.module_image_depth,
            src_dir=self.images_dir, shuffle=True, num_epochs=self.num_epochs,
            batch_size=self.validation_batch_size)

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

        Returns:
            tf.data.Dataset, con mapeo de filenames a imágenes y distorsiones
            aleatorias en caso de ser requeridas. Además, se le aplican las
            operaciones shuffle, repeat y batch.
        """
        return dataset.create_images_dataset(
            test_filenames, test_labels, image_shape=self.module_image_shape,
            image_depth=self.module_image_depth,
            src_dir=self.images_dir, shuffle=False, num_epochs=1,
            batch_size=100)

    def train(self, train_filenames, train_labels, hooks=None):
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
        self.estimator.train(input_fn=train_input_fn,
                             steps=self.num_epochs, hooks=hooks)

    def test(self, validation_filenames, validation_labels, hooks=None,
             final_test=True):
        """Evalúa el tf.Estimator

        Primero construye la input_fn con que se alimentará el modelo, y luego
        lo evalúa.

        Args:
            - validation_filenames: [str]. Nombres de los archivos que serán
            utilizados en la evaluación. Su formato es
            label_original/filename.jpg, y la ubicación es relativa a
            self.images_dir.
            - validation_labels: [int]. Etiquetas numéricas con las cuales se
            realizará la evaluación.
            - hooks: [SessionRunHooks]. Lista de hooks que deben ser ejecutados
            durante la evaluación.
        """
        test_input_fn = lambda: self.__test_input_fn(validation_filenames,
                                                     validation_labels)

        eval_results = self.estimator.evaluate(
            input_fn=test_input_fn, hooks=hooks)
        print(eval_results)

        if final_test:
            predictions = self.estimator.predict(
                input_fn=test_input_fn, hooks=hooks)
            header = "Filename / Real Class / Predicted Class\n"
            lines = [header]
            for index, pred in enumerate(predictions):
                filename = validation_filenames[index]
                real_class = validation_labels[index]
                predicted_class = pred['classes']
                line = "{fn} {real} {predicted}\n".format(
                    fn=filename, real=real_class, predicted=predicted_class)
                lines.append(line)
            output = os.path.join(self.results_dir, "predictions.txt")
            with open(output, "w") as file:
                file.writelines(lines)

    def train_and_evaluate(self, train_filenames, train_labels,
                           validation_filenames, validation_labels,
                           train_hooks=None, eval_hooks=None):
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

        train_input_fn = lambda: self.__train_input_fn(
            train_filenames, train_labels)
        eval_input_fn = lambda: self.__eval_input_fn(
            validation_filenames, validation_labels)

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                            max_steps=self.num_epochs,
                                            hooks=train_hooks)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                          hooks=eval_hooks)

        tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)

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

        map_fn = lambda img: dataset.decode_image_from_string(
            img, self.module_image_shape, self.module_image_depth)

        features['image'] = tf.map_fn(
            map_fn, features['image'], dtype=tf.float32)

        return tf.estimator.export.ServingInputReceiver(features,
                                                        received_tensors)

    def export_graph(self):
        """Exporta el grafo como un SavedModel estándar para ser utilizado
        posteriormente
        """
        estimator_for_export = self.__build_estimator(export=True)
        estimator_for_export.export_savedmodel(
            self.export_model_dir, self.__serving_input_receiver_fn,
            as_text=True)

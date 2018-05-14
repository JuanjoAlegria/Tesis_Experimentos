"""Módulo para crear un grafo de TensorFlow, en la forma de tf.Estimator,
    utilizando como base un módulo del Hub de Tensorflow. Lo que se hace es
    añadir una nueva capa final de clasificación a una CNN (Convolutional
    Neural Network) ya entrenada. Además, se pueden cachear los bottlenecks,
    para evitar recalcularlos en cada iteración y así reducir el tiempo de
    entrenamiento.
"""

import tensorflow as tf
import tensorflow_hub as hub


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
        buscar el archivo a disco(en caso de ser True) o si es que se debe
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

    Este modelo utiliza como base una CNN previamente entrenada(en la forma
    de un hub.Module) y añade encima una capa de clasificación para el problema
    que se busca trabajar. Además, permite cachear bottlenecks en disco para
    disminuir considerablemente el tiempo de entrenamiento.

    Args:
        - features: dict[str: Tensor] Features en batch entregadas por el
        iterador del dataset. Se asume que tiene dos llaves: "filename", con un
        tensor asociado con los nombres de las imágenes(en formato
        label_original / filename.jpg), y otra llave llamada "image" con las
        imágenes correspondientes(correlativas a "filename")
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
            cachear los bottlenecks en disco(default: True)
            - bottlenecks_dir: string, opcional pero requerido si es que
            cache_bottlenecks es True. Directorio donde se almacenarán los
            bottlenecks.

    Returns:
        - tf.estimator.EstimatorSpec, la configuración dependerá del valor de
        mode:
            - Si mode == PREDICT, se entregará un EstimatorSpec con predicciones,
            un diccionario con llaves "classes" y "probabilities" y tensores
            asociados correspondientes.
            - Si mode == TRAIN, se entregará un EstimatorSpec con la función de
            pérdida y el optimizador.
            - Si mode == EVAL, se entregará un EstimatorSpec con la función de
            pérdida y la métrica de evaluación(un diccionario con llave
            "accuracy" y el tensor asociado correspondiente)
    """

    # Cargamos el módulo
    module_spec = params["module_spec"]
    # Al cargar el módulo se imprimen demasiadas cosas en pantalla,
    # así que subimos momentáneamente el logging a WARN
    tf.logging.set_verbosity(tf.logging.WARN)
    bottlenecks_extractor = hub.Module(module_spec)
    # Volvemos a activar el logging INFO
    tf.logging.set_verbosity(tf.logging.INFO)
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
        filenames = tf.identity(filenames, "filenames")
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
    accuracy = tf.metrics.accuracy(labels=labels, predictions=classes,
                                   name="accuracy")

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

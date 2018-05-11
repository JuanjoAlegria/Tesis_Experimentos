"""Utilidades para trabajar con datasets.
"""
import os
import numpy as np
import tensorflow as tf


def get_filenames_and_labels(data_dir, partition,
                             labels_to_indexes=False, labels_map=None,
                             max_files=-1):
    """Obtiene los nombres de los archivos en un directorio y las etiquetas
    correspondientes.

    Se asume que el directorio tendrá el siguiente formato:
        data_dir/
            label1/
                train_image1.jpg
                train_image5.jpg
                test_image12.jpg
                ...
            label2/
                train_image9.jpg
                test_image51.jpg
                ...
            ...
            labeln/
                ....

    Luego, para cada label en data_dir, se extraen las imagenes que tengan
    {partition} en el nombre (donde partition puede ser train o test).

    Además, si es que las etiquetas no son numéricas (e.g, setosa, versicolor,
    virginica), esta función se encarga de transformar las etiquetas de strings
    a números y entrega un diccionario con el mapeo correspondiente.

    Args:
        - data_dir: string. Directorio donde buscar las imágenes
        partition: string. Indica qué tipo de imágenes queremos buscar
        (test o train).
        - labels_to_indexes: boolean. Indica si es que se debe realizar la
        conversión desde etiquetas como strings a números. En caso de ser
        False, se asume que las etiquetas son enteros en el rango
        [0, n_classes - 1].
        - labels_map: dict[str: int]. Si es que el mapeo ya se realizó
        previamente, se puede pasar como parámetro y no será recalculado, lo
        cual es necesario para mantener consistencia entre los datasets de
        entrenamiento, validación y prueba.
        - max_files: int. Cantidad máxima de archivos a considerar;
        útil para depuración.

    Returns:
        - filenames: list[str]. Lista con los nombres de los archivos en
        formato label/filename.jpg. Notar que, en este caso, label corresponde
        a la etiqueta sin transformar (e.g, acá label si puede ser setosa o
        versicolor).
        - labels: list[int]. Lista con las etiquetas correspondientes
        convertidas a enteros. Correlativo con filenames.
        - labels_map: dict[str: int]. Diccionario con el mapeo entre etiquetas
        originales y etiquetas numéricas.

    """
    filenames = []
    labels = []

    if not labels_map:
        labels_map = {}
        # Si es que hay que transformar las etiquetas a índices, pero no hay un
        # mapa de etiquetas, entonces hay que construirlo
        if labels_to_indexes:
            index = 0
            for label in os.listdir(data_dir):
                labels_map[label] = index
                index += 1

        # Por otro lado, en caso de que no haya que transformar las etiquetas a
        # índices, se asume que la etiqueta es numérica y por consistencia
        # creamos un label_map donde para todo x, label_map[x] = x
        else:  # i.e, not labels_to_indexes
            for label in os.listdir(data_dir):
                labels_map[label] = int(label)

    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        for img_name in os.listdir(label_path):
            if partition in img_name:
                image_relative_path = os.path.join(label, img_name)
                filenames.append(image_relative_path)
                labels.append(labels_map[label])

    filenames, labels = np.array(filenames), np.array(labels)
    if max_files > -1:
        filenames = np.random.choice(filenames, max_files, replace=False)
        labels = np.random.choice(labels, max_files, replace=False)

    return filenames, labels, labels_map


def should_distort_images(flip_left_right, random_crop, random_scale,
                          random_brightness):
    """Indica si es que se debe o no aplicar distorsiones a la imagen.

    Args:
      - flip_left_right: bool. Indica si es que se debe rotar horizontalmente
      la imagen.
      - random_crop: int. Porcentaje que indica el margen total a utilizar
      alrededor de la caja de recorte (crop box).
       - random_scale: int. Porcentaje que indica el rango de cuánto se debe
       variar la escala de la imagen
      - random_brightness: int. Rango por el cual multiplicar aleatoriamente
      los pixeles de la imagen.

    Returns:
      bool. Indica si es que se debe o no aplicar distorsiones a la imagen
    """
    return (flip_left_right or (random_crop != 0) or (random_scale != 0) or
            (random_brightness != 0))


def decode_image_from_string(image_string, image_shape, image_depth):
    """ Decodifica una representación de una imagen jpeg como string a un
    tensor tridimensional

    Args:
        - image_string: Tensor de tipo str. Representación de la imagen como
        string.
        - image_shape: [int, int]. Altura y ancho de la imagen.
        - image_depth: int. Cantidad de canales en la imagen.

    Returns:
        Tensor con forma [image_shape[0], image_shape[1], image_depth] que
        representa la imagen.
    """
    decoded_image = tf.image.decode_jpeg(image_string, channels=image_depth)
    as_float_image = tf.image.convert_image_dtype(decoded_image, tf.float32)
    resized_image = tf.image.resize_images(as_float_image, image_shape)
    return resized_image


def create_images_dataset(filenames, labels, image_shape, image_depth, src_dir,
                          shuffle=True, num_epochs=None, batch_size=1,
                          flip_left_right=False, random_crop=0, random_scale=0,
                          random_brightness=0):
    """Crea un dataset de Tensorflow a partir de la lista de imágenes y sus
    etiquetas.

    Se asume que la lista de nombres de archivo está en el formato
    label/filename.jpg y que el archivo es relativo a src_dir (i.e., el archivo
    está ubicado en src_dir/label/filename.jpg). Luego, se retorna un dataset
    donde cada elemento tiene dos componentes: features y labels, donde
    features es un diccionario con también dos componentes: filename e image.

    Además, el dataset puede ser aleatorizado, repetido y convertido a batch

    Args:
        - filenames: list[str]. Lista con los nombres de los archivos que serán
        parte del dataset. Se asume que el formato del archivo es
        label/filename.jpg.
        - labels: list[int]. Lista con etiquetas numéricas, correlativas a
        filenames.
        - image_shape: [int, int]. Altura y ancho que deben tener las imágenes.
        - image_depth: int. Número de canales de la imagen (3 para RGB, 1 para
        escala de grises).
        - src_dir: string. Directorio raíz donde están ubicadas las imágenes.
        - shuffle: boolean. Si es True, se aleatoriza el dataset en cada
        iteración.
        - num_epochs: int o None. Número de veces que se repetirá el dataset.
        Si es que es None, el dataset se repetirá infinitamente.
        - batch_size: int. Tamaño del batch.
        - flip_left_right: bool. Indica si es que se debe rotar horizontalmente
        la imagen.
        - random_crop: int. Porcentaje que indica el margen total a utilizar
        alrededor de la caja de recorte (crop box).
        - random_scale: int. Porcentaje que indica el rango de cuánto se debe
        variar la escala de la imagen.
        - random_brightness: int. Rango por el cual multiplicar aleatoriamente
        los pixeles de la imagen.

    """
    def _load_image(feature, label):
        file_path = tf.string_join([src_dir, feature["filename"]])
        image_string = tf.read_file(file_path)
        processed_image = decode_image_from_string(image_string, image_shape,
                                                   image_depth)
        return {"filename": feature["filename"],
                "image": processed_image}, label

    def _random_distortions(feature, label):
        image = feature["image"]
        margin_scale = 1.0 + (random_crop / 100.0)
        resize_scale = 1.0 + (random_scale / 100.0)
        margin_scale_value = tf.constant(margin_scale)
        resize_scale_value = tf.random_uniform(shape=[],
                                               minval=1.0,
                                               maxval=resize_scale)

        scale_value = tf.multiply(margin_scale_value, resize_scale_value)
        precrop_height = tf.multiply(scale_value, image_shape[0])
        precrop_width = tf.multiply(scale_value, image_shape[1])
        precrop_shape = tf.stack([precrop_height, precrop_width])
        precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)

        precropped_image = tf.image.resize_bilinear(image,
                                                    precrop_shape_as_int)
        cropped_image = tf.random_crop(
            precropped_image, [image_shape[0], image_shape[1], image_depth])
        if flip_left_right:
            flipped_image = tf.image.random_flip_left_right(cropped_image)
        else:
            flipped_image = cropped_image
        brightness_min = 1.0 - (random_brightness / 100.0)
        brightness_max = 1.0 + (random_brightness / 100.0)
        brightness_value = tf.random_uniform(shape=[],
                                             minval=brightness_min,
                                             maxval=brightness_max)
        brightened_image = tf.multiply(
            flipped_image, brightness_value, name='DistortResult')

        return {"filename": feature["filename"],
                "image": brightened_image}, label

    n_images = len(filenames)
    dataset = tf.data.Dataset.from_tensor_slices(
        ({"filename": filenames}, labels))
    # Aleatorizamos el dataset
    if shuffle:
        dataset = dataset.shuffle(n_images, reshuffle_each_iteration=True)
    # Repetimos el dataset (si num_epochs == None, se repetirá indefinidamente)
    dataset = dataset.repeat(num_epochs)
    # Mapeamos y obtenemos las imágenes
    dataset = dataset.map(_load_image)
    if should_distort_images(flip_left_right, random_crop,
                             random_scale, random_brightness):
        dataset = dataset.map(_random_distortions)
    # Batch
    dataset = dataset.batch(batch_size)
    return dataset


def create_val_set_from_train(filenames, labels, val_proportion):
    """ Crea un conjunto de validación a partir del conjunto de entrenamiento.

    Args:
        - filenames: list[str]. Lista con los nombres de los archivos que son
        parte del conjunto de entramiento.
        - labels: list[int]. Lista con etiquetas numéricas, correlativas a
        filenames.
        - val_proportion: int, 0 <  val_proportion <= 100. Porcentaje del
        conjunto de entrenamiento que debe usarse para contruir el conjunto de
        validación.
    Returns:
        (train_filenames, train_labels), (val_filenames, val_labels)
        Dos tuplas, una para el conjunto de entrenamiento y otra para el de
        validación. Cada tupla tiene dos elementos: el primer elemento es una
        lista de nombres de archivo y el segundo elemento es una lista con las
        etiquetas.
    """
    if val_proportion <= 0 or val_proportion > 100:
        error_msg = "val_proportion debe estar entre 0 y 100" + \
            " y el valor entregado fue {value}".format(value=val_proportion)
        raise ValueError(error_msg)

    n_val = int(len(filenames) * (val_proportion / 100))
    permutation = np.random.permutation(len(filenames))

    filenames, labels = filenames[permutation], labels[permutation]

    train_filenames = filenames[n_val:]
    train_labels = filenames[n_val:]

    val_filenames = filenames[:n_val]
    val_labels = labels[:n_val]

    return (train_filenames, train_labels), (val_filenames, val_labels)


def write_labels_map(labels_map, filename):
    """ Escribe a archivo el mapeo de etiquetas originales a etiquetas
    numéricas.

    Args:
        - labels_map: dict[str: int]. Mapeo de etiquetas originales a etiquetas
        numéricas (índices en el rango [0, n_classes -1]).
        - filename: str. Ubicación donde se guardará el mapeo.
    Returns:
        None
    """
    header = "Etiqueta original / Etiqueta numérica\n"
    lines = [header]
    lines += ["{key} : {value}\n".format(key=key, value=value)
              for key, value in labels_map.items()]
    with open(filename, "w") as file:
        file.writelines(lines)

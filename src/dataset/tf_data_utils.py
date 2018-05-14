"""Utilidades para crear datasets de Tensorflow"""

import tensorflow as tf


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

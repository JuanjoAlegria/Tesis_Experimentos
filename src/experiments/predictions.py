"""Módulo con funciones para que una red de TensorFlow ya entrenada sea
utilizada como predictor.
"""

import tensorflow as tf
from tensorflow.contrib import predictor
from ..dataset import tf_data_utils


def open_images_for_prediction(filenames, img_height, img_width, img_depth):
    """Abre una lista de imágenes jpg, decodificándolas y redimensionándolas
    para ser entregadas a un served_model.

    Las operaciones apliIcadas son:
        - Leer la imagen como un string
        - Decodificar a jpeg
        - Convertir a float
        - Redimensionar a (img_height, img_width, img_depth).
    Todas esas operaciones son aplicadas con funciones de tensorflow, para
    mantener coherencia con las operaciones realizadas a las imágenes con que
    fue entrenado el modelo.

    Args:
        - filenames: list[str]. Lista con las rutas de las imágenes que se
        desea abrir.
        - img_height: int. Altura que deben tener las imágenes al ser
        ingresadas a la red.
       - img_width: int. Ancho que deben tener las imágenes al ser
        ingresadas a la red.
        - img_depth: int. Profundidad que deben tener las imágenes al ser
        ingresadas a la red. Generalmente, img_depth = 3 (para imágenes RGB).

    Returns:
        np.array(len(filenames), img_height, img_width, img_depth). Arreglo de
        numpy con todas las imágenes decodificadas.
    """
    tensor_list = []
    for filename in filenames:
        img_string_tensor = tf.read_file(filename)
        resized_img_tensor = tf_data_utils.decode_image_from_string(
            img_string_tensor, (img_height, img_width), img_depth)
        tensor_list.append(resized_img_tensor)
    tensor_stack = tf.stack(tensor_list)
    with tf.Session() as sess:
        images = sess.run(tensor_stack)
    return images


def get_predictions(saved_model_dir, filepaths, batch_size=100):
    """Calcula las predicciones de un modelo para una lista de archivos
    (específicamente, imágenes).

    Args:
        - saved_model_dir: str. Directorio donde se encuentra el modelo
        guardado por TensorFlow. Preferentemente, debe ser un tf.Estimator que
        fue exportado allí.
        - filepaths: list[str]. Lista con las rutas de las imágenes cuya clase
        se desea predecir.
        - batch_size: int. Número de imágenes que le serán entregadas al mismo
        tiempo a la red para predecir.
    """
    predicted_classes = []
    predict_fn = predictor.from_saved_model(saved_model_dir)
    _, height, width, depth = predict_fn.feed_tensors['image'].shape
    height, width, depth = int(height), int(width), int(depth)
    for idx in range(0, len(filepaths), batch_size):
        filepaths_batch = filepaths[idx: idx + batch_size]
        images_batch = open_images_for_prediction(
            filepaths_batch, height, width, depth)
        predictions_batch = predict_fn({"image": images_batch})
        predicted_classes += predictions_batch["pred_output_classes"].tolist()
    return predicted_classes

"""Módulo con utilidades para trabajar con los archivos de salida generados por
la red neuronal.
"""
REAL_CLASS_INDEX = 0
PREDICTED_CLASS_INDEX = 1


def write_output(filenames, predictions, real_labels=None,
                 output_file="output.txt"):
    """Escribe las predicciones realizadas por la red a un archivo de texto.
    Si es que real_labels es None, la columna 'Real Class' será rellenada con
    -1.

    Args:
        - filename: list[str]. Nombres de los archivos.
        - predictions: list[int]. Predicciones realizadas por la red.
        - real_labels: list[int]. Etiquetas reales de los archivos.
        - output_file: str. Ubicación donde se desea guardar las predicciones.

    """
    header = "Filename / Real Class / Predicted Class\n"
    lines = [header]
    if real_labels is None:
        real_labels = [-1] * len(predictions)
    for filename, real_label, prediction in \
            zip(filenames, real_labels, predictions):
        line = "{fn} {real} {predicted}\n".format(
            fn=filename, real=real_label, predicted=prediction)
        lines.append(line)
    with open(output_file, "w") as file:
        file.writelines(lines)


def get_all_slides_ids(images_dict):
    """Obtiene todas las ids de slides en un diccionario de imágenes (como
    el de proporciones de tejido o el de resultados de la red).

    Args:
        - images_dict: dict[str -> any]. Diccionario con nombres de imágenes
        en formato label/{slide_id}_rest_of_image_name, sin importar el valor
        asociado.

    Returns:
        set[str], con todas las ids únicas de slides encontradas en el
        diccionario.
    """
    slides_ids = set()
    for label_and_name in images_dict:
        _, name = label_and_name.split("/")
        slide_id = name.split("_")[0]
        if slide_id not in slides_ids:
            slides_ids.add(slide_id)
    return slides_ids


def transform_to_dict(output_lines):
    """Transforma una lista con líneas de predicciones realizadas por la red
    a un diccionario con el nombre de la imagen como llave y valores asociados
    la clase real de la imagen y la clase predicha por la red.

    output_lines es una lista donde cada elemento es un string de la forma:
    "{label}/{slide_id}_rest.jpg {real_class} {predicted_class}", excepto el
    primer elemento, que corresponde al header.

    Args:
        - output_lines: list[str]. Lista con los nombres de archivos,
        clase real y clase predicha.

    Returns:
        dict[str -> (str, str)]. Diccionario con el nombre de la imagen como
        llave, y con valor asociado una tupla con la clase real de la imagen
        en primera posición y la clase predicha en segunda posición.

    """
    output_dict = {}
    for line in output_lines[1:]:
        label_name, real, predicted = line[:-1].split()
        classes = (real, predicted)
        output_dict[label_name] = classes
    return output_dict


def transform_to_lists(output_lines):
    """Transforma una lista con líneas de predicciones realizadas por la red
    a tres listas: la primera con los nombres de las imágenes, la segunda con
    las clases reales y la tercera con las clases predichas por la red.

    output_lines es una lista donde cada elemento es un string de la forma:
    "{label}/{slide_id}_rest.jpg {real_class} {predicted_class}", excepto el
    primer elemento, que corresponde al header.

    Args:
        - output_lines: list[str]. Lista con los nombres de archivos,
        clase real y clase predicha.

    Returns:
        list(str), list(str), list(str). La primera lista contiene los nombres
        de las imágenes, la segunda lista contiene las clases reales y la
        tercera contiene las clases predichas por la red.

    """
    filenames = []
    real_classes = []
    pred_classes = []
    for line in output_lines[1:]:
        name, real, predicted = line[:-1].split()
        filenames.append(name)
        real_classes.append(real)
        pred_classes.append(predicted)
    return filenames, real_classes, pred_classes


def get_rois_ids(slide_id, output_dict):
    """Obtiene todas las ids de ROIs almacenadas en un diccionario de
    predicciones, para determinada slide.

    Se asume que cada llave del diccionario es de la forma:
    {label}/{slide_id}_{magnification}_{zlevel}_{roi_id}_{i_row}_{j_col}.jpg.

    Args:
        - slide_id: str. Id de la slide cuyos rois se desea encontrar.
        - output_dict: dict[str -> (str, str)]. Diccionario con el nombre
        de la imagen como llave, y con valor asociado una tupla con la clase
        real de la imagen en primera posición y la clase predicha en segunda
        posición.
    Returns:
        list[str], con las ids de los rois extraídos desde slide_id.
    """
    rois_ids = set()
    for key in output_dict:
        _, name = key.split("/")
        name = name.split(".")[0]
        current_slide_id, _, _, current_roi_id, *_ = name.split("_")
        if current_slide_id == slide_id:
            rois_ids.add(current_roi_id)
    return list(rois_ids)


def get_coords_and_preds(slide_id, output_dict, roi_id=None):
    """Obtiene las coordenadas de los parches evaluados y las predicciones
    correspondientes realizadas por un algoritmo de ML para una slide
    determinada.

    Args:
        - slide_id: str. Id de la slide.
        - output_dict: dict[str -> (str, str)]. Diccionario con el nombre
        de la imagen como llave, y con valor asociado una tupla con la clase
        real de la imagen en primera posición y la clase predicha en segunda
        posición.
        - roi_id: str. Si es que se quieren obtener las coordenadas
        correspondientes a sólo un ROI, se debe pasar la id de este.

    Returns:
        - x_coords: list[int]. Lista con las coordenadas x de los parches
        de la slide pedida que fueron evaluados por el algoritmo.
        - y_coords: list[int]. Lista con las coordenadas y de los parches
        de la slide pedida que fueron evaluados por el algoritmo. Correlativa
        a x_coords.
        - preds: list[str]. Predicciones realizadas por la red para cada uno de
        los parches (correlativo a x_coords e y_coords).
    """
    x_coords = []
    y_coords = []
    preds = []
    for key in output_dict:
        _, name = key.split("/")
        name = name.split(".")[0]
        if roi_id is not None:
            # en las imágenes extraídas desde rois, el formato es i,j
            current_slide_id, *_, current_roi_id,\
                y_coord, x_coord = name.split("_")
        else:
            current_slide_id, *_, \
                x_coord, y_coord = name.split("_")
        if current_slide_id == slide_id and \
                (roi_id is None or current_roi_id == roi_id):
            x_coords.append(int(x_coord[1:]))
            y_coords.append(int(y_coord[1:]))
            preds.append(output_dict[key][PREDICTED_CLASS_INDEX])
    return x_coords, y_coords, preds

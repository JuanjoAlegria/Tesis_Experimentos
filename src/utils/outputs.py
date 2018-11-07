"""Módulo con utilidades para trabajar con los archivos de salida generados por
la red neuronal.
"""
import os

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
    for image_name in images_dict:
        properties = get_properties_from_image_name(image_name)
        slides_ids.add(properties['biopsy_id'])
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
        properties = get_properties_from_image_name(key)
        if properties['biopsy_id'] == slide_id:
            rois_ids.add(properties['roi_id'])
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
        correspondientes a sólo un ROI, se debe pasar la id de este. En caso
        contrario, el valor será None y se obtendrán las coordenadas y
        predicciones de toda la slide.

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
        properties = get_properties_from_image_name(key)
        if properties['biopsy_id'] == slide_id and \
                (roi_id is None or properties['roi_id'] == roi_id):
            x_coords.append(properties['x_coord'])
            y_coords.append(properties['y_coord'])
            preds.append(output_dict[key][PREDICTED_CLASS_INDEX])
    return x_coords, y_coords, preds


def get_properties_from_image_name(image_name):
    """Obtiene las propiedades de una imagen a partir de su nombre.

    Args:
        - image_name: str. Nombre de la imagen. El formato puede ser
        {label}/{biopsy_id}_{magn}_{zlevel}_x{x_coord}_y{y_coord}.jpg,
        en caso de ser un parche sacado directamente de la slide, o
        {label}/{biopsy_id}_{magn}_{roi_id}_{zlevel}_i{y_coord}_j{x_coord}.jpg,
        en caso de ser un parche sacado de un roi.

    Returns:
        dict[str->str|int|None], con llaves label, biopsy_id, magnification,
        zlevel, roi_id, x_coord e y_coord.
    """
    label, name_and_ext = image_name.split("/")
    name, _ = os.path.splitext(name_and_ext)
    properties = name.split("_")
    if len(properties) == 5:
        # tiene formato (x, y)
        biopsy_id, magnification, zlevel, x_coord, y_coord = properties
        x_coord, y_coord = int(x_coord[1:]), int(y_coord[1:])
        roi_id = None
    else:
        # tiene formato (i, j), por lo que hay que invertir las coordenadas
        biopsy_id, magnification, zlevel, roi_id, y_coord, x_coord = properties
        x_coord, y_coord = int(x_coord[1:]), int(y_coord[1:])

    return {"label": label, "biopsy_id": biopsy_id,
            "magnification": magnification,
            "zlevel": zlevel, "roi_id": roi_id,
            "x_coord": x_coord, "y_coord": y_coord}


def get_slides_aggregated_results(images_names, images_predicted_classes):
    """Genera un diccionario con los resultados agregados de cada slide.

    images_names es una lista donde cada elemento es un string de la
    forma: "{label}/{slide_id}_rest.jpg", e images_predicted_classes es
    correlativa a images_names. Luego, para cada slide única se cuenta la
    cantidad de imágenes predichas como clase 0,1 o 2, y se genera un
    diccionario con esa información.

    Args:
        - images_names: list[str]. Lista con los nombres de las imágenes, en la
        forma {label}/{slide_id}_rest.jpg.
        - images_predicted_classes: list[str]. Lista con las clases predichas
        por el algoritmo, correlativa a images_names.

    Returns:
        dict[str->dict[str->int]]. El diccionario exterior tiene como llaves
        las ids de las slides y como valor asociado un nuevo diccionario. Este
        diccionario interior tiene cuatro llaves ("0", "1", "2" y "total"), y
        como valor asociado está la cantidad de imágenes que fueron
        clasificadas con dichas etiquetas (excepto en el caso de "total", que
        representa la suma de todas los parches de esa slide)

        Ejemplo: {"116": {"0": 2500, "1": 500, "2": 3000, "total": 6000},
                  "4": {"0": 4890, "1": 560, "2": 300, "total": 5750}}
    """
    slides_results = {}
    for image_name, pred in zip(images_names, images_predicted_classes):
        properties = get_properties_from_image_name(image_name)
        if properties['biopsy_id'] not in slides_results:
            slides_results[properties['biopsy_id']] = \
                {"0": 0, "1": 0, "2": 0, "total": 0}
        slides_results[properties['biopsy_id']][pred] += 1
        slides_results[properties['biopsy_id']]["total"] += 1
    return slides_results


def filter_control_patches(images_names, biopsies_min_x_coords):
    """Dada una lista de imágenes, elimina todos aquellos parches que
    pertenecen a tejido control de la biopsia.

    Args:
        - images_names: list[str]. Lista con los nombres de las imágenes, en la
        forma {label}/{slide_id}_rest.jpg.
        - biopsies_min_x_coords: dict[str-> int]. Diccionario con las mínimas
        coordenadas en el eje x que debe tener un parche para ser considerado
        como parte de la biopsia y no ser parte del tejido control.
    """
    filtered_images_names = []
    for image_name in images_names:
        properties = get_properties_from_image_name(image_name)
        min_x_coord = biopsies_min_x_coords[properties['biopsy_id']]
        if properties['x_coord'] > min_x_coord:
            filtered_images_names.append(image_name)
    return filtered_images_names

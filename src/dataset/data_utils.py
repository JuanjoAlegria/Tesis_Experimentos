"""Utilidades para crear un dataset desde archivos en disco.
"""
import os
import json
import itertools
import numpy as np

FULL_DATASET = "all"


def shuffle_dataset(features, labels):
    """Reordena aleatoriamente un dataset.

    Args:
        features: np.array, arreglo con features
        labels: np.array, arreglo con labels, correlativas a features.

    Returns:
        np.array, np.array: arreglos reordenados aleatoriamente
    """
    permutation = np.random.permutation(len(features))
    shuffled_features = features[permutation]
    shuffled_labels = labels[permutation]
    return shuffled_features, shuffled_labels


def get_filenames_and_labels(data_dir, partition=FULL_DATASET,
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

    Para asegurar reprdocubilidad entre distintas máquinas, los arreglos
    filenames y labels son ordenados antes de ser retornados.

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
        - filenames: np.array(str). Arreglo de numpy con los nombres de los
        archivos en formato label/filename.jpg. Notar que, en este caso, label
        corresponde a la etiqueta sin transformar (e.g, acá label si puede ser
        setosa o versicolor).
        - labels: np.array(str). Arreglo de numpy con las etiquetas
        correspondientes convertidas a enteros. Correlativo con filenames.
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
            if partition == FULL_DATASET or partition in img_name:
                image_relative_path = os.path.join(label, img_name)
                filenames.append(image_relative_path)
                labels.append(labels_map[label])

    filenames, labels = np.array(filenames), np.array(labels)
    # Ordenamos los arreglos
    argsort_permutation = np.argsort(filenames)
    filenames = filenames[argsort_permutation]
    labels = labels[argsort_permutation]

    if max_files > -1:
        permutation = np.random.choice(len(filenames), size=max_files,
                                       replace=False)
        filenames = filenames[permutation]
        labels = labels[permutation]

    return filenames, labels, labels_map


def generate_partition(features, labels, percentages):
    """Particiona un conjunto en n subconjuntos.

    Args:
        - features: np.array. Lista con features a utilizar.
        - labels: np.array. Lista con etiquetas, correlativas a features.
        - percentages: list[int]. Porcentajes a utilizar para particionar el
        conjunto. La suma de percentages debe ser igual a 100.

    Returns:
        list[tuples]. N tuplas, donde N = len(percentages). Particiones
        obtenidas.
    """
    if sum(percentages) != 100:
        error_msg = "La suma de las porcentajes debe ser igual a 100," + \
            "pero la suma calculada fue {value}".format(value=sum(percentages))
        raise ValueError(error_msg)
    permutation = np.random.permutation(len(features))
    features, labels = features[permutation], labels[permutation]
    result = []
    prev_index = 0
    for index, percentage in enumerate(percentages):
        if index == len(percentages) - 1:
            current_index = len(features)
        else:
            n_files = int(len(features) * (percentage / 100))
            current_index = prev_index + n_files
        current_features = features[prev_index: current_index]
        current_labels = labels[prev_index: current_index]
        result.append((current_features, current_labels))
        prev_index = current_index
    return result


def generate_validation_set(train_filenames, train_labels, percentage):
    """ Crea un conjunto de validación a partir del conjunto de entrenamiento.

    Args:
        - train_filenames: np.array(). Arreglo de numpy con los nombres de los
        archivos que son parte del conjunto de entramiento.
        - train_labels: np.array(). Arreglo de numpy con etiquetas numéricas,
        correlativas a train_filenames.
        - percentage: int, 0 < percentage <= 100. Porcentaje del
        conjunto de entrenamiento que debe usarse para contruir el conjunto de
        validación.
    Returns:
        (train_filenames, train_labels), (val_filenames, val_labels)
        Dos tuplas, una para el conjunto de entrenamiento y otra para el de
        validación. Cada tupla tiene dos elementos: el primer elemento es un
        arrego de numpy con nombres de archivo y el segundo elemento es un
        arrego de numpy con las etiquetas.
    """
    if percentage <= 0 or percentage > 100:
        error_msg = "percentage debe estar entre 0 y 100" + \
            " y el valor entregado fue {value}".format(value=percentage)
        raise ValueError(error_msg)

    n_files = int(len(train_filenames) * (percentage / 100))
    permutation = np.random.permutation(len(train_filenames))

    train_filenames = train_filenames[permutation]
    train_labels = train_labels[permutation]

    validation_filenames = train_filenames[:n_files]
    validation_labels = train_labels[:n_files]

    train_filenames = train_filenames[n_files:]
    train_labels = train_labels[n_files:]

    return (train_filenames, train_labels), \
        (validation_filenames, validation_labels)


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


def dump_dataset(dataset_path, train_features, train_labels,
                 validation_features, validation_labels,
                 test_features, test_labels):
    """Escribe a disco un archivo json con los nombres de los archivos
    correspondientes a cada partición y sus etiquetas correspondientes.

    El formato del archivo producido es:
    {
        "train_features": train_features
        "train_labels": train_labels,
        "validation_features": validation_features
        "validation_labels": validation_labels,
        "test_features": test_features
        "test_labels": test_labelss
    },
    donde cada valor corresponde a una lista.

    Args:
        - dataset_path: str. Ubicación donde se guardará el dataset como
        diccionario json
        - train_features: list[str]. Archivos del conjunto de entrenamiento.
        - train_labels: list[int]. Etiquetas del conjunto de entrenamiento
        - validation_features: list[str]. Archivos del conjunto de validación.
        - validation_labels: list[int]. Etiquetas del conjunto de validación
        - test_features: list[str]. Archivos del conjunto de prueba.
        - test_labels: list[int]. Etiquetas del conjunto de prueba
    """
    dataset_dict = {"train_features": train_features,
                    "train_labels": train_labels,
                    "validation_features": validation_features,
                    "validation_labels": validation_labels,
                    "test_features": test_features,
                    "test_labels": test_labels}
    with open(dataset_path, "w") as file:
        json.dump(dataset_dict, file)


def check_integrity_dumped_dataset(dataset_path, labels_map):
    """Chequea la integridad de un dataset en formato json guardado en disco

    Args:
        - dataset_path: Ubicación del dataset a verificar
        - labels_map: Mapeo de etiquetas originales (strings) a índices
        numéricos en el rango [0, n_classes - 1]. Si es que no se entrega un
        label_map, no se realiza la verificación de que cada filename esté
        asociado a su etiqueta correcta.

    Raises:
        AssertionError, si es que alguna de las condiciones de integridad no
        se cumple.
    """
    with open(dataset_path) as file:
        dataset_json = json.load(file)

    train_features = dataset_json["train_features"]
    train_labels = dataset_json["train_labels"]
    validation_features = dataset_json["validation_features"]
    validation_labels = dataset_json["validation_labels"]
    test_features = dataset_json["test_features"]
    test_labels = dataset_json["test_labels"]

    # Chequeamos que cada arreglo de features tenga la misma cantidad
    # de elementos que el arreglo de etiquetas correspondientes
    assert len(train_features) == len(train_labels)
    assert len(validation_features) == len(validation_labels)
    assert len(test_features) == len(test_labels)

    # Chequeamos que la partición generada sea en verdad una partición;
    # es decir, que ningún elemento esté repetido en más de una partición
    # y que además la unión de cada partición corresponde al total
    full_features = train_features + validation_features + test_features
    assert len(set(full_features)) == len(train_features) + \
        len(validation_features) + len(test_features)

    # Chequeamos correspondencia entre features y labels
    partitions = [(train_features, train_labels),
                  (validation_features, validation_labels),
                  (test_features, test_labels)]

    for partition_features, partition_labels in partitions:
        for filename, label in zip(partition_features, partition_labels):
            label_from_filename, _ = os.path.split(filename)
            assert labels_map[label_from_filename] == label


def generate_uneven_partition(some_list, n_partitions):
    """Particiona una lista de forma casi equilibrada, buscando que cada
    partición tenga aproximadamente la misma cantidad de elementos.

    Args:
        - some_list: list[any]. Lista a particionar.
        - n_partitions: int. Número deseado de particiones.

    Returns:
        - list[list[any]], lista con las particiones.
    """
    return [some_list[idx::n_partitions] for idx in range(n_partitions)]


def generate_kfold(n_folds, negative_slides, equivocal_slides, positive_slides,
                   train_dir, test_dir, datasets_dst_dir,
                   train_proportions, test_proportions,
                   proportion_threshold=0.1):
    """Genera k particiones de datos, cada una subdivida en conjuntos de
    entrenamiento, validación y prueba.

    El algoritmo funciona así: existen dos fuentes de datos (imágenes), una con
    parches extraídos directamente desde los ROIs marcados por los patólogos,
    y otra con todos los parches extraídos desde las slides. Por ello, los
    parches sacados desde las ROIs serán usados para entrenar y evaluar la red
    neuronal, mientras que los otros parches (los que fueron sacados
    directamente de las slides) serán entregados a la red, para que esta los
    clasifique de forma individual y así luego poder agregar estos resultados
    y clasificar la biopsia completa.

    Además, para generar las particiones se "aisla" un conjunto de biopsias; de
    esta forma, se entrena y valida con parches extraidos de los ROIs de un
    grupo de biopsias, y se predice en parches extraídos de otro grupo de
    slides. Obviamente, se busca que los conjuntos sean balanceados, con
    cantidades similares de slides negativas (0+ o 1+), equívocas (2+) y
    positivas (3+).

    Finalmente, también se buscará que los datasets generados contengan
    imágenes útiles; es decir, que al menos una porción de la imagen
    corresponda a tejido. El umbral para decidir utilidad (proporción mínima
    de la imagen que debe tener tejido) es otro parámetro de la función.

    Args:
        - n_folds: int. Número de particiones que se desea generar.
        - negative_slides: list[str], ids de las slides negativas
        (clasificación HER2 0+ o 1+)
        - equivocal_slides: list[str], ids de las slides equívocas
        (clasificación HER2 2+)
        - positive_slides: list[str], ids de las slides positivas
        (clasificación HER2 3+)
        - train_dir: str. Directorio con imágenes de entrenamiento (aquellas
        extraídas de los ROIs)
        - test_dir: str. Directorio con imágenes de prueba (aquellas
        extraídas indistintamente de las slides)
        - datasets_dst_dir: str. Directorio donde se guardarán los distintos
        archivos json con los datasets creados.
        - train_proportions: dict[str -> float]. Diccionario que tiene como
        llaves los nombres de las imágenes de entrenamiento, y como valores
        asociados la proporción de tejido correspondiente a cada imagen.
        - test_proportions: dict[str -> float]. Diccionario que tiene como
        llaves los nombres de las imágenes de prueba, y como valores
        asociados la proporción de tejido correspondiente a cada imagen.
        - proportion_threshold: float. Proporción mínima de tejido que debe
        tener una imagen para ser considerada en el dataset final.


    """

    def should_keep_patch(label_and_name, valid_ids, proportions_dict):
        """Determina si un parche debe ser mantenido en la base de datos, en
        base a la slide de la cual proviene y la proporción de tejido en la
        imagen.

        Args:
            - label_and_name: str. Etiqueta y nombre del parche en formato
            label/{slide_id}_rest.jpg
            - valid_ids: list[str]. Lista con ids de slides válidas.
            - proportions_dict: dict[str -> float]. Diccionario que tiene como
            llaves los nombres de las imágenes, y como valores
            asociados la proporción de tejido correspondiente a cada imagen.
        """
        slide_id = label_and_name.split("/")[1].split("_")[0]
        proportion = proportions_dict[label_and_name]
        return slide_id in valid_ids and proportion > proportion_threshold

    print(n_folds)
    os.makedirs(datasets_dst_dir, exist_ok=True)
    if n_folds == -1:
        n_folds = min(len(negative_slides), len(
            equivocal_slides), len(positive_slides))

    # Se unen las clases 0 y 1 (ambas son negativas)
    labels_map = {'0': 1, '1': 1, '2': 2, '3': 3}

    # Obtenemos los nombres de las imágenes del conjunto de entrenamiento
    train_fname, train_labels, _ = get_filenames_and_labels(
        train_dir, labels_map=labels_map)
    # Y del conjunto de prueba
    test_fname, test_labels, _ = get_filenames_and_labels(
        test_dir, labels_map=labels_map)

    # Zipeamos, para posteriormente hacer más fácil el filtrado
    train_data = list(zip(train_fname, train_labels))
    test_data = list(zip(test_fname, test_labels))

    # Aleatorizamos los conjuntos de ids
    negative_slides = np.random.permutation(negative_slides).tolist()
    equivocal_slides = np.random.permutation(equivocal_slides).tolist()
    positive_slides = np.random.permutation(positive_slides).tolist()

    # Generamos particiones en los conjuntos de ids (k-fold)
    negative_part = generate_uneven_partition(negative_slides, n_folds)
    equivocal_part = generate_uneven_partition(equivocal_slides, n_folds)
    positive_part = generate_uneven_partition(positive_slides, n_folds)

    for idx in range(n_folds):
        # Nombre del archivo json donde se guardarán los datos
        fold_dataset_path = os.path.join(
            datasets_dst_dir, "dataset_dict_fold_{}.json".format(idx + 1))

        # Slides de prueba para esta iteración
        test_ids = negative_part[idx] + equivocal_part[idx] + \
            positive_part[idx]

        # Slides de entrenamiento para esta iteración
        train_ids = negative_part[:idx] + negative_part[idx + 1:] + \
            equivocal_part[:idx] + equivocal_part[idx + 1:] + \
            positive_part[:idx] + positive_part[idx + 1:]

        # Aplanamos train_ids, porque en este momento es una lista de listas
        train_ids = list(itertools.chain.from_iterable(train_ids))

        print("TRAIN", train_ids)
        print("TEST", test_ids)
        # Filtramos las imágenes correspondientes al conjunto de entrenamiento
        train_fold = list(filter(
            lambda item: should_keep_patch(item[0], train_ids,
                                           train_proportions),
            train_data))
        # Filtramos las imágenes correspondientes al conjunto de prueba
        test_fold = list(filter(
            lambda item: should_keep_patch(item[0], test_ids,
                                           test_proportions),
            test_data))

        # Unzip, recuperamos nombres y etiquetas por separado
        train_fold_fnames, train_fold_labels = zip(*train_fold)
        train_fold_fnames = np.array(train_fold_fnames)
        train_fold_labels = np.array(train_fold_labels)

        test_fold_fnames, test_fold_labels = zip(*test_fold)
        test_fold_fnames = np.array(test_fold_fnames)
        test_fold_labels = np.array(test_fold_labels)

        # Generamos conjunto de validación desde el conjunto de prueba
        (train_fold_fnames, train_fold_labels), \
            (val_fold_fnames, val_fold_labels) = generate_validation_set(
                train_fold_fnames, train_fold_labels, 20)

        # Guardamos el dataset en disco
        dump_dataset(fold_dataset_path, train_fold_fnames.tolist(),
                     train_fold_labels.tolist(), val_fold_fnames.tolist(),
                     val_fold_labels.tolist(), test_fold_fnames.tolist(),
                     test_fold_labels.tolist())

        # Revisamos la integridad del dataset generado
        check_integrity_dumped_dataset(fold_dataset_path, labels_map)

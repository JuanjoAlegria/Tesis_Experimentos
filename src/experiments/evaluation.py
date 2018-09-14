"""Módulo para evaluar los resultados de un experimento.
"""
import os
import glob
import itertools
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from ..utils import outputs

PREDICTIONS_MAP = {"0": "negative", "1": "equivocal", "2": "positive"}
SCALE_MAP = {"x40": 1, "x20": 0.5, "x10": 0.25, "x5": 0.125}
COLORS_RGB = {'2': (0, 255, 0),
              '1': (255, 255, 0),
              '0': (255, 0, 0)}
COLORS_RGBA = {'2': (0, 255, 0, 30),
               '1': (255, 255, 0, 30),
               '0': (255, 0, 0, 30)}


def calculate_tissue_area(slide_id, tissue_proportions,
                          patch_width, patch_height):
    """Calcula el área de tejido (en pixeles) presente en la slide.

    Args:
        - slide_id: str. Id de la slide.
        - tissue_proportions: dict[str -> float]. Diccionario que tiene como
        llaves nombres de parches, y como valores asociados la proporción de
        tejido que hay en cada parche (entre 0 y 1).
        - patch_width: int. Ancho de cada parche (en pixeles)
        - patch_height: int. Altura de cada parche (en pixeles)

    Returns:
        int, área en pixeles correspondiente a tejido en la slide señalada.
    """
    slide_total_area = 0.0
    patch_area = patch_width * patch_height
    for label_and_name in tissue_proportions:
        _, image_name = label_and_name.split("/")
        if image_name.startswith(slide_id):
            slide_total_area += patch_area * \
                tissue_proportions[label_and_name]
    return round(slide_total_area)


def classify_slide_her2_from_areas(negative_area, equivocal_area,
                                   positive_area):
    """Clasifica una slide de acuerdo a su sobreexpresión de proteína HER2
    intentando seguir las guías clínica, habiendo calculado ya las áreas
    correspondientes.

    El algoritmo es este:
        - Si más de un 10% de la superficie total es de tipo positivo, entonces
        toda la slide es positiva (tipo 3+)
        - Si más de un 10% de la superficie total es de tipo equívoco, entonces
        toda la slide es equívoca (tipo 2+)
        - Si no se cumple nada de eso, es negativa (puede ser 1+ o 0+, pero
        en este momento, ambos casos se considerarán indistintos)

    Args:
        - positive_area: float. "Área" de la imagen con tejido positivo. En
        realidad, corresponde al resultado de área positiva total real
        dividido por el área de cada parche.
        - equivocal_area: float. "Área" de la imagen con tejido equívoco.
        - negative_area: float. "Área" de la imagen con tejido negativo.
    """
    total_area = positive_area + equivocal_area + negative_area
    if positive_area >= 0.1 * total_area:
        return 3
    elif equivocal_area >= 0.1 * total_area:
        return 2
    else:
        return 1


def classify_slide_her2(slide_id, network_predictions,
                        tissue_proportions):
    """Clasifica una slide de acuerdo a su sobreexpresión de proteína HER2
    intentando seguir las guías clínicas.

    Para ello, para cada parche se recuperan dos valores: la proporción de
    tejido presente en el parche y la clasificación que la red le dio. Luego,
    se asume que la densidad de células es uniforme en cada slide, y con eso
    en mente se intenta replicar la guía clínica.

    Args:
        - slide_id: str. Id de la biopsia.
        - network_predictions: dict[str -> (str, str)]. Diccionario con el
        nombre de la imagen como llave, y con valor asociado una tupla con la
        clase real de la imagen en primera posición y la clase predicha en
        segunda posición.
        - tissue_proportions: dict[str -> float]. Diccionario que tiene como
        llaves nombres de parches, y como valores asociados la proporción de
        tejido que hay en cada parche (entre 0 y 1).

    Returns:
        int, correspondiente a la clasificación HER2 de la slide:
            - 3 si es positiva (correspondiente a 3+)
            - 2 si es equívoca (correspondiente a 2+)
            - 1 si es negativa (correspondiente a 1+ o 0+)

    """
    areas = {"negative": 0.0, "equivocal": 0.0, "positive": 0.0}
    n_patches = {"negative": 0, "equivocal": 0, "positive": 0}
    for label_and_name in network_predictions:
        _, image_name = label_and_name.split("/")
        current_id, *_ = image_name.split("_")
        if current_id == slide_id:
            real_and_predicted = network_predictions[label_and_name]
            predicted_class_index = real_and_predicted[
                outputs.PREDICTED_CLASS_INDEX]
            predicted_class = PREDICTIONS_MAP[predicted_class_index]
            tissue_area = tissue_proportions[label_and_name]
            areas[predicted_class] += tissue_area
            n_patches[predicted_class] += 1
    final_classification = classify_slide_her2_from_areas(
        areas["negative"], areas["equivocal"], areas["positive"])
    print(areas)
    print(n_patches)
    print(final_classification)
    return final_classification


def get_mask_of_predictions(slide_id, predictions_dict,
                            image_size, roi_id=None,
                            patches_height=300, patches_width=300,
                            magnification="x40", colors=COLORS_RGBA):
    """Genera una máscara RGBA con las predicciones realizadas por el
    algoritmo, pintando con transparencia cada parche de acuerdo a su ubicación
    y clase predicha.

    Args:
        - slide_id: str. Id de la slide con la que se quiere trabajar.
        - roi_id: str. Id del roi, perteneciente a slide_id, con el que se
        quiere trabajar.
        - predictions_dict: dict[str -> (str, str)]. Diccionario con el nombre
        de la imagen como llave, y con valor asociado una tupla con la clase
        real de la imagen en primera posición y la clase predicha en segunda
        posición.
        - image_size: (int, int, int). Tamaño del roi.
        - patches_height: int. Altura en pixeles de los parches con que se
        evaluó el algoritmo.
        - patches_width: int. Ancho en pixeles de los parches con que se
        evaluó el algoritmo.
        - magnification: magnificación en la que se encuentra la imagen.
        - colors: dict[str->(int, int, int, int)]. Mapa de colores a usar,
        incluyendo el canal alpha. Si no se entrega un valor, se usará
        COLORS_RGBA.

    Returns:
        PIL.Image en formato RGBA, con cada ubicación pintada de acuerdo al
        valor predicho por la red.
    """

    scale = SCALE_MAP[magnification]
    patches_height = int(patches_height * scale)
    patches_width = int(patches_width * scale)

    x_coords, y_coords, preds = outputs.get_coords_and_preds(
        slide_id, predictions_dict, roi_id=roi_id)
    mask = Image.new('RGBA', image_size)
    draw = ImageDraw.Draw(mask)
    for x_coord, y_coord, prediction in zip(x_coords, y_coords, preds):
        x_coord, y_coord = int(x_coord * scale), int(y_coord * scale)
        draw.rectangle([(x_coord, y_coord),
                        (x_coord + patches_width, y_coord + patches_height)],
                       fill=colors[prediction])
    return mask


def generate_map_of_predictions_roi(slide_id, roi_id, rois_dir,
                                    predictions_dict,
                                    patches_height=300, patches_width=300,
                                    magnification="x40", colors=COLORS_RGBA):
    """Genera una imagen RGB, donde el fondo es un ROI y sobre él está pintado
    (con transparencia) cada parche de acuerdo a su ubicación y clase predicha
    por el algoritmo.

    Args:
        - slide_id: str. Id de la slide con la que se quiere trabajar.
        - roi_id: str. Id del roi, perteneciente a slide_id, con el que se
        quiere trabajar.
        - rois_dir: Directorio donde se encuentran los ROIs.
        - predictions_dict: dict[str -> (str, str)]. Diccionario con el nombre
        de la imagen como llave, y con valor asociado una tupla con la clase
        real de la imagen en primera posición y la clase predicha en segunda
        posición.
        - patches_height: int. Altura en pixeles de los parches con que se
        evaluó el algoritmo.
        - patches_width: int. Ancho en pixeles de los parches con que se
        evaluó el algoritmo.
        - magnification: magnificación en la que se encuentra la imagen.
        - colors: dict[str->(int, int, int, int)]. Mapa de colores a usar,
        incluyendo el canal alpha. Si no se entrega un valor, se usará
        COLORS_RGBA.

    Returns:
        PIL.Image en formato RGB, con cada ubicación pintada de acuerdo al
        valor predicho por la red, y de fondo el ROI original.
    """
    name_template = "{slide_id}_*_{roi_id}.jpg"
    roi_path_pattern = os.path.join(rois_dir, name_template.format(
        slide_id=slide_id, roi_id=roi_id))
    roi_path = glob.glob(roi_path_pattern)[0]
    return generate_map_of_predictions(roi_path, predictions_dict,
                                       slide_id, roi_id, patches_height,
                                       patches_width, magnification, colors)


def generate_map_of_predictions(img_path, predictions_dict,
                                slide_id, roi_id=None,
                                patches_height=300, patches_width=300,
                                magnification="x40", colors=COLORS_RGBA):
    """Genera una imagen RGB, donde el fondo es un ROI y sobre él está pintado
    (con transparencia) cada parche de acuerdo a su ubicación y clase predicha
    por el algoritmo.

    Args:
        - img_path: str. Ruta a la imagen.
        - predictions_dict: dict[str -> (str, str)]. Diccionario con el nombre
        de la imagen como llave, y con valor asociado una tupla con la clase
        real de la imagen en primera posición y la clase predicha en segunda
        posición.
        - slide_id: str. Id de la slide con la que se quiere trabajar.
        - roi_id: str. Id del roi, perteneciente a slide_id, con el que se
        quiere trabajar.
        - patches_height: int. Altura en pixeles de los parches con que se
        evaluó el algoritmo.
        - patches_width: int. Ancho en pixeles de los parches con que se
        evaluó el algoritmo.
        - magnification: magnificación en la que se encuentra la imagen.
        - colors: dict[str->(int, int, int, int)]. Mapa de colores a usar,
        incluyendo el canal alpha. Si no se entrega un valor, se usará
        COLORS_RGBA.

    Returns:
        PIL.Image en formato RGB, con cada ubicación pintada de acuerdo al
        valor predicho por la red, y de fondo el ROI original.
    """
    try:
        img = Image.open(img_path)
    except IOError:
        print("No se pudo abrir imagen", img_path)
    else:
        alpha = Image.new('RGBA', img.size)
        alpha.paste(img)
        mask_predictions = get_mask_of_predictions(slide_id, predictions_dict,
                                                   img.size, roi_id,
                                                   patches_height, patches_width,
                                                   magnification, colors)
        alpha.paste(mask_predictions, mask=mask_predictions)
        rgb = alpha.convert("RGB")
        return rgb


def generate_map_of_predictions_slides(slide_id, predictions_dict,
                                       patches_height=300, patches_width=300,
                                       map_height=8, map_width=8,
                                       colors=COLORS_RGB):
    """Genera una imagen con el mapeo de predicciones realizadas por el
    algoritmo de machine learning.

    Se genera una reproducción a escala de la slide original, donde para cada
    parche evaluado por la red, se le asigna un color dependiendo de la
    predicción realizada en una ventana de tamaño map_height x map_width.

    Args:
        - slide_id: str. Id de la slide cuyo mapeo se desea generar.
        - predictions_dict: dict[str -> (str, str)]. Diccionario con el nombre
        de la imagen como llave, y con valor asociado una tupla con la clase
        real de la imagen en primera posición y la clase predicha en segunda
        posición.
        - patches_height: int. Altura en pixeles de los parches con que se
        evaluó el algoritmo.
        - patches_width: int. Ancho en pixeles de los parches con que se
        evaluó el algoritmo.
        - map_height: int. Altura en pixeles que debe tener cada mapeo.
        - map_width: int. Ancho en pixeles que debe tener cada mapeo.
        - roi_id: str. Id del ROI particular que se quiere evaluar.
        - colors: dict[str->(int, int, int)]. Mapa de colores a usar. Si no
        se entrega un valor, se usará COLORS_RGB.
    Returns:
        3D - np.array, donde cada cuadrante ha sido coloreado de acuerdo a la
        predicción realizada por el algoritmo para el parche correspondiente.
    """
    x_coords, y_coords, preds = outputs.get_coords_and_preds(
        slide_id, predictions_dict)
    x_max = max(x_coords)
    y_max = max(y_coords)
    shape = (((y_max + patches_height) // patches_height) * map_height,
             ((x_max + patches_width) // patches_width) * map_width,
             3)
    map_image = np.zeros(shape, dtype='uint8')
    for x_coord, y_coord, prediction in zip(x_coords, y_coords, preds):
        rescaled_x = (x_coord // patches_width) * map_width
        rescaled_y = (y_coord // patches_height) * map_height
        current_color = colors[prediction]
        map_image[rescaled_y: rescaled_y + map_height,
                  rescaled_x: rescaled_x + map_width] = current_color
    return map_image


def plot_confusion_matrix(matrix, classes, title,
                          normalize=False, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    fig = plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]),
                                  range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    plt.ylabel('Clase verdadera ')
    plt.xlabel('Clase predicha')
    plt.tight_layout()
    return fig

# def map_prediction2label(json_predictions):
#     map_dict = {0: 1, 1: 3, 2: 0, 3: 2}
#     new_dict = {}
#     for score_class in json_predictions:
#         current_dict = json_predictions[score_class]
#         new_score_dict = {}
#         for image_name, old_label in current_dict.items():
#             new_label = map_dict[old_label]
#             new_score_dict[image_name] = new_label
#         new_dict[score_class] = new_score_dict
#     return new_dict

# def get_correct_predictions(json_path, to_be_corrected):
#     dir_path, name_ext = os.path.split(json_path)
#     json_name, ext = os.path.splitext(name_ext)
#     with open(json_path, "r") as f:
#         predictions = json.load(f)
#     if not to_be_corrected:
#         return predictions
#     else:
#         new_predictions = map_prediction2label(predictions)
#         new_path = os.path.join(dir_path, json_name + "_corrected.json")
#         with open(new_path, "w") as f:
#             json.dump(new_predictions, f)
#         return new_predictions

# def summary(y_true, y_pred, experiment_path, experiment_name,
#             partition):

#     n_images = len(y_true)
#     n0 = len(np.where(y_true == 0)[0])
#     n1 = len(np.where(y_true == 1)[0])
#     n2 = len(np.where(y_true == 2)[0])
#     n3 = len(np.where(y_true == 3)[0])
#     accuracy = accuracy_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred, average=None)

#     file_path = os.path.join(experiment_path, "summary_" + partition + ".txt")
#     with open(file_path, "w") as f:
#         f.write(experiment_name + "\n")
#         f.write("# imágenes clase 0+: " + str(n0) + "\n")
#         f.write("# imágenes clase 1+: " + str(n1) + "\n")
#         f.write("# imágenes clase 2+: " + str(n2) + "\n")
#         f.write("# imágenes clase 3+: " + str(n3) + "\n")
#         f.write("# imágenes total: " + str(n_images) + "\n")
#         f.write("Accuracy: " + str(accuracy) + "\n")
#         f.write("F1-score: " + str(f1) + "\n")

# def save_cnf_matrix(y_true, y_pred, dir_path, partition):
#     cnf_matrix = confusion_matrix(y_true, y_pred)
#     np.set_printoptions(precision=2)

#     class_names = ["0+", "1+", "2+", "3+"]
#     # Plot non-normalized confusion matrix
#     fig = plt.figure()
#     plot_confusion_matrix(cnf_matrix, classes=class_names,
#                           experiment_name=experiment_name,
#                           partition_name=partition)
#     fig_path = os.path.join(dir_path, "cnf_matrix_" + partition + ".png")
#     fig.savefig(fig_path)

#     # Plot normalized confusion matrix
#     fig = plt.figure()
#     plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                           experiment_name=experiment_name,
#                           partition_name=partition)
#     fig_path = os.path.join(dir_path, "cnf_matrix_norm_" + partition + ".png")
#     fig.savefig(fig_path)

#     # plt.show()

# def build_labels_and_predictions(predictions_json):
#     y_true = []
#     y_pred = []
#     for score_class in predictions_json:
#         score_dict = predictions_json[score_class]
#         for image_name in score_dict:
#             y_true.append(int(score_class))
#             y_pred.append(int(score_dict[image_name]))
#     return np.array(y_true), np.array(y_pred)

# def main(experiment_path, experiment_name, to_be_corrected):
#     for part_id, part_name in [("testing", "prueba"),
#                                ("training", "entrenamiento")]:
#         json_path = os.path.join(experiment_path,
#                                  "predictions_" + part_id + ".json")
#         predictions_json = get_correct_predictions(json_path, to_be_corrected)
#         y_true, y_pred = build_labels_and_predictions(predictions_json)
#         summary(y_true, y_pred, experiment_path, experiment_name, part_id)
#         save_cnf_matrix(y_true, y_pred, experiment_path, part_name)

# if __name__ == "__main__":
#     experiment_path = sys.argv[1]
#     experiment_name = sys.argv[2]
#     if len(sys.argv) < 4:
#         to_be_corrected = True
#     else:
#         to_be_corrected = bool(sys.argv[3])
#     main(experiment_path, experiment_name, to_be_corrected)

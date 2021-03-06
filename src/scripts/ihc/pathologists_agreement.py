"""Script para calcular el acuerdo entre patólogos con distintas
métricas, y considerando biopsias de resecciones, endoscopias y todas"""
import argparse
import numpy as np
import krippendorff
from sklearn.metrics import cohen_kappa_score
from ...utils import excel


def get_krippendorffs_alpha(evals_matrix):
    """Calcula el alpha de Krippendorff, tomando en cuenta todas las
    anotaciones realizadas por los patólogos. Esta medida estadística puede
    trabajar con datos perdidos, por lo cual no es necesario trabajar estos
    datos de forma especial. Además, debido a que la clasificación es de tipo
    (0, 1, 2, 3) y existe una jerarquía entre dichos valores, se usa una
    métrica de tipo ordinal.

    Args:
        - evals_matrix: np.array(), matriz con las evaluaciones realizadas por
        los patologos. Es una matriz de forma (N, M), donde N es el número de
        muestras y M es el número de anotadores. Además, la matriz debe ser
        de tipo float, y si es que existen datos perdidos, deben estar
        codificados como np.nan.

    Returns:
        dict[str: float], con una llave: krippendorff_alpha, la cual tiene
        asociada el alpha de Krippendorff calculado.
    """
    alpha = krippendorff.alpha(evals_matrix.transpose(),
                               level_of_measurement="ordinal")
    return {"krippendorff_alpha": alpha}


def get_aggrement_percentage(evals_matrix):
    """Calcula el acuerdo entre patólogos, medido como n_acuerdo/n_total.

    Args:
        - evals_matrix: np.array(), matriz con las evaluaciones realizadas por
        los patologos. Es una matriz de forma (N, M), donde N es el número de
        muestras y M es el número de anotadores. Además, la matriz debe ser
        de tipo float, y si es que existen datos perdidos, deben estar
        codificados como np.nan.

    Returns:
        dict[str: float], con seis llaves: aggrement_01, aggrement_02 y
        aggrement_12, las cuales tienen asociadas el porcentaje de acuerdo
        entre los patólogos 0-1, 0-2 y 1-2, respectivamente. Las otras tres
        llaves son n_01, n_02, n_12, correspondientes al número de
        observaciones utilizadas para calcular el kappa respectivo.
    """
    valid_path_0 = np.where(~np.isnan(evals_matrix[:, 0]))
    valid_path_1 = np.where(~np.isnan(evals_matrix[:, 1]))
    valid_path_2 = np.where(~np.isnan(evals_matrix[:, 2]))

    index_01 = np.intersect1d(valid_path_0, valid_path_1)
    index_02 = np.intersect1d(valid_path_0, valid_path_2)
    index_12 = np.intersect1d(valid_path_1, valid_path_2)

    equals_01 = evals_matrix[index_01, 0] == evals_matrix[index_01, 1]
    equals_02 = evals_matrix[index_02, 0] == evals_matrix[index_02, 2]
    equals_12 = evals_matrix[index_12, 1] == evals_matrix[index_12, 2]

    aggrement_01 = equals_01.astype(int).sum() / len(index_01)
    aggrement_02 = equals_02.astype(int).sum() / len(index_02)
    aggrement_12 = equals_12.astype(int).sum() / len(index_12)

    return {"aggrement_01": aggrement_01, "n_01": len(index_01),
            "aggrement_02": aggrement_02, "n_02": len(index_02),
            "aggrement_12": aggrement_12, "n_12": len(index_12)}


def get_cohens_kappa(evals_matrix):
    """Calcula el kappa de Cohen para cada par de patólogos, midiendo así el
    acuerdo entre los anotadores.

    Args:
        - evals_matrix: np.array(), matriz con las evaluaciones realizadas por
        los patologos. Es una matriz de forma (N, M), donde N es el número de
        muestras y M es el número de anotadores. Además, la matriz debe ser
        de tipo float, y si es que existen datos perdidos, deben estar
        codificados como np.nan.
    Returns:
        dict[str: float], con seis llaves: kappa_01, kappa_02 y kappa_12, las
        cuales tienen asociadas el kappa de Cohen que mide el acuerdo entre los
        patólogos 0-1, 0-2 y 1-2, respectivamente. Las otras tres llaves
        son n_01, n_02, n_12, correspondientes al número de observaciones
        utilizadas para calcular el kappa respectivo.
    """
    valid_path_0 = np.where(~np.isnan(evals_matrix[:, 0]))
    valid_path_1 = np.where(~np.isnan(evals_matrix[:, 1]))
    valid_path_2 = np.where(~np.isnan(evals_matrix[:, 2]))

    index_01 = np.intersect1d(valid_path_0, valid_path_1)
    index_02 = np.intersect1d(valid_path_0, valid_path_2)
    index_12 = np.intersect1d(valid_path_1, valid_path_2)

    kappa_01 = cohen_kappa_score(evals_matrix[index_01, 0].astype(int),
                                 evals_matrix[index_01, 1].astype(int))
    kappa_02 = cohen_kappa_score(evals_matrix[index_02, 0].astype(int),
                                 evals_matrix[index_02, 2].astype(int))
    kappa_12 = cohen_kappa_score(evals_matrix[index_12, 1].astype(int),
                                 evals_matrix[index_12, 2].astype(int))

    equals_01 = evals_matrix[index_01, 0] == evals_matrix[index_01, 1]
    equals_02 = evals_matrix[index_02, 0] == evals_matrix[index_02, 2]
    equals_12 = evals_matrix[index_12, 1] == evals_matrix[index_12, 2]
    aggrement_01 = equals_01.astype(int).sum() / len(index_01)
    aggrement_02 = equals_02.astype(int).sum() / len(index_02)
    aggrement_12 = equals_12.astype(int).sum() / len(index_12)

    return {"kappa_01": kappa_01, "n_01": len(index_01),
            "kappa_02": kappa_02, "n_02": len(index_02),
            "kappa_12": kappa_12, "n_12": len(index_12),
            "aggrement_01": aggrement_01,
            "aggrement_02": aggrement_02,
            "aggrement_12": aggrement_12}


def main(excel_path, biopsy_type):
    """Calcula el acuerdo entre patólogos con distintas métricas, pudiendo
    filtrar por biopsias de resecciones, endoscopias y todas
    """
    if biopsy_type.lower() == "endoscopy":
        biopsy_code = excel.ENDOSCOPY_CODE
    elif biopsy_type.lower() == "resection":
        biopsy_code = excel.RESECTION_CODE
    elif biopsy_type.lower() == "all":
        biopsy_code = excel.ALL_BIOPSIES_CODE
    else:
        raise ValueError(
            "Opciones de tipo de biopsias son: endoscopy, resection y all")

    _, evals_matrix = excel.get_slides_ids_full_eval(excel_path, biopsy_code)
    # Convertimos a float, lo cual pasa los None a np.NaN
    evals_matrix = np.array(evals_matrix).astype(float)
    # Eliminamos la última columna, la cual contiene la evaluación de consenso
    # y no debe ser considerada para analizar el acuerdo entre patólogos
    evals_matrix = evals_matrix[:, :-1]
    statistics = {}
    statistics.update(get_cohens_kappa(evals_matrix))
    statistics.update(get_krippendorffs_alpha(evals_matrix))
    statistics.update(get_aggrement_percentage(evals_matrix))
    print(biopsy_type.upper())
    print("############################")
    for key, value in statistics.items():
        print(key, ":", value)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--excel_path',
        type=str,
        help="Ruta al archivo excel con los datos""",
        default="data/extras/ihc_slides/HER2.xlsx"
    )
    PARSER.add_argument(
        '--biopsy_type',
        type=str,
        help="""Tipo de biopsias a analizar. Opciones son: endoscopy, resection
        y all""",
        default="all"
    )
    FLAGS = PARSER.parse_args()
    main(FLAGS.excel_path, FLAGS.biopsy_type)

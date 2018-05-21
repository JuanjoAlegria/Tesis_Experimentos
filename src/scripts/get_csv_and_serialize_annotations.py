"""Script para convertir las anotaciones en XML a objetos Annotation
serializados y generar un csv con la información resultante resumida.
"""
import os
import csv
import pickle
import argparse
from ..ndp import annotation


def write_csv(annotations_list, csv_path):
    """Escribe en csv_path un archivo csv con la información de las
    anotaciones entregadas en annotations_list.

    Args:
        - annotations_list: list[Annotation], anotaciones cuya información se
        quiere sumarizar.
        - csv_path: Ubicación del archivo csv.
    """
    header = ["Nombre slide", "Id anotación", "Tipo anotación",
              "Título anotación", "Autor anotación",
              "Detalles anotación", "Región"]
    lines = [header]
    for annotation_object in annotations_list:
        slide_name = annotation_object.slide_name
        annotation_id = annotation_object.annotation_id
        annotation_type = annotation_object.annotation_type
        title = annotation_object.title
        owner = annotation_object.owner
        details = annotation_object.details
        physical_region = annotation_object.physical_region
        info = [slide_name, annotation_id, annotation_type, title,
                owner, details]
        if physical_region is not None:
            info.append(str(physical_region))
        lines.append(info)

    with open(csv_path, "w") as csv_file:
        csv_writer = csv.writer(csv_file, 'excel')
        csv_writer.writerows(lines)


def main(annotations_dir, save_dir, csv_path):
    """Serializa cada anotación descrita en cada archivo xml en el directorio
    annotations_dir y las guarda en save_dir. Además, genera un archivo csv
    con la información sumarizada.
    """
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    all_annotations = []
    for annotation_name in os.listdir(annotations_dir):
        if ".xml" not in annotation_name:
            continue
        annotation_path = os.path.join(annotations_dir, annotation_name)
        all_annotations += annotation.get_all_annotations_from_xml(
            annotation_path)

    print("Cargadas {n} anotaciones".format(n=len(all_annotations)))

    for annotation_object in all_annotations:
        file_name = "{slide_name}_{annotation_id}.pkl".format(
            slide_name=annotation_object.slide_name,
            annotation_id=annotation_object.annotation_id)
        file_path = os.path.join(save_dir, file_name)
        with open(file_path, "wb") as pickle_file:
            pickle.dump(annotation_object, pickle_file)
            print(file_name, "serializado")

    write_csv(all_annotations, csv_path)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--annotations_dir',
        type=str,
        help="""\
        Directorio donde están guardadas las anotaciones xml. En caso de no
        entregar un valor, las anotaciones serán buscadas en la carpeta
        data/extras/ihc_slides/annotations.\
        """,
        default=os.path.join(os.getcwd(), "data", "extras",
                             "ihc_slides", "annotations")
    )
    PARSER.add_argument(
        '--save_dir',
        type=str,
        help="""\
        Directorio donde se guardarán los objetos Annotation serializados. En 
        caso de no entregar un valor, las anotaciones serán guardadas en la 
        carpeta data/extras/ihc_slides/annotations_serialized.\
        """,
        default=os.path.join(os.getcwd(), "data", "extras",
                             "ihc_slides", "annotations_serialized")
    )
    PARSER.add_argument(
        '--csv_path',
        type=str,
        help="""\
        Ubicación donde se guardará el csv con la información sumarizada. En 
        caso de no entregar un valor, las anotaciones serán guardadas en la 
        carpeta data/extras/ihc_slides/annotations_serialized/summary.csv.\
        """,
        default=os.path.join(os.getcwd(), "data", "extras",
                             "ihc_slides", "annotations_serialized",
                             "summary.csv")
    )

    FLAGS = PARSER.parse_args()
    main(FLAGS.annotations_dir, FLAGS.save_dir, FLAGS.csv_path)

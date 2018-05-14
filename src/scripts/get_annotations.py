"""Script para recuperar las anotaciones hechas a biopsias en una carpeta en
ndp.microscopiavirtual.com;
"""

import os
import argparse
import requests
from bs4 import BeautifulSoup

BASE_URL = 'http://ndp.microscopiavirtual.com/NDPServe.dll'
BIOPSIES_URL_TEMPLATE = BASE_URL + '?ViewItem?ItemID={folder_id}'
ANNOTATIONS_URL_TEMPLATE = BASE_URL + '?GetAnnotations?ItemID={item_id}'


def get_session(username, password):
    """Inicia una sesión en ndp.microscopiavirtual.com

    Args:
        - username: Nombre de usuario en ndp.microscopiavirtual.com
        - password: Contraseña en ndp.microscopiavirtual.com

    Returns:
        request.Session, después de iniciar sesión en
        ndp.microscopiavirtual.com. Este objeto guarda las cookies, con lo
        cual navegar por el sitio se vuelve mucho más simple.
    """
    sess = requests.Session()
    payload = {'Username': username,
               'Password': password, "SignIn": "Sign in"}
    sess.post(BASE_URL, data=payload)
    return sess


def get_biopsies_ids(sess, folder_id):
    """Obtiene un diccionario con biopsy_item_id, biopsy_name desde
    ndp.microscopiavirtual.com.

    Args:
        - sess: requests.Session(), con cookies de ndp.microscopiavirtual.com
        ya inicializadas.
        - folder_id: str. Id de la carpeta en ndp.microscopiavirtual.com que
        contiene las biopsias a estudiar.

    Returns:
        - dict[str: str], donde las llaves corresponden a las ids de las
        biopsias en ndp.microscopiavirtual.com, y los valores corresponden a
        los nombres de las biopsias
    """
    biopsies_dir_url = BIOPSIES_URL_TEMPLATE.format(folder_id=folder_id)
    biopsies_page = sess.get(biopsies_dir_url).content
    html_parser = BeautifulSoup(biopsies_page, 'html.parser')
    all_trs = html_parser.find_all('tr', {'class': 'ndpimage'})
    all_tds = [tr.find('td', {'class': 'name'}) for tr in all_trs]

    biopsies_item_ids_raw = [td.a['href'] for td in all_tds]
    biopsies_item_ids = [raw.split("ItemID=")[1]
                         for raw in biopsies_item_ids_raw]

    biopsies_names_raw = [td.text for td in all_tds]
    biopsies_names = [raw.split("Staining")[0] for raw in biopsies_names_raw]

    dict_biopsies = {biopsy_item_id: biopsy_name
                     for biopsy_item_id, biopsy_name in zip(biopsies_item_ids,
                                                            biopsies_names)}

    return dict_biopsies


def get_biopsy_annotation(sess, biopsy_item_id):
    """Obtiene la anotación correspondiente a una biopia.

    Args:
        - sess: requests.Session(), con cookies de ndp.microscopiavirtual.com
        ya inicializadas.
        - biopsy_item_id: str. ItemId de la biopsia de la cual se quiere
        recuperar la anotación

    Returns:
        - b'str, correspondiente a la anotación. La estructura de este string
        es la de un xml.
    """
    url = ANNOTATIONS_URL_TEMPLATE.format(item_id=biopsy_item_id)
    annotation = sess.get(url).content
    return annotation


def main(username, password, folder_id, annotations_dir):
    """Inicia sesión en ndp.microscopiavirtual.com, obtiene un diccionario
    con los item ids y los nombres de las biopsias, y guarda las anotaciones
    correspondientes a cada biopsia.

    Args:
        username: Nombre de usuario en ndp.microscopiavirtual.com.
        password: Contraseña en ndp.microscopiavirtual.com.
        folder_id: Id de la carpeta con biopsias.
        annotations_dir: Directorio donde se guardarán las anotaciones.
    """
    os.makedirs(annotations_dir, exist_ok=True)
    sess = get_session(username, password)
    dict_ids = get_biopsies_ids(sess, folder_id)
    for biopsy_item_id, biopsy_name in dict_ids.items():
        annotation = get_biopsy_annotation(sess, biopsy_item_id)
        xml_path = os.path.join(annotations_dir, biopsy_name + ".xml")
        with open(xml_path, "w") as xml:
            xml.write(annotation.decode('utf-8'))


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--username',
        type=str,
        required=True,
        help="Nombre de usuario en ndp.microscopiavirtual.com."
    )
    PARSER.add_argument(
        '--password',
        type=str,
        required=True,
        help="Contraseña en ndp.microscopiavirtual.com."
    )
    PARSER.add_argument(
        '--folder_id',
        type=str,
        help="""\
        Id de la carpeta con biopsias. Default: 10590, id de la carpeta 
        Unidad de Investigación.\
        """,
        default="10590"
    )
    PARSER.add_argument(
        '--annotations_dir',
        type=str,
        help="""\
        Directorio donde se guardarán las anotaciones. En caso de no entregar
        un valor, las anotaciones serán guardadas en la carpeta 
        data/ndp_annotations.\
        """,
        default=os.path.join(os.getcwd(), "data", "ndp_annotations")
    )

    FLAGS = PARSER.parse_args()
    main(FLAGS.username, FLAGS.password,
         FLAGS.folder_id, FLAGS.annotations_dir)

    # sess = get_session("MCerda", "7rU543qP")
    # ids = get_biopsies_ids(sess)
    # for key, value in ids.items():
    #     print(key, value)
    # for key in ids:
    #     annotation = get_biopsy_annotation(sess, key)
    #     import pdb
    #     pdb.set_trace()  # breakpoint 2a120fee //
    # print(2)

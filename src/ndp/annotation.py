"""Métodos para parser una anotación de npdi y para extraer las regiones allí
contenidas desde la imagen digitalizada correspondiente"""

import os
import xml.etree.ElementTree as ET
import openslide
import numpy as np
from . import ndpisplit_wrapper


class Point2D:
    """Clase que representa un punto en dos dimensiones, y con métodos para
    cambiar de coordenadas físicas a pixeles.

    Args:
        - x: int | float. Coordenada en eje x.
        - y: int | float. Coordenada en eje y.

    Returns:
        Point2D
    """

    def __init__(self, x_coord, y_coord):
        self.x_coord = x_coord
        self.y_coord = y_coord

    def to_pixels(self, x_offset, y_offset, mpp_x, mpp_y,
                  width_l0, height_l0, force_int=False):
        """Aplica una transformación lineal sobre el punto, pasándolo de
        coordenadas físicas a pixeles.

        El punto original (self) no es modificado, sino que se crea un punto
        nuevo. Para ello, el algoritmo que se ejecuta es:
            // Coordenadas físicas (nanómetros) respecto al centro de la
            // slide completa:
            physical_[x,y] = self.[x,y]
            // Coordenadas físicas (nanómetros) respecto al centro de la
            imagen principal:
            physical_[x,y] -= [x,y]_offset
            // Coordenadas físicas en micrómetros:
            physical_[x,y] /= 1000
            // Coordenadas en pixeles, respecto al centro de la imagen
            principal:
            pixels_[x,y] = physical_[x,y] / mpp_[x,y]
            // Coordenadas en pixeles, respecto a la esquina superior
            izquierda:
            pixels_[x,y] += [width, height]_l0 / 2

        Args:
            - x_offset: float. Distancia (en nm) desde el centro de la imagen
            completa al centro de la imagen principal en el eje x.
            - y_offset: float. Distancia (en nm) desde el centro de la imagen
            completa al centro de la imagen principal en el eje y.
            - mpp_x: float. Micrómetros por pixel, eje x.
            - mpp_y: float. Micrómetros por pixel, eje y.
            - width_l0: int. Ancho de la imagen (en pixeles) en su resolución
            más alta (generalamente, 40x)
            - height_l0: int. Altura de la imagen (en pixeles) en su resolución
            más alta (generalamente, 40x)
            - force_int: True si es que las coordenadas resultantes deben ser
            números enteros, False en caso de que se permitan floats.

        Returns:
            Point2D nuevo con las coordenadas transformadas de acuerdo al
            algoritmo descrito.
        """
        nm_x_center_whole = self.x_coord
        nm_x_center_main = nm_x_center_whole - x_offset
        um_x_center_main = nm_x_center_main / 1000
        pixels_x_center_main = um_x_center_main / mpp_x
        pixels_x_top_left_corner = pixels_x_center_main + (width_l0 / 2)

        nm_y_center_whole = self.y_coord
        nm_y_center_main = nm_y_center_whole - y_offset
        um_y_center_main = nm_y_center_main / 1000
        pixels_y_center_main = um_y_center_main / mpp_y
        pixels_y_top_left_corner = pixels_y_center_main + (height_l0 / 2)

        if force_int:
            pixels_x_top_left_corner = int(round(pixels_x_top_left_corner))
            pixels_y_top_left_corner = int(round(pixels_y_top_left_corner))
        return Point2D(pixels_x_top_left_corner, pixels_y_top_left_corner)

    def move(self, delta_x, delta_y):
        """Crea un nuevo punto desplazado.

        Args:
            - delta_x: float. Distancia a mover en el eje x.
            - delta_y: float. Distancia a mover en el eje y.

        Returns:
            Point2D nuevo trasladado de acuerdo a delta_x, delta_y
        """
        return Point2D(self.x_coord + delta_x,
                       self.y_coord + delta_y)

    def as_list(self):
        """Lista con las coordenadas (x, y).

        Returns:
            list[float] con coordenadas x, y.
        """
        return [self.x_coord, self.y_coord]

    def __str__(self):
        """Representación como string de un Point2D

        Returns:
            str que representa el punto
        """
        return "Point2D({x},{y})".format(x=self.x_coord,
                                         y=self.y_coord)


class CircularRegion:
    """Clase que representa una región circular.

    Args:
        center: Point, en coordenadas físicas (nanómetros)
        radius: float, en coordenadas físicas (nanómetros)

    Returns:
        CircularRegion
    """

    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def to_pixels(self, x_offset, y_offset, mpp_x, mpp_y,
                  width_l0, height_l0, force_int=False):
        """Transforma la región circular, pasándola desde coordenadas físicas
        en nanómetros relativas al centro de la imagen completa, hacia
        coordenadas en pixeles relativas a la esquina superior izquierda.

        El centro de la región es transformado de acuerdo al algoritmo
        descrito en Point.to_pixels, mientras que el radio, dado que es un
        escalar, sólo es escalado y no desplazado.

        Args:
            - x_offset: float. Distancia (en nm) desde el centro de la imagen
            completa al centro de la imagen principal en el eje x.
            - y_offset: float. Distancia (en nm) desde el centro de la imagen
            completa al centro de la imagen principal en el eje y.
            - mpp_x: float. Micrómetros por pixel, eje x.
            - mpp_y: float. Micrómetros por pixel, eje y.
            - width_l0: int. Ancho de la imagen (en pixeles) en su resolución
            más alta (generalamente, 40x)
            - height_l0: int. Altura de la imagen (en pixeles) en su resolución
            más alta (generalamente, 40x)
            - force_int: True si es que las coordenadas resultantes deben ser
            números enteros, False en caso de que se permitan floats.

        Returns:
            CircularRegion con coordenadas tranformadas.
        """
        mpp_average = (mpp_x + mpp_y) / 2
        new_center = self.center.to_pixels(x_offset, y_offset,
                                           mpp_x, mpp_y,
                                           width_l0, height_l0,
                                           force_int)
        new_radius = self.radius / (1000 * mpp_average)
        if force_int:
            new_radius = int(new_radius)
        return CircularRegion(new_center, new_radius)

    def get_bounding_box(self):
        """Crea una cuadro delimitador (bounding box) que contiene
        completamente a CircularRegion.

        Returns:
            np.array(shape=(4,2), dtype='int64') que representa el bounding
            box de CircularRegion.
        """
        corners = [self.center.move(-self.radius, -self.radius).as_list(),
                   self.center.move(+self.radius, -self.radius).as_list(),
                   self.center.move(-self.radius, +self.radius).as_list(),
                   self.center.move(+self.radius, +self.radius).as_list()]
        return np.array(corners, dtype='int64')

    def __str__(self):
        """Representación como string de una CircularRegion

        Returns:
            string que representa la región.
        """
        return "Center: {c} \nRadius: {r}".format(c=str(self.center),
                                                  r=self.radius)


class RectangularRegion:
    """Clase que representa una región rectangular.

    Args:
        points: list[Point], en coordenadas físicas (nanómetros)

    Returns:
        RectangularRegion
    """

    def __init__(self, points):
        self.points = points

    def to_pixels(self, x_offset, y_offset, mpp_x, mpp_y,
                  width_l0, height_l0, force_int=False):
        """Transforma la región rectangular, pasándola desde coordenadas
        físicas en nanómetros relativas al centro de la imagen completa, hacia
        coordenadas en pixeles relativas a la esquina superior izquierda.

        Todos los vértices de la región son transformados de acuerdo al
        algoritmo descrito en Point.to_pixels.

        Args:
            - x_offset: float. Distancia (en nm) desde el centro de la imagen
            completa al centro de la imagen principal en el eje x.
            - y_offset: float. Distancia (en nm) desde el centro de la imagen
            completa al centro de la imagen principal en el eje y.
            - mpp_x: float. Micrómetros por pixel, eje x.
            - mpp_y: float. Micrómetros por pixel, eje y.
            - width_l0: int. Ancho de la imagen (en pixeles) en su resolución
            más alta (generalamente, 40x)
            - height_l0: int. Altura de la imagen (en pixeles) en su resolución
            más alta (generalamente, 40x)
            - force_int: True si es que las coordenadas resultantes deben ser
            números enteros, False en caso de que se permitan floats.

        Returns:
            RectangularRegion con coordenadas tranformadas.
        """
        new_points = [p.to_pixels(x_offset, y_offset,
                                  mpp_x, mpp_y,
                                  width_l0, height_l0,
                                  force_int)
                      for p in self.points]

        return RectangularRegion(new_points)

    def get_bounding_box(self):
        """Crea una cuadro delimitador (bounding box) que contiene para
        RectangularRegion. En este caso, el bounding box corresponde a los
        mismos vértices que definen la RectangularRegion.

        Returns:
            np.array(shape=(4,2), dtype='int64') que representa el bounding
            box de RectangularRegion.
        """
        corners = [p.as_list() for p in self.points]
        return np.array(corners, dtype='int64')

    def __str__(self):
        """Representación como string de una RectangularRegion.

        Returns:
            string que representa la región.
        """
        string = [str(p) for p in self.points]
        return "\n".join(string)


class Annotation:
    """Clase que encapsula una anotación de NDP

    Args:
        - slide_name: str. Nombre de la biopsia a la que pertenece
        la anotación.
        - annotation_id: str. Id de la anotación, tal como está definida en
        el archivo xml correspondiente.
        - annotation_type: str. Tipo de anotación (circle, freehand, pin).
        - owner: str. Autor de la anotación, tal como está definida en
        el archivo xml correspondiente.
        - title: str. Título de la anotación.
        - details: str. Detalles añadidos a la anotación.
        - region: CircularRegion o RectangularRegion: ROI asociado a la
        anotación, en coordenadas físicas.

    Return:
        Annotation.
    """

    def __init__(self, slide_name, annotation_id, annotation_type,
                 title="", owner="", details="", region=None):
        self.slide_name = slide_name
        self.annotation_id = annotation_id
        self.annotation_type = annotation_type
        self.title = title
        self.owner = owner
        self.details = details
        self.physical_region = region

    def extract_region_from_ndpi(self, slide_path, slide_magnification="x40"):
        """Extrae la región contenida en la anotación desde un slide ndpi y la
        guarda en disco.

        Lamentablemente, ndpisplit no da la opción de dónde guardar la imagen
        extraída, por lo cual siempre es guardada en el mismo directorio donde
        reside la slide original.

        Args:
            - slide_path: str. Ubicación de la slide ndpi.
            - slide_magnification: str. Magnificación a la cual se quiere
            extraer la imagen (x5, x10, x20, x40).
        """

        x_offset, y_offset, mpp_x, mpp_y, \
            width_l0, height_l0 = get_properties_ndpi(slide_path)
        pixels_region = self.physical_region.to_pixels(
            x_offset, y_offset, mpp_x, mpp_y, width_l0, height_l0)

        bounding_box = pixels_region.get_bounding_box()
        (x_coord, y_coord, width, height) = get_top_left_and_size(bounding_box)

        ndpisplit_wrapper.extract_region(slide_path, x_coord, y_coord,
                                         width, height,
                                         magnification=slide_magnification,
                                         zlevel="z0",
                                         label=self.annotation_id)


def create_annotation_from_view(view, slide_name):
    """Crea una anotación a partir de un nodo xml de tipo ndpviewstate.

    Args:
        - view: xml.etree.ElementTree.Element, nodo de la anotación xml de
        tipo ndpviewstate.
        - slide_name: str. Nombre de la biopsia.
    """
    annotation_id = view.get('id')
    owner = view.get('owner')
    title = view.find("title").text
    details = view.find("details").text
    annotation_region = view.find("annotation")
    annotation_type = annotation_region.attrib["type"]
    if annotation_type == "circle":
        x_coord = float(annotation_region.find('x').text)
        y_coord = float(annotation_region.find('y').text)
        radius = float(annotation_region.find('radius').text)
        region = CircularRegion(Point2D(x_coord, y_coord), radius)
    elif annotation_type == "freehand":
        points = annotation_region.findall('pointlist/point')
        if len(points) != 4:
            error_message = """\
            Anotación {id} de biopsia {b_name} es de tipo freehand,
            pero tiene más de cuatro puntos, lo cual no está soportado
            """.format(id=annotation_id, b_name=slide_name)
            raise ValueError(error_message)
        corners = []
        for point in points:
            x_coord = float(point.findall('x')[0].text)
            y_coord = float(point.findall('y')[0].text)
            corners.append(Point2D(x_coord, y_coord))
        region = RectangularRegion(corners)
    else:
        region = None

    return Annotation(slide_name, annotation_id, annotation_type,
                      title=title, owner=owner, details=details, region=region)


def get_top_left_and_size(box):
    """Dado un bounding box, obtiene las coordenadas del punto ubicado en la
    esquina superior izquierda, y el ancho y alto de la imagen.

    Args:
        - box: np.array(shape=(4,2), dtype='int64') que representa un bounding
        box.

    Returns:
        list[num, num, num, num], lista con la coordenadas x,y de la esquina
        superior izquierda del bounding box (primeros dos elementos), y su
        ancho y alto (tercer y cuarto elemento).
    """
    x_min = box[:, 0].min()
    x_max = box[:, 0].max()
    y_min = box[:, 1].min()
    y_max = box[:, 1].max()
    width = x_max - x_min
    height = y_max - y_min
    coords = [x_min, y_min, width, height]
    return coords


def get_properties_ndpi(ndpi_path):
    """Obtiene las propiedades de un slide ndpi.

    Args:
        - ndpi_path: str. Ubicación de la imagen ndpi.

    Returns:
        - x_offset: float. Distancia (en nm) desde el centro de la imagen
        completa al centro de la imagen principal en el eje x.
        - y_offset: float. Distancia (en nm) desde el centro de la imagen
        completa al centro de la imagen principal en el eje y.
        - mpp_x: float. Micrómetros por pixel, eje x.
        - mpp_y: float. Micrómetros por pixel, eje y.
        - width_l0: int. Ancho de la imagen (en pixeles) en su resolución
        más alta (generalamente, 40x)
        - height_l0: int. Altura de la imagen (en pixeles) en su resolución
        más alta (generalamente, 40x)
    """
    slide = openslide.OpenSlide(ndpi_path)
    x_offset = float(slide.properties['hamamatsu.XOffsetFromSlideCentre'])
    y_offset = float(slide.properties['hamamatsu.YOffsetFromSlideCentre'])
    mpp_x = float(slide.properties['openslide.mpp-x'])
    mpp_y = float(slide.properties['openslide.mpp-y'])
    width_l0 = float(slide.properties['openslide.level[0].width'])
    height_l0 = float(slide.properties['openslide.level[0].height'])
    return x_offset, y_offset, mpp_x, mpp_y, width_l0, height_l0


def get_all_annotations_from_xml(xml_path):
    """Obtiene todas las anotaciones de un xml, incluyendo su id, autor,
    detalle y región asociaada

    Args:
        - xml_path: str. Ubicación del xml con anotaciones ndp
        - ndpi_path: str. Ubicación de la imagen ndpi correspondiente.  En caso
        de que este valor sea distinto de None, en cada anotación se agregará
        una región transformada a pixeles.

    Returns:
        list[Annotation], con todas las anotaciones en el archivo xml
    """

    _, slide_name_and_ext = os.path.split(xml_path)
    slide_name, _ = os.path.splitext(slide_name_and_ext)
    tree = ET.parse(xml_path)

    ndp_views = tree.findall('ndpviewstate')
    all_annotations = []
    for view in ndp_views:
        all_annotations.append(create_annotation_from_view(view, slide_name))
    return all_annotations

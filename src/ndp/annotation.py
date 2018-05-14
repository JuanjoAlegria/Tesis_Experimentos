"""Métodos para parser una anotación de npdi y para extraer las regiones allí
contenidas desde la imagen digitalizada correspondiente"""

# import cv2
# from . import ndpisplitWrapper
import xml.etree.ElementTree as ET
import openslide
import numpy as np


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
        pixels_x_main = um_x_center_main / (mpp_x * 1000)
        pixels_x_top_corner = pixels_x_main / (width_l0 / 2)

        nm_y_center_whole = self.y_coord
        nm_y_center_main = nm_y_center_whole - y_offset
        um_y_center_main = nm_y_center_main / 1000
        pixels_y_main = um_y_center_main / (mpp_y * 1000)
        pixels_y_top_corner = pixels_y_main / (height_l0 / 2)

        if force_int:
            pixels_x_top_corner = int(round(pixels_x_top_corner))
            pixels_y_top_corner = int(round(pixels_y_top_corner))
        return Point2D(pixels_x_top_corner, pixels_y_top_corner)

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

    def bounding_box(self):
        """Crea una cuadro delimitador (bounding box) que contiene
        completamente a CircularRegion.

        Returns:
            np.array(shape=(4,2), dtype='int64') que representa el bounding
            box de CircularRegion.
        """
        corners = [self.center.move(-self.radius, -self.radius).toArray(),
                   self.center.move(+self.radius, -self.radius).toArray(),
                   self.center.move(-self.radius, +self.radius).toArray(),
                   self.center.move(+self.radius, +self.radius).toArray()]
        return np.array(corners)

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

    def bounding_box(self):
        """Crea una cuadro delimitador (bounding box) que contiene para
        RectangularRegion. En este caso, el bounding box corresponde a los
        mismos vértices que definen la RectangularRegion.

        Returns:
            np.array(shape=(4,2), dtype='int64') que representa el bounding
            box de RectangularRegion.
        """
        corners = [p.toArray() for p in self.points]
        return np.array(corners)

    def __str__(self):
        """Representación como string de una RectangularRegion.

        Returns:
            string que representa la región.
        """
        string = [str(p) for p in self.points]
        return "\n".join(string)


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
    # coords = map(lambda x: int(round(x)), coords)
    return coords


def get_region_from_view(view):
    """Extrae las regiones desde un nodo xml de tipo ndpviewstate.

    Args:
        - view: xml.etree.ElementTree.Element, nodo de la anotación xml de
        tipo ndpviewstate.

    Return:
        CircularRegion o RectangularRegion; la región descrita en la anotación
        de ese ndpviewstate.

    Raises:
        ValueError, si es que el tipo de la anotación no es circle ni
        freehand.
    """

    annotation = view.find('annotation')
    a_type = annotation.attrib['type']
    if a_type == "circle":
        x_coord = float(annotation.find('x').text)
        y_coord = float(annotation.find('y').text)
        radius = float(annotation.find('radius').text)
        return CircularRegion(Point2D(x_coord, y_coord), radius)
    elif a_type == "freehand":
        points = view.findall('annotation/pointlist/point')
        corners = []
        for point in points:
            x_coord = float(point.findall('x')[0].text)
            y_coord = float(point.findall('y')[0].text)
            corners.append(Point2D(x_coord, y_coord))
        return RectangularRegion(corners)
    else:
        raise ValueError("Tipo " + a_type + " no soportado; sólo están" +
                         " soportados circle y freehand")


def get_info_from_view(view):
    """Extrae la id de la anotación, su autor y los detalles de ésta desde un
    nodo xml de tipo ndpviewstate.

    Args:
        - view: xml.etree.ElementTree.Element, nodo de la anotación xml de
        tipo ndpviewstate.

    Return:
        tuple(str, str, str), que contiene la id de la anotación, su autor y
        los detalles añadidos por el autor (potencialmente, el diagnóstico HER2
        de la región).
    """
    annotation_id = view.get('id')
    owner = view.get('owner')
    details = view.find("details").text

    return (annotation_id, owner, details)


def get_all_annotations_from_xml(xml_path, ndpi_path=None):
    """Obtiene todas las anotaciones de un xml, incluyendo su id, autor,
    detalle y región asociaada

    Args:
        - xml_path: str. Ubicación del xml con anotaciones ndp
        - ndpi_path: str. Ubicación de la imagen ndpi correspondiente.  En caso
        de que este valor sea distinto de None, en cada anotación se agregará
        una región transformada a pixeles.

    Returns:
        dict[str: dict[str:str, str:str,
                       str:CircularRegion | RectangularRegion,
                       str:CircularRegion | RectangularRegion // opcional
                      ]
            ].
        Diccionario con todas las anotaciones obtenidas desde xml_path, con la
        siguiente estructura:
        {
            annotation_id_1:
                {
                    "owner": owner, // string con autor
                    "details": details, // string con detalle
                    "physical_region": region // CircularRegion o
                        RectangularRegion
                    "pixel_region": region // CircularRegion o
                        RectangularRegion, opcional, sólo si es que
                        ndpi_path != None.
                }
            ...
        }
    """
    tree = ET.parse(xml_path)
    ndp_views = tree.findall('ndpviewstate')
    regions_and_info = {}
    if ndpi_path is not None:
        x_offset, y_offset, mpp_x, mpp_y, \
            width_l0, height_l0 = get_properties_ndpi(ndpi_path)

    for view in ndp_views:
        annotation_id, owner, details = get_info_from_view(view)
        physical_region = get_region_from_view(view)
        sub_dict = {}
        sub_dict["owner"] = owner
        sub_dict["details"] = details
        sub_dict["physical_region"] = physical_region
        if ndpi_path is not None:
            sub_dict["pixel_region"] = physical_region.to_pixels(
                x_offset, y_offset, mpp_x, mpp_y, width_l0, height_l0)
        regions_and_info[annotation_id] = sub_dict
    return regions_and_info


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


# if __name__ == "__main__":
#     pathXML = "/home/juanjo/U/2017/Tesis/Código/cellsFromAnnotations/annotation.xml"
#     pathNDPI = "/home/juanjo/U/2017/Tesis/Scaneos/IHQ/3/10/10.ndpi"
#     physical_regions = get_regions_from_xml(pathXML)
#     x_offset, y_offset, mpp_x, mpp_y, width_L0, height_L0 = get_properties_ndpi(
#         pathNDPI)
#     pixels_regions = [r.to_pixels(x_offset, y_offset, mpp_x,
#                                   mpp_y, width_L0, height_L0,
#                                   force_int=True)
#                       for r in physical_regions]

#     for r in pixels_regions:
#         bb = r.bounding_box()
#         x, y, w, h = get_coordinates_from_bb(bb)
#         ndpisplitWrapper.extractRegion(pathNDPI, x, y, w, h)

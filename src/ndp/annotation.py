import cv2
from ..utilities.ndpi import ndpisplitWrapper
import openslide
import numpy as np
import xml.etree.ElementTree as ET


class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def to_pixels(self, x_offset, y_offset, mpp_x, mpp_y,
                  width_L0, height_L0, force_int=False):
        x, y = self.x, self.y

        x -= x_offset
        x /= (mpp_x * 1000)
        x += width_L0 / 2

        y -= y_offset
        y /= (mpp_y * 1000)
        y += height_L0 / 2
        if force_int:
            x = int(round(x))
            y = int(round(y))
        return Point(x, y)

    def move(self, delta_x, delta_y):
        return Point(self.x + delta_x,
                     self.y + delta_y)

    def toArray(self):
        return [self.x, self.y]

    def __str__(self):
        return "Point({x},{y})".format(x=self.x,
                                       y=self.y)


class CircularRegion:

    def __init__(self, center, radius):
        """
            center: Point, en coordenadas físicas (nanómentros)
            radius: float, en coordenadas físicas (nanómentros)
        """
        self.center = center
        self.r = radius

    def to_pixels(self, x_offset, y_offset, mpp_x, mpp_y,
                  width_L0, height_L0, force_int=False):
        mpp_average = (mpp_x + mpp_y) / 2
        new_center = self.center.to_pixels(x_offset, y_offset,
                                           mpp_x, mpp_y,
                                           width_L0, height_L0,
                                           force_int)
        new_radius = self.r / (1000 * mpp_average)
        if force_int:
            new_radius = int(new_radius)
        return CircularRegion(new_center, new_radius)

    def bounding_box(self):
        corners = [self.center.move(-self.r, -self.r).toArray(),
                   self.center.move(+self.r, -self.r).toArray(),
                   self.center.move(-self.r, +self.r).toArray(),
                   self.center.move(+self.r, +self.r).toArray()]
        return np.array(corners)

    def __str__(self):
        return "Center: {c} \nRadius: {r}".format(c=str(self.center),
                                                  r=self.r)


class RectangularRegion:

    def __init__(self, points):
        """
            points: list[Point]
        """
        self.points = points

    def to_pixels(self, x_offset, y_offset, mpp_x, mpp_y,
                  width_L0, height_L0, force_int=False):
        new_points = [p.to_pixels(x_offset, y_offset,
                                  mpp_x, mpp_y,
                                  width_L0, height_L0,
                                  force_int)
                      for p in self.points]

        return RectangularRegion(new_points)

    def bounding_box(self):
        bb = [p.toArray() for p in self.points]
        return np.array(bb)

    def __str__(self):
        s = [str(p) for p in self.points]
        return "\n".join(s)


def get_coordinates_from_bb(box):
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
    annotation = view.find('annotation')
    a_type = annotation.attrib['type']
    if a_type == "circle":
        x = float(annotation.find('x').text)
        y = float(annotation.find('y').text)
        r = float(annotation.find('radius').text)
        return CircularRegion(Point(x, y), r)
    elif a_type == "freehand":
        points = view.findall('annotation/pointlist/point')
        corners = []
        for point in points:
            x = float(point.findall('x')[0].text)
            y = float(point.findall('y')[0].text)
            corners.append(Point(x, y))
        return RectangularRegion(corners)


def get_regions_from_xml(path_xml):
    tree = ET.parse(path_xml)
    ndpViews = tree.findall('ndpviewstate')
    regions = []
    for view in ndpViews:
        physical_region = get_region_from_view(view)
        regions.append(physical_region)
    return regions


def get_properties_ndpi(path_ndpi):
    slide = openslide.OpenSlide(path_ndpi)
    XOffset = float(slide.properties['hamamatsu.XOffsetFromSlideCentre'])
    YOffset = float(slide.properties['hamamatsu.YOffsetFromSlideCentre'])
    mppX = float(slide.properties['openslide.mpp-x'])
    mppY = float(slide.properties['openslide.mpp-y'])
    widthL0 = float(slide.properties['openslide.level[0].width'])
    heightL0 = float(slide.properties['openslide.level[0].height'])
    return XOffset, YOffset, mppX, mppY, widthL0, heightL0


if __name__ == "__main__":
    pathXML = "/home/juanjo/U/2017/Tesis/Código/cellsFromAnnotations/annotation.xml"
    pathNDPI = "/home/juanjo/U/2017/Tesis/Scaneos/IHQ/3/10/10.ndpi"
    physical_regions = get_regions_from_xml(pathXML)
    x_offset, y_offset, mpp_x, mpp_y, width_L0, height_L0 = get_properties_ndpi(
        pathNDPI)
    pixels_regions = [r.to_pixels(x_offset, y_offset, mpp_x,
                                  mpp_y, width_L0, height_L0,
                                  force_int=True)
                      for r in physical_regions]

    for r in pixels_regions:
        bb = r.bounding_box()
        x, y, w, h = get_coordinates_from_bb(bb)
        ndpisplitWrapper.extractRegion(pathNDPI, x, y, w, h)

import pdb
pdb.set_trace()  # breakpoint 60ac6cee //

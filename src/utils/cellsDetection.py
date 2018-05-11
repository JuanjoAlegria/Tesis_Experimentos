# -*- coding: utf-8 -*-

import cv2
import numpy as np

# LOWER_RANGE = None
# UPPER_RANGE = None
# KERNEL_MORPH = None


# def init(configDict):
#     global LOWER_RANGE, UPPER_RANGE, KERNEL_MORPH
#     LOWER_RANGE = np.array(configDict['system-independent']['lower_range_binarize'])
#     UPPER_RANGE = np.array(configDict['system-independent']['upper_range_binarize'])
#     if configDict['system-independent']['kernel_morph_shape'] == "MORPH_RECT":
#         kernel_shape = cv2.MORPH_RECT
#     elif configDict['system-independent']['kernel_morph_shape'] == "MORPH_ELLIPSE":
#         kernel_shape = cv2.MORPH_ELLIPSE
#     else:
#         raise Exception
#     kernel_size = tuple(configDict['system-independent']['kernel_morph_size'])
#     KERNEL_MORPH = cv2.getStructuringElement(kernel_shape, kernel_size)


def bgr2hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def binarize(img, lower, upper):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # máscara al aplicar threshold
    mask = cv2.inRange(img, lower, upper)
    # eliminar artefactos
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    # cerrar células
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=4)
    return closing


def getContours(binaryImage):
    # OJO: acá se invierte la posición, desde el formato fila columna de numpy
    # al formato x,y de imágenes; se mantiene que el 0,0 está en la esquina superior
    # izquierda
    img2, contours, hierarchy = cv2.findContours(binaryImage,
                                                 cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def getCentroid(moment):
    cx = int(moment['m10'] / moment['m00'])
    cy = int(moment['m01'] / moment['m00'])
    return (cx, cy)


def getCentroids(contours):
    centroids = np.zeros((len(contours), 2), dtype=int)
    for i in range(len(contours)):
        c = contours[i]
        moment = cv2.moments(c)
        centroids[i] = getCentroid(moment)
    return centroids


def getPixelsInRange(img, dictOfRanges):
    hsv = bgr2hsv(img)
    result = {}
    for colorName in dictOfRanges:
        lower = np.array(dictOfRanges[colorName][0])
        upper = np.array(dictOfRanges[colorName][1])
        mask = cv2.inRange(hsv, lower, upper)
        count = np.count_nonzero(mask)
        result[colorName] = count
    return result


def processAndGetCells(img, lower, upper):
    hsv = bgr2hsv(img)
    processedImage = binarize(hsv, lower, upper)
    del(hsv)
    contours = getContours(processedImage)
    return getCentroids(contours)

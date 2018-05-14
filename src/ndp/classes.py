# -*- coding: utf-8 -*-
import os
import re
import cv2
import glob
import pickle
import numpy as np
from . import ndpisplitWrapper
from .. import cellsDetection
from scipy.spatial import cKDTree


class MosaicException(Exception):
    pass


class NDPIMosaic:

    def __init__(self, ndpiPath, memory, compression, overlap, magnification, zlevel):
        folder, nameAndExt = os.path.split(ndpiPath)
        self.__filename = os.path.splitext(nameAndExt)[0]
        self.__folder = folder
        self.__overlap = overlap
        self.__magnification = magnification
        self.__zlevel = zlevel
        if compression.startswith("J"):
            self.__extension = ".jpg"
        else:
            self.__extension = ".tif"

        pattern = "{filename}_{magnification}_{zlevel}_*{fileExt}".format(
            filename=os.path.join(self.__folder, self.__filename),
            magnification=self.__magnification,
            zlevel=self.__zlevel,
            fileExt=self.__extension)

        if len(glob.glob(pattern)) == 0:
            print("Generando mosaico para imagen", self.__filename)
            ndpisplitWrapper.makeMosaic(ndpiPath, memory, compression,
                                        overlap, magnification, zlevel,
                                        forceMosaic=True)

        dummyFile = "{filename}_{magnification}_{zlevel}.tif".format(
            filename=os.path.join(self.__folder, self.__filename),
            magnification=self.__magnification,
            zlevel=self.__zlevel)

        if os.path.exists(dummyFile) and os.path.getsize(dummyFile) < 20:
            raise MosaicException(
                "Ocurrió un error. Tal vez problemas con la magnificación")

        self.__nRows, self.__nColumns = self.__getNRowsNCols()

        self.__filenameTemplate = os.path.join(self.__folder, self.__filename) + \
            "_" + self.__magnification + "_" + self.__zlevel + \
            "_i{i_format}j{j_format}" + self.__extension

        if self.__nRows < 10 and self.__nColumns < 10:
            self.__filenameTemplate = self.__filenameTemplate.format(i_format="{i}",
                                                                     j_format="{j}",)
        elif self.__nRows < 10:
            self.__filenameTemplate = self.__filenameTemplate.format(i_format="{i}",
                                                                     j_format="{j:02d}",)
        elif self.__nColumns < 10:
            self.__filenameTemplate = self.__filenameTemplate.format(i_format="{i:02d}",
                                                                     j_format="{j}",)
        else:  # nRows >= 10 and nColumns >=10
            self.__filenameTemplate = self.__filenameTemplate.format(i_format="{i:02d}",
                                                                     j_format="{j:02d}",)

        self.__height, self.__width = self.__getHeightWidth()

    # Calcula la cantidad de filas y columnas que tiene el mosaico generado
    def __getNRowsNCols(self):
        pattern = "{filename}_{magnification}_{zlevel}_*{fileExt}".format(
            filename=os.path.join(self.__folder, self.__filename),
            magnification=self.__magnification,
            zlevel=self.__zlevel,
            fileExt=self.__extension)
        L = glob.glob(pattern)
        L.sort()
        lastFile = L[-1]
        matchObject = re.match(r'.*i(\d*)j(\d*)', lastFile)
        i, j = int(matchObject.group(1)), int(matchObject.group(2))
        return i, j

    # Calcula la altura y ancho de cada imagen generada
    def __getHeightWidth(self):
        firstFilename = self.__filenameTemplate.format(i=1, j=1)
        height, width, _ = cv2.imread(firstFilename).shape
        return height - self.__overlap, width - self.__overlap

    # Dado que el módulo cellsDetection entrega centroides relativos a cada imagen del mosaico,
    # es necesario mover las coordenadas para que sean centroides absolutos
    def __absoluteCentroids(self, centroids, i, j):
        deltaX = (j - 1) * self.__width
        deltaY = (i - 1) * self.__height
        delta = [deltaX, deltaY]
        if (j > 1):
            delta[0] -= self.__overlap
        if (i > 1):
            delta[1] -= self.__overlap
        return centroids + delta

    # Eliminamos centroides redundantes, que se hayan producido por el overlap,
    # y que estén demasiado cerca (radio = 10). Luego, retornamos la lista de
    # centroides ordenada según la coordenada X
    def __cleanAndSortCentroids(self, centroids, radio=10):
        # Eliminamos repetidos
        centroids = np.unique(centroids, axis=0)
        # Eliminamos cercanos
        tree = cKDTree(centroids)
        rowsToFuse = np.array(list(tree.query_pairs(r=radio)))
        if len(rowsToFuse) > 0:
            for i in range(len(rowsToFuse)):
                rows = rowsToFuse[i]
                centroids[rows[0]] = (
                    centroids[rows[0]] + centroids[rows[1]]) / 2

            mask = np.ones(len(centroids), dtype=bool)
            mask[rowsToFuse[:, 1]] = False
            centroids = centroids[mask]
        # ordenamos por coordenada x
        centroids = centroids[np.argsort(centroids[:, 0])]
        return centroids

    # Obtiene los centroides de una imagen NDPI, usando para ello sus mosaicos
    def getCentroids(self, fLambda=None):

        totalCentroids = np.empty((0, 2), dtype=int)
        lower = np.array([92, 54, 127])
        upper = np.array([126, 105, 215])

        for i in range(1, self.__nRows + 1):
            for j in range(1, self.__nColumns + 1):
                imgPath = self.__filenameTemplate.format(i=i, j=j)
                img = cv2.imread(imgPath)
                relativeCentroids = cellsDetection.processAndGetCells(img,
                                                                      lower, upper)
                if fLambda and len(relativeCentroids) > 0:
                    lambdaFunction = fLambda(img)
                    relativeCentroids = list(
                        filter(lambdaFunction, relativeCentroids))

                if len(relativeCentroids) > 0:
                    relativeCentroids = np.array(relativeCentroids)
                    absoluteCentroids = self.__absoluteCentroids(
                        relativeCentroids, i, j)
                    totalCentroids = np.concatenate((totalCentroids,
                                                     absoluteCentroids))
                print("I =", i, "J =", j, "Shape =", totalCentroids.shape)

        return self.__cleanAndSortCentroids(totalCentroids)

    # Obtiene los centroides de una imagen NDPI, usando para ello sus mosaicos
    def getCentroidsAndExtractCells(self, cell_width, cell_height, fLambda):
        lower = np.array([92, 54, 127])
        upper = np.array([126, 105, 215])
        allCellImages = None
        for i in range(1, self.__nRows + 1):
            for j in range(1, self.__nColumns + 1):
                imgPath = self.__filenameTemplate.format(i=i, j=j)
                img = cv2.imread(imgPath)
                relativeCentroids = cellsDetection.processAndGetCells(img,
                                                                      lower,
                                                                      upper)
                if fLambda and len(relativeCentroids) > 0:
                    lambdaFunction = fLambda(img)
                    relativeCentroids = list(
                        filter(lambdaFunction, relativeCentroids))

                if len(relativeCentroids) > 0:
                    relativeCentroids = np.array(relativeCentroids)
                    currentRois = []
                    for centroid in relativeCentroids:
                        y, x = centroid
                        roi = img[x - cell_height // 2:x + cell_height // 2,
                                  y - cell_width // 2:y + cell_width // 2]
                        currentRois.append(roi)
                    currentRois = np.stack(currentRois)
                    if allCellImages is None:
                        allCellImages = currentRois
                    else:
                        allCellImages = np.concatenate(
                            (allCellImages, currentRois))
                N = 0 if allCellImages is None else len(allCellImages)
                print("I =", i, "J =", j, "Len =", N)
        print(allCellImages.shape)
        return allCellImages

    # Obtiene parches de la imagen
    def getPatches(self, patch_width, patch_height, lambdaFunction):
        for i in range(1, self.__nRows + 1):
            for j in range(1, self.__nColumns + 1):
                imgPath = self.__filenameTemplate.format(i=i, j=j)
                img = cv2.imread(imgPath)
                img_height, img_width, _ = img.shape
                for x in range(0, img_width, patch_width):
                    for y in range(0, img_height, patch_height):
                        current = img[y: y + patch_height,
                                      x: x + patch_width]
                        if current.shape == (patch_height, patch_width, 3) and \
                                lambdaFunction(current):
                            newName = os.path.splitext(imgPath)[0] + \
                                        "_x{}y{}.jpg".format(x, y)
                            newPath = os.path.join(self.__folder, newName)            
                            cv2.imwrite(newPath, current)

    # Elimina todas las imágenes generadas
    def clean(self):
        for i in range(1, self.__nRows + 1):
            for j in range(1, self.__nColumns + 1):
                try:
                    os.remove(self.__filenameTemplate.format(i=i, j=j))
                except FileNotFoundError:
                    print("No se pudo eliminar imagen",
                          self.__filenameTemplate.format(i, j))
        tifPath = "{filename}_{magnification}_{zlevel}.tif".format(
            filename=os.path.join(self.__folder, self.__filename),
            magnification=self.__magnification,
            zlevel=self.__zlevel)
        try:
            os.remove(tifPath)
        except FileNotFoundError:
            print("No se pudo eliminar imagen", tifPath)

    def getPixelsInRange(self, dictOfRanges):
        finalResult = {}
        for colorName in dictOfRanges:
            finalResult[colorName] = 0

        for i in range(1, self.__nRows + 1):
            for j in range(1, self.__nColumns + 1):
                imgPath = self.__filenameTemplate.format(i=i, j=j)
                img = cv2.imread(imgPath)
                localResult = cellsDetection.getPixelsInRange(
                    img, dictOfRanges)
                for colorName in dictOfRanges:
                    finalResult[colorName] += localResult[colorName]
                print("I: ", i, "J: ", j, "Pixels: ", localResult)

        return finalResult


class NDPImage:

    def __init__(self, path, magnification, zlevel):
        self.__path = path
        self.__folder, nameAndExt = os.path.split(path)
        self.__filename = os.path.splitext(nameAndExt)[0]
        self.__magnification = magnification
        self.__zlevel = zlevel
        self.__mosaic = None
        self.__centroidsPath = os.path.join(self.__folder,
                                            "centroids_{filename}.npy".
                                            format(filename=self.__filename))
        self.__pixelsInRangePath = os.path.join(self.__folder,
                                                "filterPixelsInRange_{filename}.pkl".
                                                format(filename=self.__filename))
        self.__cellsImages = os.path.join(self.__folder,
                                          "cellsImages_{filename}.pkl".
                                          format(filename=self.__filename))

    def makeMosaic(self, memory, compression, overlap):
        self.__mosaic = NDPIMosaic(self.__path, memory, compression,
                                   overlap, self.__magnification, self.__zlevel)

    def getPixelsInRange(self, dictOfRanges, memory="",
                         compression="", overlap=0, override=False):
        if os.path.exists(self.__pixelsInRangePath) and not override:
            print("Pixeles en rangos ya calculados en ruta",
                  self.__pixelsInRangePath)
            with open(self.__pixelsInRangePath, "rb") as file:
                return pickle.load(file)
        else:
            # Buscar solución más bonita
            self.makeMosaic(memory, compression, overlap)
            print("Calculando pixeles en rango a partir de mosaico")
            distribution = self.__mosaic.getPixelsInRange(dictOfRanges)
            with open(self.__pixelsInRangePath, "wb") as file:
                pickle.dump(distribution, file)
            return distribution

    def getCentroids(self, memory=None, compression=None,
                     overlap=None, override=False, fLambda=None, path=None):
        if not path:
            path = self.__centroidsPath
        if os.path.exists(path) and not override:
            print("Utilizando centroides ya calculados en ruta", path)
            centroids = np.load(path)
        else:
            if self.__mosaic is None and memory and compression and overlap:
                self.makeMosaic(memory, compression, overlap)
            print("Calculando centroides a partir de mosaico")
            centroids = self.__mosaic.getCentroids(fLambda)
            np.save(path, centroids)
        return centroids

    def getCentroidsAndExtractCells(self, cell_width, cell_height, fLambda):
        cells = self.__mosaic.getCentroidsAndExtractCells(cell_width,
                                                          cell_height, fLambda)
        np.save(self.__cellsImages, cells)
        return cells


    def getPatches(self, patch_width, patch_height, lambdaFunction):
        self.__mosaic.getPatches(patch_width, patch_height, lambdaFunction)

    def cleanMosaic(self):
        if self.__mosaic:
            self.__mosaic.clean()

    def extractCells(self, cell_width, cell_height, centroidsFile=None):
        centroids = self.getCentroids(path=centroidsFile)
        if self.__mosaic:
            self.__mosaic.extractCells(cell_width, cell_height, centroids)
        for i in range(len(centroids)):
            if i % 1000 == 0:
                print("Extraídas ", i, "/", len(centroids), "células")
            x, y = centroids[i]
            x = (0 if x - cell_width // 2 < 0
                 else x - cell_width // 2)
            y = (0 if y - cell_height // 2 < 0
                 else y - cell_height // 2)
            ndpisplitWrapper.extractRegion(self.__path, x, y, cell_width,
                                           cell_height, self.__magnification,
                                           self.__zlevel, str(i))

# -*- coding: utf-8 -*-
import platform
from subprocess import call

NDPISPLIT_BINARY = "ndpisplit"


def init(configDict):
    global NDPISPLIT_BINARY
    NDPISPLIT_BINARY = configDict[platform.system()]['ndpi_binary']


def makeMosaic(ndpiFilePath, memory=100, compression="J100", overlap=500,
               magnification="x40", z_level="z0", forceMosaic=False):
    baseCommand = "{ndpiBinary} -{mosaicCommand}{memory}{compression} -o{overlap}" + \
                  " -{magnification} -{z_level} {ndpiFilePath}"
    mosaic = "M" if forceMosaic else "m"
    command = baseCommand.format(ndpiBinary=NDPISPLIT_BINARY,
                                 mosaicCommand=mosaic, memory=memory,
                                 compression=compression, overlap=overlap,
                                 magnification=magnification, z_level=z_level,
                                 ndpiFilePath=ndpiFilePath)
    print("Se ejecuta el comando", command)
    call(command, shell=True)


def extractRegion(ndpiFilePath, x, y, width, height,
                  magnification="x40", zlevel="z0", label=None):
    baseCommand = "{ndpiBinary} -E{magnification},{zlevel},{x},{y},{width},{height}"
    if label:
        baseCommand += "," + label
    baseCommand += " {path}"

    command = baseCommand.format(ndpiBinary=NDPISPLIT_BINARY,
                                 magnification=magnification,
                                 zlevel=zlevel, x=x, y=y, width=width,
                                 height=height, path=ndpiFilePath)

    call(command, shell=True)


def extractFullImage(ndpiFilePath, magnification, zlevel="z0", memory=""):
    baseCommand = "{ndpiBinary} -m{memory}J -{magnification}{zlevel} {path}"
    command = baseCommand.format(ndpiBinary=NDPISPLIT_BINARY,
                                 memory=memory, magnification=magnification,
                                 zlevel=zlevel, path=ndpiFilePath)
    call(command, shell=True)

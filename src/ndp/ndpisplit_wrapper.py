"""Módulo para llamar programáticamente a la utilidad ndpisplit
"""
from subprocess import call


def make_mosaic(ndpi_path, memory=100, compression="J100", overlap=500,
                magnification="x40", z_level="z0", force_mosaic=False):
    """Genera un mosaico a partir de una imagen NDPI

    Args:
        - ndpi_path: str. Ubicación de la imagen NDPI.
        - memory: int. Tamaño máximo (en megabytes) que puede pesar la imagen
        descomprimida.
        - compression: str. "J" para imágenes JPEG, "j" para imágenes TIFF con
        compresión JPEG, "n" para ninguna compresión, "l" para compresión lzw.
        "J" y "j" pueden estar seguidos de un entero entre 0 y 100 indicando
        la calidad de la imagen producida (si no se provee ningún número, la
        calidad por defecto es 75).
        - overlap: int. Cantidad de pixeles de superposición al generar el
        mosaico.
        - magnification: str. Magnificación de la imagen desde la cual se deben
        extraer los mosaicos (x5, x10, x20, x40).
        - z_level: Nivel desde el cual se deben extraer las imágenes. Sólo
        tiene sentido si es que la imagen está tomada en varios niveles.
        - force_mosaic: si es True, se generará un mosaico independientemente
        de que la imagen completa quepa en memoria.
    """
    base_command = "ndpisplit -{mosaicCommand}{memory}{compression}" + \
        " -o{overlap} -{magnification} -{z_level} {ndpi_path}"
    mosaic = "M" if force_mosaic else "m"
    command = base_command.format(mosaicCommand=mosaic, memory=memory,
                                  compression=compression, overlap=overlap,
                                  magnification=magnification, z_level=z_level,
                                  ndpi_path=ndpi_path)
    print("Se ejecuta el comando", command)
    call(command, shell=True)


def extract_region(ndpi_path, x_coord, y_coord, width, height,
                   magnification="x40", zlevel="z0", label=None):
    """Extrae una región rectangular desde una imagen ndpi.

    Args:
        - ndpi_path: str. Ubicación de la imagen NDPI.
        - x_coord: int. coordenada x de la esquina superior izquierda de la
        región a extraer.
        - y_coord: int. coordenada y de la esquina superior izquierda de la
        región a extraer.
        - width: int. Ancho en pixeles de la región a extraer.
        - height: int. Altura en pixeles de la región a extraer.
        - magnification: str. Magnificación de la imagen desde la cual se deben
        extraer la región deseada (x5, x10, x20, x40).
        - z_level: Nivel desde el cual se deben extraer la región. Sólo
        tiene sentido si es que la imagen está tomada en varios niveles.
        - label: str. Sufijo que se debe añadir al nombre de la imagen
        extraida.
    """
    base_command = "ndpisplit " + \
        "-E{magnification},{zlevel},{x},{y},{width},{height}"
    if label:
        base_command += "," + label
    base_command += " {path}"

    command = base_command.format(magnification=magnification,
                                  zlevel=zlevel, x=x_coord, y=y_coord,
                                  width=width, height=height, path=ndpi_path)
    print("Se ejecuta el comando", command)
    call(command, shell=True)

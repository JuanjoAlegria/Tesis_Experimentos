import os
import cv2


def traverseHer2Dir(path, functionBiopsy, functionScore, verbose=True):
    """
        traverseHer2Dir: str, (str -> any), (str -> any) -> None
        - path es el string que representa el directorio con scaneos,
        con estructura score/biopsyName/biopsy.ndpi (y otros archivos).
        - functionBiopsy es una función que será invocada al llegar a cada
        directorio interno y tomará como parámetro dicho directorio
        (formato path/score/biopsyName). Lo que functionBiopsy retorne
        será almacenado en un diccionario.
        - functionScore es una función que será invocada al terminar de
        aplicar functionBiopsy sobre cada biopsia de esa clase, y recibirá
        como parámetro el directorio de la clase (formato path/score) y un
        diccionario donde estarán almacenados los resultados obtenidos
        por functionBiopsy sobre cada biopsia correspondiente
    """
    for her2Score in os.listdir(path):  # score será igual a 0,1,2 y 3
        if her2Score not in ["0", "1", "2", "3"]:
            continue
        if verbose:
            print("HER2 Score:", her2Score)
        scoreFullPath = os.path.join(path, her2Score)
        resultsScore = {}
        for biopsyDir in os.listdir(scoreFullPath):
            biopsyDirFullPath = os.path.join(scoreFullPath, biopsyDir)
            if not os.path.isdir(biopsyDirFullPath) or biopsyDir == "Samples":
                continue
            if verbose:
                print("Biopsy name:", biopsyDir)
            if functionBiopsy:
                resultsBiopsy = functionBiopsy(biopsyDirFullPath)
                resultsScore[biopsyDir] = resultsBiopsy
        if functionScore:
            functionScore(scoreFullPath, resultsScore)


def show(img, title, fx, fy, close=True):
    img = cv2.resize(img, None, fx=fx, fy=fy)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    if close:
        cv2.destroyAllWindows()


def paintPoint(img, point, size, colour):
    diff = size // 2
    img[point[0] - diff: point[0] + diff, point[1] - diff: point[1] + diff] = colour


def cleanFolder(folder, ignoreExt):
    # elimina todos los archivos, excepto la imagen ndpi y los centroides
    for file in os.listdir(folder):
        name, ext = os.path.splitext(file)
        if ext not in ignoreExt:
            os.remove(os.path.join(folder, file))

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy.signal import medfilt
from skimage.color import hdx_from_rgb, separate_stains
from sklearn.cluster import KMeans


def deconv_ihc(image):
    ihc_hdx = separate_stains(image, hdx_from_rgb)
    return ihc_hdx[:, :, 0], ihc_hdx[:, :, 1], ihc_hdx[:, :, 2]


def average_neighborhood(src, radius):
    """Para cada punto de la imagen, calcula el promedio de intensidad de 
    los pixeles vecinos.

    Esto es equivalente a realizar una convolución con un filtro constante, lo
    cual permite optimizar la operación.

    Args:
        - src: np.array(float). Imagen, con sólo un canal.
        - radius: int. Radio de la vecindad.

    Returns:
        floats. Promedios de intensidades en la vecindad señalada.
    """
    if len(src.shape) > 2:
        raise ValueError("Imagen debe tener sólo un canal")
    diameter = 2 * radius + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (diameter, diameter)).astype(float)
    n_pixels = np.count_nonzero(kernel)
    kernel /= n_pixels

    new_image = cv2.filter2D(src, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return new_image


def minimum_neighborhood(src, radius):
    if len(src.shape) > 2:
        raise ValueError("Imagen debe tener sólo un canal")
    diameter = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    new_image = cv2.erode(src, kernel, iterations=1,
                          borderType=cv2.BORDER_REPLICATE)
    return new_image


def clusterize(image, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters)
    image_as_vector = image.ravel().reshape(-1, 1)
    result = kmeans.fit_predict(image_as_vector)
    clusterized_image = result.reshape(image.shape[0], image.shape[1])
    return clusterized_image, kmeans.cluster_centers_


def threshold(image, radius):
    diameter = 2 * radius + 1
    # medians = medfilt(image, (diameter, diameter))
    # image_minus_medians = image - medians
    normalized_image = np.zeros(image.shape)
    cv2.normalize(image, normalized_image, alpha=0, beta=255,
                  norm_type=cv2.NORM_MINMAX)
    normalized_image = normalized_image.astype('uint8')
    binary_image = cv2.adaptiveThreshold(normalized_image, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, diameter, C=0)
    return binary_image


def cluster_cancer(image, radius):
    h_channel, dab_channel, rest_channel = deconv_ihc(image)

    avg_image = average_neighborhood(dab_channel, radius)
    min_image = minimum_neighborhood(avg_image, radius)
    clusterized_image, centroids = clusterize(min_image)

    clusterized_4_colors = np.zeros(image.shape, dtype='uint8')
    clusterized_4_colors[clusterized_image == 0] = [0, 0, 0]
    clusterized_4_colors[clusterized_image == 1] = [255, 0, 0]
    clusterized_4_colors[clusterized_image == 2] = [0, 255, 0]
    clusterized_4_colors[clusterized_image == 3] = [0, 0, 255]

    min_centroid = np.argmin(centroids)
    clusterized_binary = np.zeros(image.shape, dtype='uint8')
    clusterized_binary[clusterized_image == min_centroid] = [0, 0, 0]
    clusterized_binary[clusterized_image != min_centroid] = [255, 255, 255]

    fig, axes = plt.subplots(2, 4, figsize=(7, 6), sharex=True, sharey=True)
    axes = axes.ravel()

    axes[0].imshow(image)
    axes[0].set_title("Original image")

    axes[1].imshow(h_channel, cmap="gray")
    axes[1].set_title("H channel")

    axes[2].imshow(dab_channel, cmap="gray")
    axes[2].set_title("DAB channel")

    axes[3].imshow(avg_image, cmap="gray")
    axes[3].set_title("Avg image")

    axes[4].imshow(min_image, cmap="gray")
    axes[4].set_title("Min image")

    axes[5].imshow(clusterized_4_colors)
    axes[5].set_title("Clusterized")

    axes[6].imshow(clusterized_binary)
    axes[6].set_title("Clusterized binary")

    fig.tight_layout()
    plt.show()


def segment_nuclei(image, radius):
    h_channel, dab_channel, rest_channel = deconv_ihc(image)
    h_channel = h_channel
    binary = threshold(h_channel, radius)

    fig, axes = plt.subplots(1, 3, figsize=(7, 6), sharex=True, sharey=True)
    axes = axes.ravel()

    axes[0].imshow(image)
    axes[0].set_title("Original image")

    axes[1].imshow(h_channel, cmap="gray")
    axes[1].set_title("H channel")

    axes[2].imshow(binary, cmap="gray")
    axes[2].set_title("Threshold")

    fig.tight_layout()
    plt.show()


def main(image_path, radius_neighborhood, radius_threshold, random_seed):
    np.random.seed(random_seed)
    bgr_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    cluster_cancer(rgb_image, radius_neighborhood)
    #segment_nuclei(rgb_image, radius_threshold)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--image_path',
        type=str,
        help="Ubicación de la imagen",
        required=True
    )
    PARSER.add_argument(
        '--radius_neighborhood',
        type=int,
        help="Radio de la vecindad",
        default=10
    )
    PARSER.add_argument(
        '--radius_threshold',
        type=int,
        help="Radio de la vecindad para el umbral adaptativo",
        default=10
    )
    PARSER.add_argument(
        '--random_seed',
        type=int,
        default=0,
        help='Semilla aleatoria (útil para obteer resultados reproducibles)',
    )
    FLAGS = PARSER.parse_args()
    main(FLAGS.image_path, FLAGS.radius_neighborhood,
         FLAGS.radius_threshold, FLAGS.random_seed)

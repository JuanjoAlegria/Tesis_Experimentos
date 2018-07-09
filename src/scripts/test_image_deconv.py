import argparse

import numpy as np
from skimage.color import hdx_from_rgb, separate_stains
from skimage.io import imread
from skimage.filters import threshold_yen, threshold_local, try_all_threshold
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import img_as_ubyte, img_as_float
from skimage.exposure import rescale_intensity
from scipy import ndimage as ndi

import cv2
import matplotlib.pyplot as plt


def try_all(img):
    fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
    plt.show()


def deconv_ihc(image):
    ihc_hdx = separate_stains(image, hdx_from_rgb)
    return ihc_hdx[:, :, 0], ihc_hdx[:, :, 1], ihc_hdx[:, :, 2]


def binarize(channel):
    channel *= 5
    kernel = np.ones((3, 3), np.uint8)
    tresh = threshold_yen(channel)
    mask = channel > tresh
    binarized_channel = channel.copy()
    binarized_channel[mask] = 255
    binarized_channel[~mask] = 0
    binarized_channel = cv2.morphologyEx(binarized_channel,
                                         cv2.MORPH_OPEN,
                                         kernel, iterations=2)
    binarized_channel = cv2.erode(binarized_channel, kernel, iterations=5)
    return binarized_channel


def get_background_and_foreground(channel):
    # Binarizar imagen
    tresh = threshold_yen(channel)
    mask = channel > tresh
    binarized_channel = channel.copy().astype('uint8')
    binarized_channel[mask] = 255
    binarized_channel[~mask] = 0
    # Eliminar ruido
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(
        binarized_channel, cv2.MORPH_OPEN, kernel, iterations=3)
    # Área que con seguridad no es una célula
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Area que con seguridad está dentro de la célula
    # almost_sure_fg = cv2.erode(opening, kernel, iterations=1)
    dist_transform = cv2.distanceTransform(
        opening, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, sure_fg = cv2.threshold(
        dist_transform, 0.1 * dist_transform.max(),
        255, cv2.THRESH_BINARY)
    # Área en la cual no tenemos certeza
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    return sure_bg, sure_fg, unknown


def watershed(image, sure_fg, unknown):
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    # Pasamos de un canal a tres canales
    # image = cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]
    plt.imshow(image)
    plt.show()
    return markers


def merge_channels(h_channel, dab_channel):
    return (h_channel * 2) - dab_channel


def plot_deconv(image, h_channel, dab_channel, rest_channel):

    fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
    axes = axes.ravel()

    axes[0].imshow(image)
    axes[0].set_title("Original image")

    axes[1].imshow(h_channel, cmap="Greys")
    axes[1].set_title("Hematoxylin")

    axes[2].imshow(dab_channel, cmap="Greys")
    axes[2].set_title("DAB")

    axes[3].imshow(rest_channel, cmap="Greys")
    axes[3].set_title("Rest")

    # for a in axes:
    #     a.axis('off')

    fig.tight_layout()
    plt.show()


def plot_merge(image, merged):
    fig, axes = plt.subplots(1, 2, figsize=(7, 6), sharex=True, sharey=True)
    axes = axes.ravel()

    axes[0].imshow(image)
    axes[0].set_title("Original image")

    axes[1].imshow(merged, cmap="Greys")
    axes[1].set_title("Merged image")

    # for a in axes:
    #     a.axis('off')

    fig.tight_layout()
    plt.show()


def plot_binarized(image, channel, binarized_channel):
    fig, axes = plt.subplots(1, 3, figsize=(7, 6), sharex=True, sharey=True)
    axes = axes.ravel()

    axes[0].imshow(image)
    axes[0].set_title("Original image")

    axes[1].imshow(channel, cmap="Greys")
    axes[1].set_title("Channel")

    axes[2].imshow(binarized_channel, cmap="Greys")
    axes[2].set_title("Binarized channel")

    # for a in axes:
    #     a.axis('off')

    fig.tight_layout()
    plt.show()


def merge_ihc(h_channel, dab_channel):
    h_scaled = rescale_intensity(h_channel, out_range=(0, 1))
    d_scaled = rescale_intensity(dab_channel, out_range=(0, 1))
    zdh = np.dstack((np.zeros_like(h_scaled), d_scaled, h_scaled))
    return zdh


def main(image_path):
    # matrix = np.array([[0.6877, 0.5756, 0.4424],
    #                    [0.3368, 0.4831, 0.8081],
    #                    [0.5858, 0.6110, 0.5325]])
    # inverse = np.linalg.inv(matrix)

    image = imread(image_path)

    h_channel, dab_channel, rest_channel = deconv_ihc(image)
    plt.imshow(binary_adaptive)
    plt.show()
    # try_all(h_channel)
    sure_bg, sure_fg, unknown = get_background_and_foreground(h_channel)
    ihc_image = merge_ihc(h_channel, dab_channel)
    # plot_deconv(image, sure_bg, sure_fg, unknown)
    import pdb
    pdb.set_trace()  # breakpoint daad0538 //
    merged = merge_channels(h_channel, dab_channel)
    merged_scaled = rescale_intensity(h_channel, out_range=(0, 1))
    merged_int = img_as_ubyte(merged_scaled)
    merged_int_3d = cv2.cvtColor(merged_int, cv2.COLOR_GRAY2BGR)
    markers = watershed(merged_int_3d, sure_fg, unknown)
    image[markers == -1] = [255, 0, 0]
    plt.imshow(image)
    plt.show()
    # binarized = binarize(h_channel)
    # plot_binarized(image, h_channel, binarized)
    # # plt.show()
    # # new_h_channel = binarize(h_channel)
    # # plot_binarized(image, h_channel, new_h_channel)

    # segment_channel(merged)
    # plot_merge(image, merged)

    # plot_deconv(image, h_channel, dab_channel, rest_channel)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--image_path',
        type=str,
        help="Ubicación de la imagen",
        required=True
    )
    FLAGS = PARSER.parse_args()
    main(FLAGS.image_path)

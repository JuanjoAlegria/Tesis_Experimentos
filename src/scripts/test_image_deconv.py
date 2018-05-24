import argparse

import numpy as np
from skimage.color import hdx_from_rgb, separate_stains
from skimage.io import imread

import matplotlib.pyplot as plt


def main(image_path):
    # matrix = np.array([[0.6877, 0.5756, 0.4424],
    #                    [0.3368, 0.4831, 0.8081],
    #                    [0.5858, 0.6110, 0.5325]])
    # inverse = np.linalg.inv(matrix)

    ihc_rgb = imread(image_path)
    ihc_hdx = separate_stains(ihc_rgb, hdx_from_rgb)

    fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
    axes = axes.ravel()

    axes[0].imshow(ihc_rgb)
    axes[0].set_title("Original image")

    axes[1].imshow(ihc_hdx[:, :, 0], cmap="Greys")
    axes[1].set_title("Hematoxylin")

    axes[2].imshow(ihc_hdx[:, :, 1], cmap="Greys")
    axes[2].set_title("DAB")

    axes[3].imshow(ihc_hdx[:, :, 2], cmap="Greys")
    axes[3].set_title("Rest")

    for a in axes:
        a.axis('off')

    fig.tight_layout()
    plt.show()

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

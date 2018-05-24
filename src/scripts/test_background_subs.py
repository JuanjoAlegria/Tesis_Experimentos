import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ..utils import cellsDetection
#== Parameters ===========================================================
BLUR = 21
CANNY_THRESH_1 = 10
CANNY_THRESH_2 = 200
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (0.0, 0.0, 1.0)  # In BGR format


#== Processing ===========================================================
def test(image_path):
    image = cv2.imread(image_path)
    lower = np.array([225, 225, 225])
    upper = np.array([255, 255, 255])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    mask = cv2.inRange(image, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)

    import pdb
    pdb.set_trace()  # breakpoint edecf56f //

    foreground = image.copy()
    foreground[mask != 0] = 0
    fig, axes = plt.subplots(1, 2, figsize=(7, 6), sharex=True, sharey=True)
    axes = axes.ravel()

    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original image")

    axes[1].imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Background Removed")

    for a in axes:
        a.axis('off')

    fig.tight_layout()
    plt.show()
    cv2.imwrite("image.jpg", foreground)


def main(image_path):
    #-- Read image -----------------------------------------------------------
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #-- Edge detection -------------------------------------------------------
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    #-- Find contours in edges, sort by area ---------------------------------
    contour_info = []
    import pdb
    pdb.set_trace()  # breakpoint b7cc431c //

    _, contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    #-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    #-- Smooth mask, then blur it --------------------------------------------
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask] * 3)    # Create 3-channel alpha mask

    #-- Blend masked img into MASK_COLOR background --------------------------
    mask_stack = mask_stack.astype(
        'float32') / 255.0          # Use float matrices,
    img = img.astype('float32') / 255.0  # for easy blending

    masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)  # Blend
    # Convert back to 8-bit
    masked = (masked * 255).astype('uint8')

    cv2.imshow('img', masked)                                   # Display
    cv2.waitKey()

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--image_path',
        type=str,
        help="Ubicación de la imagen",
        required=True
    )
    FLAGS = PARSER.parse_args()
    test(FLAGS.image_path)

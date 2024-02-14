import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def display_image_with_polygons(image: np.ndarray, polygons: list):
    """
    Display an image with polygons overlaid.

    Parameters:
    - image: A NumPy array representing the image.
    - polygons: A list of polygons, where each polygon is defined by [xmin, ymin, xmax, ymax].
    """

    # Automatically adjust the image shape if necessary
    if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2] and len(image.shape) == 3:
        image = np.transpose(image, (1, 2, 0))

    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(image)

    # Overlay each polygon
    for xmin, ymin, xmax, ymax in polygons:
        # Create a Rectangle patch
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

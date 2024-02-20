import base64
from pycocotools import mask as mask_utils
import cv2
import numpy as np
from shapely import Polygon


def polygon_to_coco_coordinates(polygon: Polygon):
    """
        Encodes a polygon into a list of coordinates supported by COCO.
    """
    return [coord for xy in polygon.exterior.coords[:-1] for coord in xy]


def polygon_to_coco_rle_mask(polygon: Polygon, tile_height: int, tile_width: int) -> dict:
    """
    Encodes a polygon into an RLE mask.
    """
    contours = np.array(polygon.exterior.coords).reshape((-1, 1, 2)).astype(np.int32)
    binary_mask = np.zeros((tile_height, tile_width), dtype=np.uint8)
    cv2.fillPoly(binary_mask, [contours], 1)
    binary_mask_fortran = np.asfortranarray(binary_mask)
    rle = mask_utils.encode(binary_mask_fortran)
    rle['counts'] = base64.b64encode(rle['counts']).decode('utf-8')  # JSON can't save bytes
    return rle

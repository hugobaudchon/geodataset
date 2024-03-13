import base64
from pathlib import Path
from typing import List

import cv2
import laspy
import numpy as np
import rasterio
import shapely
from matplotlib import pyplot as plt, patches as patches
from matplotlib.colors import ListedColormap
from pycocotools import mask as mask_utils

from rasterio.enums import Resampling
from shapely import Polygon, box, MultiPolygon
from shapely.ops import transform


def polygon_to_coco_coordinates(polygon: Polygon or MultiPolygon):
    """
        Encodes a polygon into a list of coordinates supported by COCO.
    """
    if type(polygon) is Polygon:
        coordinates = [[coord for xy in polygon.exterior.coords[:-1] for coord in xy]]
    elif type(polygon) is MultiPolygon:
        coordinates = []
        for geom in polygon.geoms:
            coordinates.append([coord for xy in geom.exterior.coords[:-1] for coord in xy])
    else:
        raise Exception(f"The polygon is not a shapely.Polygon or shapely.MultiPolygon. It is a {type(polygon)}.")

    return coordinates


def polygon_to_coco_rle_mask(polygon: Polygon or MultiPolygon, tile_height: int, tile_width: int) -> dict:
    """
    Encodes a Polygon or MultiPolygon object into an RLE mask.
    """
    binary_mask = np.zeros((tile_height, tile_width), dtype=np.uint8)

    # Function to process each polygon
    def process_polygon(p):
        contours = np.array(p.exterior.coords).reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(binary_mask, [contours], 1)

    if isinstance(polygon, Polygon):
        process_polygon(polygon)
    elif isinstance(polygon, MultiPolygon):
        for polygon in polygon.geoms:
            process_polygon(polygon)
    else:
        raise TypeError("Geometry must be a Polygon or MultiPolygon")

    binary_mask_fortran = np.asfortranarray(binary_mask)
    rle = mask_utils.encode(binary_mask_fortran)

    # Encode the counts to base64 to be able to store it in a json file
    rle['counts'] = base64.b64encode(rle['counts']).decode('utf-8')  # JSON can't save bytes
    return rle


def rle_segmentation_to_bbox(segmentation: dict) -> box:
    """
    Calculates the bounding box from a binary mask.
    """
    # Decode the counts from base64
    if 'counts' in segmentation and isinstance(segmentation['counts'], str):
        counts = base64.b64decode(segmentation['counts'])
        segmentation['counts'] = counts

    mask = mask_utils.decode(segmentation)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    return box(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)


def polygon_segmentation_to_bbox(segmentation: list) -> box:
    """
    Calculates the bounding box from a polygon.
    """
    polygons = []
    for polygon_coords in segmentation:
        # Reshape the flat list of coords into a list of (x, y) tuples
        it = iter(polygon_coords)
        polygon = Polygon([(x, y) for x, y in zip(it, it)])
        polygons.append(polygon)

    # Create a MultiPolygon from the list of Polygon objects
    multipolygon = MultiPolygon(polygons)
    return box(*multipolygon.bounds)


def get_tiles_array(tiles: list, tile_coordinate_step: int):
    """
    :param tiles: a list of Tile
    :param tile_coordinate_step: usually = (1 - tile_overlap) * tile_size
    :return: a binary grid array with 0 if no tile and 1 if tile
    """
    numpy_coordinates = [(int(tile.row / tile_coordinate_step),
                          int(tile.col / tile_coordinate_step)) for tile in tiles]

    # Determine dimensions based on coordinates if not provided
    max_x = max(numpy_coordinates, key=lambda coord: coord[0])[0] if numpy_coordinates else 0
    max_y = max(numpy_coordinates, key=lambda coord: coord[1])[1] if numpy_coordinates else 0
    dimensions = (max_x + 1, max_y + 1)

    # Create an array of zeros with the determined dimensions
    array = np.zeros(dimensions, dtype=int)

    # Mark the coordinates in the array
    for x, y in numpy_coordinates:
        if x < dimensions[0] and y < dimensions[1]:
            array[x, y] = 1

    return array


def read_raster(path: Path, ground_resolution: float = None, scale_factor: float = None):
    assert not (ground_resolution and scale_factor), ("Both a ground_resolution and a scale_factor were provided."
                                                      " Please only specify one.")

    ext = path.suffix
    if ext in ['.tif', '.png', '.jpg']:
        with rasterio.open(path) as src:
            # Check if the CRS uses meters as units
            crs = src.crs
            if ground_resolution:
                if crs:
                    if crs.is_projected:
                        current_x_resolution = src.transform[0]
                        current_y_resolution = -src.transform[4]
                        # Calculate scale factors to achieve the specified ground_resolution
                        x_scale_factor = current_x_resolution / ground_resolution
                        y_scale_factor = current_y_resolution / ground_resolution
                        print(f'Rescaling the raster with x_scale_factor={x_scale_factor}'
                              f' and x_scale_factor={y_scale_factor}'
                              f' to match ground_resolution={ground_resolution}...')
                    else:
                        raise Exception("The CRS of the raster is not projected (in meter units),"
                                        " so the ground_resolution cannot be applied.")
                else:
                    raise Exception("The raster doesn't have a CRS, so the ground_resolution cannot be applied.")
            elif scale_factor:
                x_scale_factor = scale_factor
                y_scale_factor = scale_factor
            else:
                x_scale_factor = 1
                y_scale_factor = 1

            data = src.read(
                out_shape=(src.count,
                           int(src.height * y_scale_factor),
                           int(src.width * x_scale_factor)),
                resampling=Resampling.bilinear)

            if src.transform:
                # scale image transform
                new_transform = src.transform * src.transform.scale(
                    (src.width / data.shape[-1]),
                    (src.height / data.shape[-2]))
            else:
                new_transform = None

            new_profile = src.profile
            new_profile.update(transform=new_transform,
                               driver='GTiff',
                               height=data.shape[-2],
                               width=data.shape[-1])
    else:
        raise Exception(f'Data format {ext} not supported yet.')

    return data, new_profile, x_scale_factor, y_scale_factor


def read_point_cloud(path: Path):
    ext = path.suffix
    if ext in ['.las', '.laz']:
        with laspy.open(str(path)) as pc_file:
            data = pc_file.read()
            crs = None
            transform = None
            for vlr in data.vlrs:
                if isinstance(vlr, laspy.vlrs.known.WktCoordinateSystemVlr):
                    crs = vlr.parse_crs()
    else:
        raise Exception(f'Data format {ext} not supported yet.')

    return data, crs, transform


def display_image_with_polygons(image: np.ndarray, polygons: List[shapely.Polygon]):
    """
    Display an image with polygons overlaid.

    Parameters:
    - image: A NumPy array representing the image.
    - polygons: A list of polygons.
    """

    # Automatically adjust the image shape if necessary
    if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2] and len(image.shape) == 3:
        image = np.transpose(image, (1, 2, 0))

    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(image)

    # Overlay each polygon
    for p in polygons:
        # Extract exterior coordinates from each shapely.Polygon
        x, y = p.exterior.xy
        # Create a list of (x, y) tuples for matplotlib patches.Polygon
        vertices = list(zip(x, y))
        # Add the patch to the Axes
        patch = patches.Polygon(vertices, edgecolor='r', facecolor='none')
        ax.add_patch(patch)

    plt.show()


def save_aois_tiles_picture(aois_tiles: dict[str, list], save_path: Path, tile_coordinate_step: int):
    """
    Display the original array of 1s and 0s with specific coordinates highlighted.

    Parameters:
    :param aois_tiles: a dict of format {aoi: list[Tile]}, with aoi being 'train', 'valid'... Must have 'all' key.
    :param save_path: the output Path.
    :param tile_coordinate_step: usually = (1 - tile_overlap) * tile_size

    """

    # Copy the original array to avoid altering it
    display_array = get_tiles_array(tiles=aois_tiles['all'], tile_coordinate_step=tile_coordinate_step)

    # Colors - generate dynamically based on the number of AOIs
    colors = plt.cm.tab10(np.linspace(0, 1, len(aois_tiles)))

    # Create a custom color map - starting with a base color
    base_cmap = ['#BBBBBB', '#555555']  # Light gray for 0, dark gray for unassigned tiles (1)
    color_labels = ['Background', 'Unassigned Tiles']

    for idx, aoi in enumerate(aois_tiles, start=1):
        if aoi == 'all':
            continue

        for tile in aois_tiles[aoi]:
            tile_x_numpy = int(tile.row / tile_coordinate_step)
            tile_y_numpy = int(tile.col / tile_coordinate_step)
            display_array[tile_x_numpy, tile_y_numpy] = idx + 1

        base_cmap.append(colors[idx-1])
        color_labels.append(aoi)  # Use the dict key as the label

    custom_cmap = ListedColormap(base_cmap)

    plt.figure(figsize=(10, 10))
    plt.imshow(display_array, cmap=custom_cmap, interpolation='nearest')

    # Create a color bar with a tick and label for each AOI
    ticks = list(range(len(base_cmap)))
    cbar = plt.colorbar(ticks=ticks)
    cbar.set_ticklabels(color_labels)
    plt.clim(0, len(base_cmap) - 1)  # Adjust color limit to include all AOI colors

    plt.savefig(save_path)


def strip_all_extensions(path: Path):
    """
    Strips all extensions from a given Path object or path string and returns the base name.
    """
    p = Path(path)
    while p.suffix:
        p = p.with_suffix('')
    return p.name


def generate_label_coco(polygon: shapely.Polygon,
                        tile_height: int,
                        tile_width: int,
                        tile_id: int,
                        use_rle_for_labels: bool,
                        category_id: int or None,
                        other_attributes_dict: dict or None) -> dict:
    if use_rle_for_labels:
        # Convert the polygon to a COCO RLE mask
        segmentation = polygon_to_coco_rle_mask(polygon=polygon,
                                                tile_height=tile_height,
                                                tile_width=tile_width)
    else:
        # Convert the polygon's exterior coordinates to the format expected by COCO
        segmentation = polygon_to_coco_coordinates(polygon=polygon)

    # Calculate the area of the polygon
    area = polygon.area

    # Get the bounding box in COCO format: [x, y, width, height]
    bbox = list(polygon.bounds)
    bbox_coco_format = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

    # Generate COCO annotation data from each associated label
    coco_annotation = {
        "segmentation": segmentation,
        "is_rle_format": use_rle_for_labels,
        "area": area,
        "iscrowd": 0,  # Assuming this polygon represents a single object (not a crowd)
        "image_id": tile_id,
        "bbox": bbox_coco_format,
        "category_id": category_id,
        "other_attributes": other_attributes_dict
    }

    return coco_annotation


def apply_affine_transform(geom: shapely.geometry, affine: rasterio.Affine):
    return transform(lambda x, y: affine * (x, y), geom)

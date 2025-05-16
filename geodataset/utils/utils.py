import glob
import json
import os
import tempfile

import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path
import random
from typing import List, Union, Dict

import cv2
import laspy
import numpy as np
import pandas as pd
import rasterio
import shapely
from matplotlib import pyplot as plt, patches as patches
from matplotlib.colors import ListedColormap
from pycocotools import mask as mask_utils
import geopandas as gpd
from pycocotools.coco import COCO
from rasterio import CRS, MemoryFile

from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform
from rasterio.windows import Window
from shapely import Polygon, box, MultiPolygon, make_valid
from shapely.affinity import affine_transform
from shapely.ops import transform
from tqdm import tqdm

from geodataset.utils.file_name_conventions import CocoNameConvention


def polygon_to_coco_coordinates_segmentation(polygon: Polygon or MultiPolygon):
    """
        Encodes a polygon into a list of coordinates supported by COCO.

        Parameters
        ----------
        polygon: shapely.Polygon or shapely.MultiPolygon
            The polygon to encode.

        Returns
        -------
        list
            A list of coordinates in the format expected by COCO.
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


def polygon_to_mask(polygon: Polygon or MultiPolygon, array_height: int, array_width: int) -> np.ndarray:
    """
    Encodes a Polygon or MultiPolygon object into a binary mask.

    Parameters
    ----------
    polygon: Polygon or MultiPolygon
        The polygon to encode.
    array_height: int
        The height of the array to encode the polygon into.
    array_width: int
        The width of the array to encode the polygon into.

    Returns
    -------
    np.ndarray
        A binary mask of the polygon.
    """
    binary_mask = np.zeros((array_height, array_width), dtype=np.uint8)

    # Function to process each polygon
    def process_polygon(p):
        # Fill the exterior of the polygon
        exterior_contour = np.array(p.exterior.coords).reshape((-1, 1, 2)).astype(np.int32)
        if len(exterior_contour) == 0:
            return
        cv2.fillPoly(binary_mask, [exterior_contour], 1)

        # Fill each interior ring (hole) with 0
        for interior in p.interiors:
            interior_contour = np.array(interior.coords).reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(binary_mask, [interior_contour], 0)

    if isinstance(polygon, Polygon):
        process_polygon(polygon)
    elif isinstance(polygon, MultiPolygon):
        for poly in polygon.geoms:
            process_polygon(poly)
    else:
        raise TypeError(f"Geometry must be a Polygon or MultiPolygon. Got {type(polygon)}.")

    return binary_mask


def polygon_to_coco_rle_segmentation(polygon: Polygon or MultiPolygon,
                                     tile_height: int,
                                     tile_width: int) -> dict or List[dict]:
    """
    Encodes a Polygon or MultiPolygon object into a COCO annotation RLE mask.

    Parameters
    ----------
    polygon: Polygon or MultiPolygon
        The polygon to encode.
    tile_height: int
        The height of the tile the polygon is in.
    tile_width: int
        The width of the tile the polygon is in.

    Returns
    -------
    dict
        A COCO RLE mask segmentation.
    """

    if isinstance(polygon, Polygon):
        polygons = [polygon]
    elif isinstance(polygon, MultiPolygon):
        polygons = list(polygon.geoms)
    else:
        raise ValueError(f"Input must be a Shapely Polygon or MultiPolygon, not {type(polygon)}.")

        # Convert each polygon to COCO format [x1,y1,x2,y2,...]
    coco_coords = []
    for polygon in polygons:
        coords = np.array(polygon.exterior.coords)[:-1]  # Remove last point (same as first)
        coords = coords.flatten().tolist()
        coco_coords.append(coords)

    # Convert directly to RLE using frPyObjects
    rle = mask_utils.frPyObjects(coco_coords, tile_height, tile_width)
    rle = mask_utils.merge(rle)  # Merge all polygons into single RLE

    # Convert to Python native types for JSON serialization
    rle['counts'] = rle['counts'].decode('utf-8')

    return rle


def coco_rle_segmentation_to_mask(rle_segmentation: dict) -> np.ndarray:
    """
    Decodes a COCO annotation RLE segmentation into a binary mask.

    Parameters
    ----------
    rle_segmentation: dict
        The RLE segmentation to decode of a Polygon or MultiPolygon.

    Returns
    -------
    np.ndarray
        A binary mask of the segmentation.
    """

    if isinstance(rle_segmentation['counts'], str):
        rle_segmentation = rle_segmentation.copy()
        rle_segmentation['counts'] = rle_segmentation['counts'].encode('utf-8')

    mask = mask_utils.decode(rle_segmentation)
    return mask


def coco_rle_segmentation_to_bbox(rle_segmentation: dict) -> box:
    """
    Calculates the bounding box from a COCO annotation RLE segmentation.

    Parameters
    ----------
    rle_segmentation: dict
        The RLE segmentation to decode.

    Returns
    -------
    shapely.box
        A shapely box representing the bounding box of the segmentation.
    """

    mask = coco_rle_segmentation_to_mask(rle_segmentation)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    return box(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)


def mask_to_polygon(
        mask: np.ndarray,
        simplify_tolerance: float = 1.0,
        min_contour_points: int = 3,
        remove_rings: bool = False,
        remove_small_geoms: int or None = 10) -> Union[Polygon, MultiPolygon]:
    """
    Converts a binary mask to simplified shapely Polygon(s).

    Parameters
    ----------
    mask: np.ndarray
        The mask to convert, in HW format
    simplify_tolerance: float
        The tolerance for simplifying polygons
    min_contour_points: int
        Minimum number of points required for a valid contour
    remove_rings: bool
        Whether to remove inner rings (holes) from the polygons
    remove_small_geoms: int or None
        Remove small geoms with less than this area from the MultiPolygon

    Returns
    -------
    Union[Polygon, MultiPolygon]
        Simplified polygon(s) representing the mask
    """
    if mask.ndim != 2:
        raise ValueError("Mask must be in HW format (2D array).")

    # Pad the mask
    padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)

    # Find contours
    contours, _ = cv2.findContours(
        padded_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []

    for contour in contours:
        if len(contour) >= min_contour_points:
            # Convert to (x,y) format and adjust for padding
            points = [(x[0][0] - 1, x[0][1] - 1) for x in contour]
            poly = Polygon(points)

            # Simplify if requested
            if simplify_tolerance > 0:
                poly = poly.simplify(
                    tolerance=simplify_tolerance,
                    preserve_topology=True
                )

            if not poly.is_empty:
                polygons.append(poly)

    if not polygons:
        return Polygon()
    elif len(polygons) == 1:
        polygon = polygons[0]
    else:
        polygon = MultiPolygon(polygons)
        if remove_small_geoms:
            polygon = remove_small_geoms_from_multipolygon(polygon, remove_small_geoms)

    if remove_rings:
        polygon = remove_rings_from_polygon(polygon)
    if not polygon.is_valid:
        polygon = make_valid(polygon)

    polygon = fix_geometry_collection(polygon)

    if not isinstance(polygon, (Polygon, MultiPolygon)):
        # If everything failed, return an empty Polygon
        return Polygon()
    else:
        return polygon


def remove_small_geoms_from_multipolygon(multi_polygon: MultiPolygon, min_area: int):
    """
    Removes small geometries from a MultiPolygon.

    Parameters
    ----------
    multi_polygon: MultiPolygon
        The MultiPolygon to process.
    min_area: int
        The minimum area for a geometry to be kept.

    Returns
    -------
    MultiPolygon
        The MultiPolygon with small geometries removed.
    """
    geoms = [geom for geom in multi_polygon.geoms if geom.area >= min_area]
    return MultiPolygon(geoms)


def remove_rings_from_polygon(polygon: Polygon or MultiPolygon):
    """
    Removes inner rings (holes) from a polygon.

    Parameters
    ----------
    polygon: Polygon or MultiPolygon
        The polygon to process.

    Returns
    -------
    Polygon or MultiPolygon
        The polygon with inner rings removed.
    """
    if isinstance(polygon, Polygon):
        return Polygon(polygon.exterior)
    elif isinstance(polygon, MultiPolygon):
        geoms = [Polygon(geom.exterior) for geom in polygon.geoms]
        return MultiPolygon(geoms)
    else:
        raise ValueError("Input must be a Polygon or MultiPolygon.")


def coco_coordinates_segmentation_to_polygon(segmentation: list) -> Polygon or MultiPolygon:
    """
    Converts a list of polygon coordinates in COCO format to a shapely Polygon or MultiPolygon.

    Parameters
    ----------
    segmentation: list
        A list of coordinates in the format expected by COCO.

    Returns
    -------
    Polygon
        A shapely Polygon object representing the outer boundary of the polygon.
    """
    geoms = []
    for polygon_coords in segmentation:
        # Reshape the flat list of coords into a list of (x, y) tuples
        it = iter(polygon_coords)
        geom = Polygon([(x, y) for x, y in zip(it, it)])
        geoms.append(geom)

    # Create a MultiPolygon from the list of Polygon objects
    if len(geoms) == 1:
        return geoms[0]
    else:
        return MultiPolygon(geoms)


def fix_geometry_collection(geometry: shapely.Geometry):
    """
    Fixes a GeometryCollection into a Polygon or MultiPolygon by converting LineStrings to Polygons if they are closed.
    """
    if geometry.geom_type == 'GeometryCollection':
        final_geoms = []
        # Iterate through each geometry in the collection
        for geom in geometry.geoms:
            if geom.geom_type == 'Polygon':
                final_geoms.append(geom)
            elif geom.geom_type == 'MultiPolygon':
                final_geoms.extend(geom.geoms)
            elif geom.geom_type == 'LineString':
                # Check if the LineString is closed and can be considered a polygon
                if geom.is_ring:
                    # Convert the LineString to a Polygon
                    final_geoms.append(Polygon(geom))
        return MultiPolygon(final_geoms)
    else:
        return geometry


def coco_coordinates_segmentation_to_bbox(segmentation: list) -> box:
    """
    Calculates the bounding box from a polygon list of coordinates in COCO format.

    Parameters
    ----------
    segmentation: list
        A list of coordinates in the format expected by COCO.

    Returns
    -------
    shapely.box
        A shapely box representing the bounding box of the polygon.
    """
    polygon = coco_coordinates_segmentation_to_polygon(segmentation)
    return box(*polygon.bounds)


def coco_rle_segmentation_to_polygon(
        rle_segmentation,
        simplify_tolerance: float = 1.0,
        min_contour_points: int = 3):
    """
    Decodes a COCO annotation RLE segmentation into a shapely Polygon or MultiPolygon.

    Parameters
    ----------
    rle_segmentation: dict
        The RLE segmentation to decode.
    simplify_tolerance: float
        The tolerance for simplifying polygons.
    min_contour_points: int
        Minimum number of points required for a valid contour.

    Returns
    -------
    Polygon or MultiPolygon
        A shapely Polygon or MultiPolygon representing the segmentation.
    """
    # Decode the RLE
    mask = coco_rle_segmentation_to_mask(rle_segmentation)
    return mask_to_polygon(mask, simplify_tolerance, min_contour_points)


def decode_coco_segmentation(coco_annotation: dict, output_type: str):
    """
    Decodes a COCO annotation segmentation into a shapely Polygon or MultiPolygon.

    Parameters
    ----------
    coco_annotation: dict
        The segmentation to decode.
    output_type: str
        The desired output type. Can be 'polygon', 'bbox' or 'mask'.

    Returns
    -------
    Polygon or MultiPolygon or box or np.ndarray
        The decoded segmentation.
    """
    segmentation = coco_annotation['segmentation']

    if ('is_rle_format' in coco_annotation and coco_annotation['is_rle_format']) or isinstance(segmentation, dict):
        # Compressed RLE format
        if output_type == 'polygon':
            return coco_rle_segmentation_to_polygon(segmentation)
        elif output_type == 'bbox':
            if 'bbox' in coco_annotation:
                # Directly use the provided bbox
                bbox_coco = coco_annotation['bbox']
                bbox = box(*[bbox_coco[0], bbox_coco[1], bbox_coco[0] + bbox_coco[2], bbox_coco[1] + bbox_coco[3]])
            else:
                bbox = coco_rle_segmentation_to_bbox(segmentation)
            return bbox
        elif output_type == 'mask':
            return coco_rle_segmentation_to_mask(segmentation)
        else:
            raise ValueError(f"Output type '{output_type}' not supported. Must be 'polygon', 'bbox' or 'mask'.")
    elif ('is_rle_format' in coco_annotation and not coco_annotation['is_rle_format']) or isinstance(segmentation, list):
        # Coordinates format
        if output_type == 'polygon':
            return coco_coordinates_segmentation_to_polygon(segmentation)
        elif output_type == 'bbox':
            if 'bbox' in coco_annotation:
                # Directly use the provided bbox
                bbox_coco = coco_annotation['bbox']
                bbox = box(*[bbox_coco[0], bbox_coco[1], bbox_coco[0] + bbox_coco[2], bbox_coco[1] + bbox_coco[3]])
            else:
                bbox = coco_coordinates_segmentation_to_bbox(segmentation)
            return bbox
        elif output_type == 'mask':
            polygon = coco_coordinates_segmentation_to_polygon(segmentation)
            return polygon_to_mask(polygon, coco_annotation['height'], coco_annotation['width'])
        else:
            raise ValueError(f"Output type '{output_type}' not supported. Must be 'polygon', 'bbox' or 'mask'.")
    else:
        raise NotImplementedError("Could not find the segmentation type (RLE vs polygon coordinates).")


def get_tiles_array(tiles: list,
                    tile_coordinate_step: int):
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


def try_cast_multipolygon_to_polygon(geometry):
    if isinstance(geometry, Polygon):
        return geometry
    elif isinstance(geometry, MultiPolygon):
        polygons = list(geometry.geoms)
        if len(polygons) == 1:
            return Polygon(polygons[0])
        else:
            return None
    else:
        # Return None if the geometry is neither Polygon nor MultiPolygon
        return None


def get_utm_crs(lon, lat):
    """
    Determine the UTM CRS for a given longitude and latitude.
    """
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return CRS.from_epsg(code=32600 + zone)  # Northern Hemisphere
    else:
        return CRS.from_epsg(code=32700 + zone)  # Southern Hemisphere


def read_raster(path: Path, ground_resolution: float = None, scale_factor: float = None, temp_dir: Path = './tmp'):
    """
    Open a raster file and return a view (WarpedVRT) that applies the given scaling.

    Parameters
    ----------
    path: Path
        The path to the raster file.
    ground_resolution: float, optional
        The desired ground resolution in meters.
    scale_factor: float, optional
        The scale factor to apply to the raster.

    Returns
    -------
    vrt: WarpedVRT
        A virtual dataset that you can use to read windows on the fly.
    profile: dict
        An updated profile of the raster reflecting the scaling.
    x_scale_factor, y_scale_factor: float
        The scale factors applied in the x and y directions.
    """

    print("Reading raster...")

    os.environ["GDAL_CACHEMAX"] = "10240" # Set a temporary 10 GB max cache size to avoid memory issues with very large rasters

    assert not (ground_resolution and scale_factor), (
        "Both a ground_resolution and a scale_factor were provided. Please only specify one."
    )

    ext = path.suffix.lower()
    if ext not in ['.tif', '.png', '.jpg']:
        raise Exception(f'Data format {ext} not supported yet.')

    src = rasterio.open(path)
    crs = src.crs

    # Determine the scale factors
    if ground_resolution:
        if crs is not None and not crs.is_projected:
            # Calculate the centroid of the raster
            bounds = src.bounds
            centroid_lon = (bounds.left + bounds.right) / 2.0
            centroid_lat = (bounds.top + bounds.bottom) / 2.0

            # Determine a suitable UTM CRS based on the centroid
            target_crs = get_utm_crs(centroid_lon, centroid_lat)
            print(f"Raster is not projected. Reprojecting to {target_crs.to_string()} based on Raster centroid ({centroid_lon}, {centroid_lat}).")
        elif crs is not None:
            target_crs = crs
        else:
            raise ValueError("The Raster doesn't have a CRS, please use scale_factor instead of ground_resolution.")

        new_transform, new_width, new_height = calculate_default_transform(
            crs, target_crs, src.width, src.height, *src.bounds, resolution=ground_resolution
        )

        # Determine current resolution (assumed equal in x and y for simplicity)
        x_scale_factor = new_width / src.width
        y_scale_factor = new_height / src.height

        print(f'Rescaling the raster with x_scale_factor={x_scale_factor} '
              f'and y_scale_factor={y_scale_factor} to match ground_resolution={ground_resolution}...')
    elif scale_factor:
        x_scale_factor = scale_factor
        y_scale_factor = scale_factor
        target_crs = crs
        new_width = int(src.width * x_scale_factor)
        new_height = int(src.height * y_scale_factor)
        new_transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )
    else:
        x_scale_factor = 1
        y_scale_factor = 1
        target_crs = crs
        new_width = src.width
        new_height = src.height
        new_transform = src.transform

    profile = src.profile.copy()
    profile.update({
        "driver": "GTiff",
        "height": new_height,
        "width": new_width,
        "transform": new_transform,
        "crs": target_crs,
        "BIGTIFF": "IF_NEEDED",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256
    })

    # Estimate memory footprint
    itemsize = np.dtype(profile['dtype']).itemsize
    nbands = profile.get('count', 1)
    expected_bytes = new_width * new_height * nbands * itemsize
    max_in_mem_gb = 10
    max_in_mem_bytes = max_in_mem_gb * 1024 ** 3  # 10GB

    print(f"Expected raster size in memory: {expected_bytes / (1024 ** 3):.2f} GB.")

    # If the expected size is less than 10GB, load into memory using a MemoryFile;
    # otherwise, write to a temporary file.
    if expected_bytes < max_in_mem_bytes:
        print("Loading raster in memory (while resampling to scale_factor/ground_resolution)...")
        if crs != target_crs:
            data = WarpedVRT(
                src,
                crs=target_crs,
                width=new_width,
                height=new_height,
                transform=new_transform,
                resampling=Resampling.bilinear
            ).read()
        else:
            # Faster, but only works if CRS is kept the same
            data = src.read(
                out_shape=(src.count,
                           int(src.height * y_scale_factor),
                           int(src.width * x_scale_factor)),
                resampling=Resampling.bilinear
            )
        memfile = MemoryFile()
        with memfile.open(**profile) as mem_ds:
            mem_ds.write(data)
        dataset = memfile.open()
        temp_path = None
    else:
        if x_scale_factor != 1 or y_scale_factor != 1:
            print(f"The resampled Raster would be more than {max_in_mem_gb} GB in memory, writing to temporary file on disk instead...")
            vrt = WarpedVRT(
                src,
                crs=target_crs,
                width=new_width,
                height=new_height,
                transform=new_transform,
                resampling=Resampling.bilinear
            )

            Path(temp_dir).mkdir(exist_ok=True, parents=True)
            temp = tempfile.NamedTemporaryFile(suffix=".tif", prefix="resampled_raster_", delete=False, dir=temp_dir)
            temp_path = temp.name
            temp.close()

            print(f"Temporary re-sampled Raster will be at at {temp_path}.")

            # Adjust the profile: disable tiling and enable BIGTIFF for large files.
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
            profile.update({"tiled": False, "BIGTIFF": "YES"})

            # Define the target chunk size in bytes (e.g. 1GB)
            chunk_bytes = 1 * 1024 ** 3
            bytes_per_row = new_width * nbands * itemsize
            rows_per_chunk = max(1, int(chunk_bytes // bytes_per_row))
            print(f"Processing {rows_per_chunk} rows per chunk "
                  f"(each chunk ≈{rows_per_chunk * bytes_per_row / (1024 ** 3):.2f} GB).")

            with rasterio.open(temp_path, "w", **profile) as dst:
                for row in tqdm(range(0, new_height, rows_per_chunk), desc="Writing chunks to temp file"):
                    h = min(rows_per_chunk, new_height - row)
                    window = Window(col_off=0, row_off=row, width=new_width, height=h)
                    data_block = vrt.read(window=window)
                    dst.write(data_block, window=window)
                    # Try to flush the cache; if not available, ignore.
                    try:
                        dst.flush_cache()
                    except AttributeError:
                        pass
            dataset = rasterio.open(temp_path)
            warnings.warn(f"Raster too large for in-memory load. Temporary file created at {temp_path}. You might want to delete it after use.")
        else:
            # If no resampling is needed, just open the file directly, no need to duplicate it to temp file
            dataset = src
            temp_path = None

    return dataset, profile, x_scale_factor, y_scale_factor, temp_path


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

    Parameters
    ----------
    image: np.ndarray
        The image to display.
    polygons: List[shapely.Polygon]
        The polygons to overlay on the image.
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

    Parameters
    ----------
    aois_tiles: dict
        A dictionary of format {aoi: list[Tile]}, with aoi being 'train', 'valid'... Must have 'all' key.
    save_path: Path
        The output Path.
    tile_coordinate_step: int
        Usually = (1 - tile_overlap) * tile_size
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

        base_cmap.append(colors[idx - 1])
        color_labels.append(aoi)  # Use the dict key as the label

    custom_cmap = ListedColormap(base_cmap)

    plt.figure(figsize=(10, 10))
    plt.imshow(display_array, cmap=custom_cmap, interpolation='nearest')

    # Create a color bar with a tick and label for each AOI
    ticks = list(range(len(base_cmap)))
    plt.clim(-0.5, len(base_cmap) - 0.5)  # Adjust color limit to include all AOI colors
    cbar = plt.colorbar(ticks=np.arange(len(base_cmap)))
    cbar.set_ticklabels(color_labels)
    plt.savefig(save_path)


def strip_all_extensions_and_path(path: str or Path):
    """
    Strips all extensions from a given Path object or path string and returns the base name.

    Parameters
    ----------
    path: str or Path
        The path to strip extensions from.

    Returns
    -------
    str
        The base name of the file, with all extensions removed.
    """
    p = Path(path)
    while p.suffix:
        p = p.with_suffix('')
    return p.name


def convert_polygons_to_pixel_coordinates(gdf: gpd.GeoDataFrame, tiles_paths_column: str) -> gpd.GeoDataFrame:
    """
    Convert polygon geometries in a GeoDataFrame from CRS coordinates to pixel coordinates
    using the affine transform of the associated raster tiles.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing polygon geometries. It must include a column with paths to
        raster tiles.
    tiles_paths_column : str
        The name of the column in `gdf` that contains the file path for each tile.

    Returns
    -------
    geopandas.GeoDataFrame
        A new GeoDataFrame in which the polygon geometries have been transformed to the
        pixel coordinate system of their respective tile. The CRS is set to None since pixel
        coordinates are not georeferenced.
    """

    print('Converting polygons from CRS to pixel coordinates...')

    processed_groups = []

    # Process each unique tile path
    for tile_path in gdf[tiles_paths_column].unique():
        try:
            with rasterio.open(tile_path) as src:
                # Check for necessary spatial information
                if not src.crs or not src.transform:
                    warnings.warn(f"Tile {tile_path} is missing a CRS or transform. Skipping this tile.")
                    continue

                tile_crs = src.crs
                # Compute the inverse transform to go from CRS -> pixel coordinates
                inv_transform = ~src.transform
                affine_params = [inv_transform.a, inv_transform.b, inv_transform.d, inv_transform.e,
                                 inv_transform.c, inv_transform.f]

                # Subset the GeoDataFrame rows corresponding to the current tile
                tile_gdf = gdf[gdf[tiles_paths_column] == tile_path].copy()

                # If the GeoDataFrame has a CRS and it does not match the tile's CRS,
                # reproject the geometries to the tile's CRS.
                if tile_gdf.crs and tile_gdf.crs != tile_crs:
                    tile_gdf = tile_gdf.to_crs(tile_crs)

                # Apply the inverse affine transform to each geometry
                tile_gdf['geometry'] = tile_gdf['geometry'].apply(
                    lambda geom: affine_transform(geom, affine_params)
                )

                # Pixel coordinates are not georeferenced, so we remove the CRS.
                tile_gdf.set_crs(None, inplace=True, allow_override=True)

                processed_groups.append(tile_gdf)
        except Exception as e:
            warnings.warn(f"Error processing tile {tile_path}: {e}")

    # Concatenate the processed groups into a single GeoDataFrame
    if processed_groups:
        result_gdf = gpd.GeoDataFrame(pd.concat(processed_groups, ignore_index=True))
        result_gdf.crs = None
    else:
        result_gdf = gpd.GeoDataFrame(columns=gdf.columns)

    return result_gdf


class COCOGenerator:
    """
    A class to generate a COCO dataset from a list of tiles and their associated polygons.
    After instantiating the class, the :meth:`generate_coco` method should be used to generate and save the COCO
    dataset.

    Parameters
    ----------
    description: str
        A description of the COCO dataset.
    tiles_paths: List[Path]
        A list of paths to the tiles/images.
    polygons: List[List[Polygon]]
        A list of lists of polygons associated with each tile.
    scores: List[List[float or None]] or None
        A list of lists of scores associated with each polygon.
    categories: List[List[str or int]] or None
        A list of lists of categories (str or int) associated with each polygon.
    other_attributes: List[List[Dict]] or None
        A list of lists of dictionaries of other attributes associated with each polygon.
        Such a dict could be::

            {
                'attribute1': value1,
                'attribute2': value2
            }

        .. raw:: html

            <br>

        **IMPORTANT**: the 'score' attribute is reserved for the score associated with the polygon.
    output_path: Path
        The path to save the COCO dataset JSON file (should have .json extension).
    use_rle_for_labels: bool
        Whether to use RLE encoding for the labels or not. If False, the polygon's exterior coordinates will be used.
        RLE Encoding takes less space on disk but takes more time to encode.
    n_workers: int
        The number of workers to use for parallel processing.
    coco_categories_list: List[dict] or None
        A list of category dictionaries in COCO format.

        If provided, category ids for the annotations in the final COCO file
        will be determined by matching the category name (defined by 'main_label_category_column_name' parameter) of
        each polygon with the categories names in coco_categories_list.

        If a polygon has a category that is not in this list, its category_id will be set to None in its COCO annotation.

        If 'main_label_category_column_name' is not provided, but 'coco_categories_list' is a single
        coco category dictionary, then it will be used for all annotations automatically.

        If 'coco_categories_list' is None, the categories ids will be automatically generated from the
        unique categories found in the 'main_label_category_column_name' column.

        .. raw:: html

            <br>

        To assign a category_id to a polygon, the code will check the 'name' and 'other_names' fields of the categories.

        .. raw:: html

            <br>

        **IMPORTANT**: It is strongly advised to provide this list if you want to have consistent category ids across
        multiple COCO datasets.

        .. raw:: html

            <br>

        Exemple of 2 categories, one being the parent of the other::

            [{
                "id": 1,
                "name": "Pinaceae",
                "other_names": [],
                "supercategory": null
            },
            {
                "id": 2,
                "name": "Picea",
                "other_names": ["PIGL", "PIMA", "PIRU"],
                "supercategory": 1
            }]
    """
    def __init__(self,
                 description: str,
                 tiles_paths: List[Path],
                 polygons: List[List[Polygon]],
                 scores: List[List[float or None]] or None,
                 categories: List[List[str or int]] or None,
                 other_attributes: List[List[Dict]] or None,
                 output_path: Path,
                 use_rle_for_labels: bool,
                 n_workers: int,
                 coco_categories_list: List[dict] or None):
        self.description = description
        self.tiles_paths = tiles_paths
        self.polygons = polygons
        self.scores = scores
        self.categories = categories
        self.other_attributes = other_attributes
        self.output_path = output_path
        self.use_rle_for_labels = use_rle_for_labels
        self.n_workers = n_workers
        self.coco_categories_list = coco_categories_list

        assert len(self.tiles_paths) == len(self.polygons), "The number of tiles and polygons must be the same."
        if self.scores:
            assert len(self.tiles_paths) == len(self.scores), "The number of tiles and scores must be the same."
            assert all(score is not None for score in self.scores), "If passed, all values in scores must be non-None."
        if self.categories:
            assert len(self.tiles_paths) == len(self.categories), "The number of tiles and categories must be the same."
        else:
            # If no categories are provided, set all categories to 'NoCategory', except if there is only one
            # coco category provided, in which case we use that one for all annotations.
            self.categories = [[coco_categories_list[0]['name']
                                if (type(coco_categories_list) is list and len(coco_categories_list) == 1)
                                else 'NoCategory', ] * len(tile_polygons) for tile_polygons in self.polygons]
        if self.other_attributes:
            assert len(self.tiles_paths) == len(
                self.other_attributes), "The number of tiles and other_attributes must be the same."

        if self.scores and self.other_attributes:
            self.other_attributes = [[dict(self.other_attributes[t][i], score=self.scores[t][i]) for i in
                                     range(len(self.other_attributes[t]))] for t in range(len(self.tiles_paths))]
        elif self.scores:
            self.other_attributes = [[{'score': s} for s in tile_scores] for tile_scores in self.scores]

    @classmethod
    def from_gdf(cls,
                 description: str,
                 gdf: gpd.GeoDataFrame,
                 tiles_paths_column: str,
                 polygons_column: str,
                 scores_column: str or None,
                 categories_column: str or None,
                 other_attributes_columns: List[str] or None,
                 output_path: Path,
                 use_rle_for_labels: bool,
                 n_workers: int,
                 coco_categories_list: List[dict] or None,
                 tiles_paths_order: List[Path] or None = None):
        """
        Instantiate a COCOGenerator from a GeoDataFrame.

        Parameters
        ----------
        description : str
            A description for the COCO dataset.
        gdf : gpd.GeoDataFrame
            A GeoDataFrame containing the annotations. Each row is expected to represent one polygon
            associated with a tile/image.
        tiles_paths_column : str
            The name of the column in the GeoDataFrame that contains the tile/image path.
        polygons_column : str
            The name of the column in the GeoDataFrame that contains the polygon geometry.
        scores_column : str or None, optional
            The name of the column in the GeoDataFrame that contains the score for the polygon.
            If None, scores will not be provided.
        categories_column : str or None, optional
            The name of the column in the GeoDataFrame that contains the category for the polygon.
            If None, categories will not be provided.
        other_attributes_columns : List[str] or None, optional
            A list of column names in the GeoDataFrame whose values should be included as additional
            attributes for each polygon. If None, no additional attributes will be provided.
        output_path : Path
            The path where the generated COCO JSON file will be saved.
        use_rle_for_labels : bool
            Whether to use RLE encoding for the labels or not.
        n_workers : int
            The number of workers to use for parallel processing.
        coco_categories_list : List[dict] or None, optional
            A list of COCO category dictionaries in COCO format. If provided, category ids for the annotations in the
            final COCO file will be determined by matching the category name of each polygon with the categories names
            in coco_categories_list.
        tiles_paths_order : List[Path] or None, optional
            The order in which the tiles should be stored in the COCO file. If None, the order will be determined by
            the order in which the tiles are encountered in the GeoDataFrame. This parameter could be useful if you plan
            to use the same order for multiple COCO datasets (e.g using pycocotools COCOEval between truth and preds).

        Returns
        -------
        COCOGenerator
            An instance of COCOGenerator initialized with data extracted from the GeoDataFrame.
        """

        if gdf.crs is not None:
            gdf = convert_polygons_to_pixel_coordinates(gdf, tiles_paths_column)

        assert gdf.crs is None, "The GeoDataFrame must not have a CRS. Pixel coordinates for the polygons are expected."

        # Create a dictionary mapping tile paths (as Path objects) to their corresponding groups.
        groups = {}
        for key, group in gdf.groupby(tiles_paths_column):
            tile_path = key if isinstance(key, Path) else Path(key)
            groups[tile_path] = group

        # Determine the order of groups.
        if tiles_paths_order is not None:
            # Use the provided order: only include those tiles that exist in the grouped results.
            ordered_groups = [(tile, groups[tile]) for tile in tiles_paths_order if tile in groups]
        else:
            # Otherwise, use the natural order from the grouping.
            ordered_groups = list(groups.items())

        # Extract data from each group in the desired order.
        tiles_paths = []
        polygons_list = []
        scores_list = [] if scores_column is not None else None
        categories_list = [] if categories_column is not None else None
        other_attributes_list = [] if other_attributes_columns is not None else None

        for tile_path, group in ordered_groups:
            tiles_paths.append(tile_path)
            polygons_list.append(group[polygons_column].tolist())

            if scores_column is not None:
                scores_list.append(group[scores_column].tolist())

            if categories_column is not None:
                categories_list.append(group[categories_column].tolist())

            if other_attributes_columns is not None:
                attrs = group[other_attributes_columns].to_dict(orient='records')
                other_attributes_list.append(attrs)

        return cls(
            description=description,
            tiles_paths=tiles_paths,
            polygons=polygons_list,
            scores=scores_list,
            categories=categories_list,
            other_attributes=other_attributes_list,
            output_path=output_path,
            use_rle_for_labels=use_rle_for_labels,
            n_workers=n_workers,
            coco_categories_list=coco_categories_list
        )

    def generate_coco(self):
        """
        Generate the COCO dataset from the provided tiles, polygons, scores and other metadata.
        """
        categories_coco, category_to_id_map = self._generate_coco_categories()

        polygons_ids = self.get_polygon_ids(self.polygons)

        with ThreadPoolExecutor(self.n_workers) as pool:
            results = list(pool.map(self._generate_tile_coco, enumerate(
                zip(self.tiles_paths,
                    self.polygons,
                    polygons_ids,
                    self.scores if self.scores else [None, ] * len(self.tiles_paths),
                    [[category_to_id_map[c] if c in category_to_id_map else -1 for c in cs] for cs in self.categories],
                    self.other_attributes if self.other_attributes else [None, ] * len(self.tiles_paths),
                    [self.use_rle_for_labels, ] * len(self.tiles_paths)
                    )
            )))

        images_cocos = []
        annotations_cocos = []
        for result in results:
            images_cocos.append(result["image_coco"])
            annotations_cocos.extend(result["annotations_coco"])

        # Save the COCO dataset to a JSON file
        with self.output_path.open('w') as f:
            json.dump({
                "info": {
                    "description": self.description,
                    "version": "1.0",
                    "year": str(date.today().year),
                    "date_created": str(date.today())
                },
                "licenses": [
                    # Placeholder for licenses
                ],
                "images": images_cocos,
                "annotations": annotations_cocos,
                "categories": categories_coco
            }, f, ensure_ascii=False, indent=2)

        print(f'Verifying COCO file integrity...')
        # Verify COCO file integrity
        with open(os.devnull, 'w') as devnull, redirect_stdout(devnull):
            _ = COCO(self.output_path)
        print(f'Saved COCO dataset to {self.output_path}.')

    @staticmethod
    def get_polygon_ids(polygons: List[List[Polygon]]):
        start_id = 1
        polygon_ids = []
        for tile_polygons in polygons:
            tile_polygons_ids = list(range(start_id, start_id + len(tile_polygons)))
            polygon_ids.append(tile_polygons_ids)
            if tile_polygons_ids:
                start_id = tile_polygons_ids[-1] + 1

        return polygon_ids

    def _generate_tile_coco(self, tile_data):
        (tile_id,
         (tile_path, tile_polygons, tile_polygons_ids, tile_polygons_scores,
          tiles_polygons_category_ids, tiles_polygons_other_attributes, use_rle_for_labels)) = tile_data

        local_annotations_coco = []

        assert Path(tile_path).exists(), "Please make sure to save the tiles/images before creating the COCO dataset."

        with rasterio.open(tile_path) as tile:
            tile_width, tile_height = tile.width, tile.height

        for i in range(len(tile_polygons)):
            detection = self._generate_label_coco(
                polygon=tile_polygons[i],
                polygon_id=tile_polygons_ids[i],
                score=tile_polygons_scores[i] if tile_polygons_scores else None,
                tile_height=tile_height,
                tile_width=tile_width,
                tile_id=tile_id,
                use_rle_for_labels=use_rle_for_labels,
                category_id=tiles_polygons_category_ids[i] if tiles_polygons_category_ids else None,
                other_attributes_dict=tiles_polygons_other_attributes[i] if tiles_polygons_other_attributes else None
            )
            local_annotations_coco.append(detection)
        return {
            "image_coco": {
                "id": tile_id,
                "width": tile_width,
                "height": tile_height,
                "file_name": str(tile_path.name),
            },
            "annotations_coco": local_annotations_coco
        }

    @staticmethod
    def _generate_label_coco(polygon: shapely.Polygon,
                             polygon_id: int,
                             score: float,
                             tile_height: int,
                             tile_width: int,
                             tile_id: int,
                             use_rle_for_labels: bool,
                             category_id: int or None,
                             other_attributes_dict: dict or None) -> dict:
        if use_rle_for_labels:
            # Convert the polygon to a COCO RLE mask
            segmentation = polygon_to_coco_rle_segmentation(polygon=polygon,
                                                            tile_height=tile_height,
                                                            tile_width=tile_width)
        else:
            # Convert the polygon's exterior coordinates to the format expected by COCO
            segmentation = polygon_to_coco_coordinates_segmentation(polygon=polygon)

        # Calculate the area of the polygon
        area = polygon.area

        # Get the bounding box in COCO format: [x, y, width, height]
        bbox = list(polygon.bounds)
        bbox_coco_format = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

        # Generate COCO annotation data from each associated label
        coco_annotation = {
            "id": polygon_id,
            "segmentation": segmentation,
            "is_rle_format": use_rle_for_labels,
            "area": area,
            "iscrowd": 0,  # Assuming this polygon represents a single object (not a crowd)
            "image_id": tile_id,
            "bbox": bbox_coco_format,
            "category_id": category_id,
        }

        if score is not None:
            coco_annotation["score"] = score    # Add the score to the annotation if it's not None (only predictions have scores, not labels)
        if other_attributes_dict is None:
            coco_annotation["other_attributes"] = {}
        else:
            coco_annotation["other_attributes"] = other_attributes_dict

        return coco_annotation

    def _generate_coco_categories(self):
        """
        Generate COCO categories from the unique label categories in the dataset.
        """

        categories_set = set([category for categories in self.categories for category in categories]
                             if self.categories else {})

        if self.coco_categories_list:
            category_name_to_id_map = {}
            id_to_category_dict_map = {}
            for category_dict in self.coco_categories_list:
                assert 'name' in category_dict, "The id_to_category_map dictionary must contain a 'name' key in each id dict."
                assert 'id' in category_dict, "The id_to_category_map dictionary must contain an 'id' key in each id dict."
                assert 'supercategory' in category_dict, "The id_to_category_map dictionary must contain a 'supercategory' key in each id dict."
                assert category_dict['name'] not in category_name_to_id_map, \
                    f"The category names ('name' and 'other_names') must be unique. Found {category_dict['name']} twice."
                assert category_dict['id'] not in id_to_category_dict_map, \
                    f"The category ids must be unique. Found {category_dict['id']} twice."
                category_name_to_id_map[category_dict['name']] = category_dict['id']

                if "other_names" in category_dict and category_dict["other_names"]:
                    for other_name in category_dict["other_names"]:
                        assert other_name not in category_name_to_id_map, \
                            f"The category names ('name' and 'other_names') must be unique. Found {other_name} twice."
                        category_name_to_id_map[other_name] = category_dict['id']

            categories_coco = self.coco_categories_list

            if self.categories:
                for category in categories_set:
                    if category not in category_name_to_id_map:
                        warnings.warn(f"The category '{category}' is not in the provided COCO categories list.")
                        # raise Exception(f"The category '{category}' is not in the provided COCO categories list.")
            else:
                raise Exception("A 'coco_categories_list' was provided,"
                                " but categories haven't been provided for the polygons."
                                " Please set 'coco_categories_list' to None.")
        else:
            if self.categories:
                categories_coco = [{'id': i + 1, 'name': category, 'supercategory': ''} for i, category in
                                   enumerate(categories_set)]
                category_name_to_id_map = {category: i + 1 for i, category in enumerate(categories_set)}
            else:
                categories_coco = []
                category_name_to_id_map = {}
                warnings.warn("The GeoDataFrame containing the labels doesn't contain a category column,"
                              " so labels won't have categories.")

        return categories_coco, category_name_to_id_map


class PointCloudCOCOGenerator:
    """
    A class to generate a COCO dataset from a list of tiles and their associated polygons.
    After instantiating the class, the :meth:`generate_coco` method should be used to generate and save the COCO
    dataset.

    Parameters
    ----------
    description: str
        A description of the COCO dataset.
    tiles_metadata: List[Path]
        A list of paths to the tiles/images.
    polygons: List[List[Polygon]]
        A list of lists of polygons associated with each tile.
    scores: List[List[float]] or None
        A list of lists of scores associated with each polygon.
    categories: List[List[Union[str, int]]] or None
        A list of lists of categories (str or int) associated with each polygon.
    other_attributes: List[List[Dict]] or None
        A list of lists of dictionaries of other attributes associated with each polygon.
        Such a dict could be::

            {
                'attribute1': value1,
                'attribute2': value2
            }

        .. raw:: html

            <br>

        **IMPORTANT**: the 'score' attribute is reserved for the score associated with the polygon.
    output_path: Path
        The path to save the COCO dataset JSON file (should have .json extension).
    use_rle_for_labels: bool
        Whether to use RLE encoding for the labels or not. If False, the polygon's exterior coordinates will be used.
        RLE Encoding takes less space on disk but takes more time to encode.
    n_workers: int
        The number of workers to use for parallel processing.
    coco_categories_list: List[dict] or None
        A list of category dictionaries in COCO format. If a polygon has a category that is not in this list, its
        category_id will be set to None in its COCO annotation. If 'coco_categories_list' is None, the categories ids
        will be automatically generated from the unique categories found in the 'categories' parameter.

        .. raw:: html

            <br>

        To assign a category_id to a polygon, the code will check the 'name' and 'other_names' fields of the categories.

        .. raw:: html

            <br>

        **IMPORTANT**: It is strongly advised to provide this list if you want to have consistent category ids across
        multiple COCO datasets.

        .. raw:: html

            <br>

        Exemple of 2 categories, one being the parent of the other::

            [{
                "id": 1,
                "name": "Pinaceae",
                "other_names": [],
                "supercategory": null
            },
            {
                "id": 2,
                "name": "Picea",
                "other_names": ["PIGL", "PIMA", "PIRU"],
                "supercategory": 1
            }]
    """

    def __init__(self,
                 description: str,
                 tiles_metadata: List[Path],
                 polygons: List[List[Polygon]],
                 output_path: Path,
                 scores: List[List[float]] or None = None,
                 categories: List[List[Union[str, int]]] or None = None,
                 other_attributes: List[List[Dict]] or None = None,
                 use_rle_for_labels: bool = True,
                 n_workers: int = 1,
                 coco_categories_list: List[dict] or None = None):

        self.description = description
        self.tiles_metadata = tiles_metadata
        self.polygons = polygons
        self.scores = scores
        self.categories = categories
        self.other_attributes = other_attributes
        self.output_path = output_path
        self.use_rle_for_labels = use_rle_for_labels
        self.n_workers = n_workers
        self.coco_categories_list = coco_categories_list

        assert len(self.tiles_metadata) == len(self.polygons), "The number of tiles and polygons must be the same."
        if self.scores:
            assert len(self.tiles_metadata) == len(self.scores), "The number of tiles and scores must be the same."
        if self.categories:
            assert len(self.tiles_metadata) == len(
                self.categories), "The number of tiles and categories must be the same."
        if self.other_attributes:
            assert len(self.tiles_metadata) == len(
                self.other_attributes), "The number of tiles and other_attributes must be the same."

        if self.scores and self.other_attributes:
            self.other_attributes = [[dict(self.other_attributes[t][i], score=self.scores[t][i]) for i in
                                      range(len(self.other_attributes[t]))] for t in range(len(self.tiles_metadata))]

        elif self.scores:
            self.other_attributes = [[{'score': s} for s in tile_scores] for tile_scores in self.scores]

    def generate_coco(self):
        """
        Generate the COCO dataset from the provided tiles, polygons, scores and other metadata.
        """
        categories_coco, category_to_id_map = self._generate_coco_categories()

        polygons_ids = self.get_polygon_ids(self.polygons)

        with ThreadPoolExecutor(self.n_workers) as pool:
            results = list(pool.map(self._generate_tile_coco, enumerate(
                zip(self.tiles_metadata,
                    self.polygons,
                    polygons_ids,
                    [[category_to_id_map[c] if c in category_to_id_map else None for c in cs] for cs in self.categories]
                    if self.categories else [None, ] * len(self.tiles_metadata),
                    self.other_attributes if self.other_attributes else [None, ] * len(self.tiles_metadata),
                    [self.use_rle_for_labels, ] * len(self.tiles_metadata)
                    )
            )))

        point_cloud_cocos = []
        detections_cocos = []
        for result in results:
            point_cloud_cocos.append(result["point_cloud_coco"])
            detections_cocos.extend(result["detections_coco"])

        # Save the COCO dataset to a JSON file
        with self.output_path.open('w') as f:
            json.dump({
                "info": {
                    "description": self.description,
                    "version": "1.0",
                    "year": str(date.today().year),
                    "date_created": str(date.today())
                },
                "licenses": [
                    # Placeholder for licenses
                ],
                "point_cloud": point_cloud_cocos,
                "annotations": detections_cocos,
                "categories": categories_coco
            }, f, ensure_ascii=False, indent=2)

        print(f'Verifying COCO file integrity...')
        # Verify COCO file integrity
        _ = COCO(self.output_path)
        print(f'Saved COCO dataset to {self.output_path}.')

        return categories_coco, category_to_id_map

    @staticmethod
    def get_polygon_ids(polygons: List[List[Polygon]]):
        start_id = 1
        polygon_ids = []
        for tile_polygons in polygons:
            tile_polygons_ids = list(range(start_id, start_id + len(tile_polygons)))
            polygon_ids.append(tile_polygons_ids)
            start_id = tile_polygons_ids[-1] + 1

        return polygon_ids

    def _generate_tile_coco(self, tile_data):
        (tile_id,
         (tile_metadata, tile_polygons, tile_polygons_ids,
          tiles_polygons_category_ids, tiles_polygons_other_attributes, use_rle_for_labels)) = tile_data

        local_detections_coco = []

        tile_width, tile_height = tile_metadata.height, tile_metadata.width

        for i in range(len(tile_polygons)):
            detection = self._generate_label_coco(
                polygon=tile_polygons[i],
                polygon_id=tile_polygons_ids[i],
                tile_height=tile_height,
                tile_width=tile_width,
                tile_id=tile_id,
                use_rle_for_labels=use_rle_for_labels,
                category_id=tiles_polygons_category_ids[i] if tiles_polygons_category_ids else None,
                other_attributes_dict=tiles_polygons_other_attributes[i] if tiles_polygons_other_attributes else None
            )
            local_detections_coco.append(detection)
        return {
            "point_cloud_coco": {
                "id": tile_id,
                "width": tile_width,
                "height": tile_height,
                "file_name": tile_metadata.tile_name,
                "min_x": tile_metadata.min_x,
                "max_x": tile_metadata.max_x,
                "min_y": tile_metadata.min_y,
                "max_y": tile_metadata.max_y,
            },
            "detections_coco": local_detections_coco
        }

    @staticmethod
    def _generate_label_coco(polygon: shapely.Polygon,
                             polygon_id: int,
                             tile_height: int,
                             tile_width: int,
                             tile_id: int,
                             use_rle_for_labels: bool,
                             category_id: int or None,
                             other_attributes_dict: dict or None) -> dict:

        if use_rle_for_labels:
            # Convert the polygon to a COCO RLE mask
            segmentation = polygon_to_coco_rle_segmentation(polygon=polygon,
                                                            tile_height=tile_height,
                                                            tile_width=tile_width)
        else:
            # Convert the polygon's exterior coordinates to the format expected by COCO
            segmentation = polygon_to_coco_coordinates_segmentation(polygon=polygon)

        # Calculate the area of the polygon
        area = polygon.area

        # Get the bounding box in COCO format: [x, y, width, height]
        bbox = list(polygon.bounds)
        bbox_coco_format = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

        # Generate COCO annotation data from each associated label
        coco_annotation = {
            "id": polygon_id,
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

    def _generate_coco_categories(self):
        """
        Generate COCO categories from the unique label categories in the dataset.
        """

        categories_set = set([category for categories in self.categories for category in categories]
                             if self.categories else {})
        if self.coco_categories_list:
            category_name_to_id_map = {}
            id_to_category_dict_map = {}
            for category_dict in self.coco_categories_list:
                assert 'name' in category_dict, "The id_to_category_map dictionary must contain a 'name' key in each id dict."
                assert 'id' in category_dict, "The id_to_category_map dictionary must contain an 'id' key in each id dict."
                assert 'supercategory' in category_dict, "The id_to_category_map dictionary must contain a 'supercategory' key in each id dict."
                assert category_dict['name'] not in category_name_to_id_map, \
                    f"The category names ('name' and 'other_names') must be unique. Found {category_dict['name']} twice."
                assert category_dict['id'] not in id_to_category_dict_map, \
                    f"The category ids must be unique. Found {category_dict['id']} twice."
                category_name_to_id_map[category_dict['name']] = category_dict['id']

                if "other_names" in category_dict and category_dict["other_names"]:
                    for other_name in category_dict["other_names"]:
                        assert other_name not in category_name_to_id_map, \
                            f"The category names ('name' and 'other_names') must be unique. Found {other_name} twice."
                        category_name_to_id_map[other_name] = category_dict['id']

            categories_coco = self.coco_categories_list
            if self.categories:
                for category in categories_set:
                    if category not in category_name_to_id_map:
                        warnings.warn(f"The category '{category}' is not in the provided COCO categories list.")
                        # raise Exception(f"The category '{category}' is not in the provided COCO categories list.")
            else:
                raise Exception("A 'coco_categories_list' was provided,"
                                " but categories haven't been provided for the polygons."
                                " Please set 'coco_categories_list' to None.")
        else:
            if self.categories:
                categories_coco = [{'id': i + 1, 'name': category, 'supercategory': ''} for i, category in
                                   enumerate(categories_set)]
                category_name_to_id_map = {category: i + 1 for i, category in enumerate(categories_set)}
            else:
                categories_coco = []
                category_name_to_id_map = {}
                warnings.warn("The GeoDataFrame containing the labels doesn't contain a category column,"
                              " so labels won't have categories.")

        return categories_coco, category_name_to_id_map


def create_coco_folds(train_coco_path: str or Path, output_dir: str or Path, num_folds=5, seed=0, predefined_image_folds=None):
    """
    Create folds for a COCO dataset by splitting the images randomly or using predefined folds.

    Parameters
    ----------
    train_coco_path: str or Path
        The path to the train COCO JSON file.
    output_dir: str or Path
        The directory where the folds will be saved.
    num_folds: int
        The number of folds to create.
    seed: int or None
        The random seed for shuffling image IDs if predefined_image_folds is None.
    predefined_image_folds: dict or None
        A dictionary mapping image ids to fold IDs. If provided, this overrides random splitting.
    """

    # Set the random seed
    if seed is not None:
        random.seed(seed)

    train_coco_path = Path(train_coco_path)

    # Load the train COCO JSON file
    with open(train_coco_path, 'r') as f:
        train_coco = json.load(f)

    product_name, scale_factor, ground_resolution, _ = CocoNameConvention.parse_name(train_coco_path.name)

    # Get the list of images
    images = train_coco['images']

    # Assign images to folds
    if predefined_image_folds:
        folds = [[] for _ in range(num_folds)]
        for img in images:
            fold_id = predefined_image_folds.get(img['id'])
            if fold_id is not None and 0 <= fold_id < num_folds:
                folds[fold_id].append(img['id'])
            else:
                raise ValueError(f"Invalid fold ID for image {img['id']}: {fold_id}")
    else:
        # Shuffle the image IDs randomly
        image_ids = [img['id'] for img in images]
        random.shuffle(image_ids)

        # Split the images into folds
        folds = [[] for _ in range(num_folds)]
        for idx, img_id in enumerate(image_ids):
            folds[idx % num_folds].append(img_id)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_dir = Path(output_dir)

    for fold in range(num_folds):
        # Get the image IDs for the current fold
        valid_image_ids = folds[fold]
        train_image_ids = [img_id for f in folds if f != valid_image_ids for img_id in f]

        # Create the train and valid COCO datasets for the current fold
        train_coco_fold = {
            'categories': train_coco['categories'],
            'images': [img for img in train_coco['images'] if img['id'] in train_image_ids],
            'annotations': [ann for ann in train_coco['annotations'] if ann['image_id'] in train_image_ids]
        }
        valid_coco_fold = {
            'categories': train_coco['categories'],
            'images': [img for img in train_coco['images'] if img['id'] in valid_image_ids],
            'annotations': [ann for ann in train_coco['annotations'] if ann['image_id'] in valid_image_ids]
        }

        if 'info' in train_coco:
            train_coco_fold['info'] = train_coco['info']
            valid_coco_fold['info'] = train_coco['info']
        if 'licenses' in train_coco:
            train_coco_fold['licenses'] = train_coco['licenses']
            valid_coco_fold['licenses'] = train_coco['licenses']

        train_fold_coco_name = CocoNameConvention.create_name(
            product_name=product_name,
            fold=f'train{fold}',
            scale_factor=scale_factor,
            ground_resolution=ground_resolution
        )

        valid_fold_coco_name = CocoNameConvention.create_name(
            product_name=product_name,
            fold=f'valid{fold}',
            scale_factor=scale_factor,
            ground_resolution=ground_resolution
        )

        # Save the train and valid COCO JSON files for the current fold
        with open(output_dir / train_fold_coco_name, 'w') as f:
            json.dump(train_coco_fold, f, ensure_ascii=False, indent=2)
        with open(output_dir / valid_fold_coco_name, 'w') as f:
            json.dump(valid_coco_fold, f, ensure_ascii=False, indent=2)

    print(f"Created {num_folds} folds in {output_dir}.")
    return output_dir


def apply_affine_transform(geom: shapely.geometry, affine: rasterio.Affine):
    """
    Apply an affine transformation to a geometry.

    Parameters
    ----------
    geom: shapely.geometry
        The geometry to transform.
    affine: rasterio.Affine
        The affine transformation to apply.

    Returns
    -------
    shapely.geometry
        The transformed geometry.
    """
    return transform(lambda x, y: affine * (x, y), geom)


def apply_inverse_transform(polygons: List[Polygon], raster_path: str or Path):
    """
    Apply the inverse transform of a raster to a list of polygons.

    Parameters
    ----------
    polygons: List[Polygon]
        The list of polygons to transform.
    raster_path: str or Path
        The path to the raster file.

    Returns
    -------
    List[Polygon]
    """
    src = rasterio.open(raster_path)
    inverse_transform = ~src.transform
    polygons = [apply_affine_transform(polygon, inverse_transform) for polygon in polygons]
    return polygons


def coco_to_geopackage(coco_json_path: str,
                       images_directory: str,
                       convert_to_crs_coordinates: bool,
                       geopackage_output_path: str or None):
    """
    Converts a COCO JSON dataset into a GeoDataFrame, then saved if needed as a GeoPackage file.

    The resulting GeoDataFrame (or GeoPackage if saved) will have the following columns:

    - geometry: The polygon geometry
    - tile_id: The ID of the tile the polygon belongs to
    - tile_path: The path to the tile image
    - category_id: The ID of the category of the polygon
    - category_name: The name of the category of the polygon
    - any other attributes found in the 'other_attributes' field of the COCO JSON annotations

    Parameters
    ----------
    coco_json_path: str
        The path to the COCO JSON dataset (.json).
    images_directory: str
        The directory containing the images associated with the COCO dataset.
    convert_to_crs_coordinates: bool
        Whether to convert the polygon pixel coordinates to a common CRS (uses the CRS of the first .tif tile).
    geopackage_output_path: str or None
        The path to save the GeoPackage file. If None, the GeoPackage file will not be saved to the disk.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the polygons from the COCO dataset
    """

    # Load COCO JSON
    with open(coco_json_path, 'r') as file:
        coco_data = json.load(file)

    tiles_data = coco_data['images']
    annotations_data = coco_data['annotations']
    categories_data = coco_data['categories']

    print("Found {} tiles and {} annotations.".format(len(tiles_data), len(annotations_data)))

    # Create a mapping of category IDs to category names
    categories_ids_to_names_map = {category['id']: category['name'] for category in categories_data}
    categories_ids_to_names_map[None] = None

    tiles_ids_to_tiles_map = {tile['id']: tile for tile in tiles_data}
    tiles_ids_to_annotations_map = {tile['id']: [] for tile in tiles_data}
    for annotation in annotations_data:
        tiles_ids_to_annotations_map[annotation['image_id']].append(annotation)

    gdfs = []
    for tile_id in tiles_ids_to_tiles_map:
        tile_data = tiles_ids_to_tiles_map[tile_id]
        tile_annotations = tiles_ids_to_annotations_map[tile_id]

        polygons = []
        for annotation in tile_annotations:
            polygon = decode_coco_segmentation(annotation, 'polygon')
            polygons.append(polygon)

        gdf = gpd.GeoDataFrame({
            'geometry': polygons,
            'tile_id': tile_id,
            'tile_path': f"{images_directory}/{tile_data['file_name']}",
            'category_id': [annotation.get('category_id') for annotation in tile_annotations],
            'category_name': [categories_ids_to_names_map[annotation.get('category_id')] for annotation in tile_annotations],
        })

        if tile_annotations:
            if 'other_attributes' in tile_annotations[0] and type(tile_annotations[0]['other_attributes']) is dict:
                other_attributes = set()
                for tile_annotation in tile_annotations:
                    other_attributes = other_attributes.union(set(tile_annotation['other_attributes'].keys()))

                for other_attribute in other_attributes:
                    gdf[other_attribute] = [annotation['other_attributes'][other_attribute]
                                            if (other_attribute in annotation['other_attributes']
                                                and type(annotation['other_attributes'][other_attribute])
                                                in [str, int, float]) else None for annotation in tile_annotations]

            if 'score' in tile_annotations[0]:
                gdf['score'] = [annotation['score'] for annotation in tile_annotations]

        gdfs.append(gdf)

    all_polygons_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    all_polygons_gdf.set_geometry('geometry')

    if convert_to_crs_coordinates:
        all_polygons_gdf = tiles_polygons_gdf_to_crs_gdf(all_polygons_gdf)

    if geopackage_output_path:
        ext = Path(geopackage_output_path).suffix
        if ext == '.gpkg':
            all_polygons_gdf.to_file(geopackage_output_path, driver='GPKG')
            print(f"Successfully converted the COCO json into a GeoPackage file (.gpkg) saved at {geopackage_output_path}.")
        elif ext == '.geojson':
            all_polygons_gdf.to_file(geopackage_output_path, driver='GeoJSON')
            print(f"Successfully converted the COCO json into a GeoJSON file saved at {geopackage_output_path}.")
        elif ext == '.shp':
            all_polygons_gdf.to_file(geopackage_output_path, driver='ESRI Shapefile')
            print(f"Successfully converted the COCO json into a Shapefile saved at {geopackage_output_path}.")
        else:
            raise Exception(f"Output file format {ext} not supported. Please use .gpkg, .geojson or .shp.")

    return all_polygons_gdf


def tiles_polygons_gdf_to_crs_gdf(dataframe: gpd.GeoDataFrame):
    """
    Converts a GeoDataFrame of polygons from multiple tiles to a common CRS.
    The dataframe passed must have a 'tile_path' column containing the path to the tile image, as the function
    needs to read each tile metadata to get their respective CRS.

    Parameters
    ----------
    dataframe: GeoDataFrame
        The GeoDataFrame containing the polygons from multiple tiles.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the polygons in a common CRS.
    """

    assert 'tile_path' in dataframe.columns, "The GeoDataFrame must contain a 'tile_path' column."

    # get the first tile_path
    first_tile_path = dataframe['tile_path'].iloc[0]
    common_crs = rasterio.open(first_tile_path).crs

    gdfs = []
    for tile_path in dataframe['tile_path'].unique():
        tile_gdf = dataframe[dataframe['tile_path'] == tile_path].copy(deep=True)

        tile_src = rasterio.open(tile_path)
        tile_gdf['geometry'] = tile_gdf.apply(lambda row: apply_affine_transform(row['geometry'], tile_src.transform), axis=1)
        tile_gdf.crs = tile_src.crs
        tile_gdf = tile_gdf.to_crs(common_crs)
        gdfs.append(tile_gdf)

    all_tiles_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    all_tiles_gdf.set_geometry('geometry')
    all_tiles_gdf.crs = common_crs

    return all_tiles_gdf


def find_tiles_paths(directories: List[Path], extensions: List[str]) -> dict[str, Path]:
    """
    Find all files with given extensions in a list of directories, recursively.
    """
    tiles_names_to_paths = {}

    for directory in directories:
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"Directory {directory} not found or is not a directory.")

        files = []
        # Loop over each extension and find matching files
        for ext in extensions:
            files.extend(glob.glob(os.path.join(directory, '**', f'*.{ext}'), recursive=True))

        files = [Path(file) for file in files]

        for file in files:
            if file.name in tiles_names_to_paths:
                raise Exception(f"Multiple files with the same name found: {file.name}. Files must have unique names.")
            else:
                tiles_names_to_paths[file.name] = file

    return tiles_names_to_paths

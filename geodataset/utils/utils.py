import base64
import json

import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
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


def rle_segmentation_to_mask(segmentation: dict) -> np.ndarray:
    """
    Decodes an RLE segmentation into a binary mask.
    """
    # Decode the counts from base64
    if 'counts' in segmentation and isinstance(segmentation['counts'], str):
        counts = base64.b64decode(segmentation['counts'])
        segmentation['counts'] = counts

    mask = mask_utils.decode(segmentation)
    return mask


def rle_segmentation_to_bbox(segmentation: dict) -> box:
    """
    Calculates the bounding box from a binary mask.
    """

    mask = rle_segmentation_to_mask(segmentation)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    return box(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)


def polygon_segmentation_to_polygon(segmentation: list) -> Polygon:
    """
    Calculates the bounding box from a polygon.
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


def polygon_segmentation_to_bbox(segmentation: list) -> box:
    """
    Calculates the bounding box from a polygon.
    """
    polygon = polygon_segmentation_to_polygon(segmentation)
    return box(*polygon.bounds)


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

        base_cmap.append(colors[idx - 1])
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


class COCOGenerator:
    def __init__(self,
                 description: str,
                 tiles_paths: List[Path],
                 polygons: List[List[Polygon]],
                 scores: List[List[float]] or None,
                 categories: List[List[Union[str, int]]] or None,
                 other_attributes: List[Dict] or None,
                 output_path: Path,
                 use_rle_for_labels: bool,
                 n_workers: int,
                 main_label_category_to_id_map: dict or None):
        self.description = description
        self.tiles_paths = tiles_paths
        self.polygons = polygons
        self.scores = scores
        self.categories = categories
        self.other_attributes = other_attributes
        self.output_path = output_path
        self.use_rle_for_labels = use_rle_for_labels
        self.n_workers = n_workers
        self.main_label_category_to_id_map = main_label_category_to_id_map

        assert len(self.tiles_paths) == len(self.polygons), "The number of tiles and polygons must be the same."
        if self.scores:
            assert len(self.tiles_paths) == len(self.scores), "The number of tiles and scores must be the same."
        if self.categories:
            assert len(self.tiles_paths) == len(self.categories), "The number of tiles and categories must be the same."
        if self.other_attributes:
            assert len(self.tiles_paths) == len(
                self.other_attributes), "The number of tiles and other_attributes must be the same."

        if self.scores and self.other_attributes:
            self.other_attributes = [[dict(self.other_attributes[t][i], score=self.scores[t][i]) for i in
                                     range(len(self.other_attributes[t]))] for t in range(len(self.tiles_paths))]

        elif self.scores:
            self.other_attributes = [[{'score': s} for s in tile_scores] for tile_scores in self.scores]

    def generate_coco(self):
        id_to_category_map, category_to_id_map = self._generate_coco_categories()

        with ThreadPoolExecutor(self.n_workers) as pool:
            results = list(pool.map(self._generate_tile_coco, enumerate(
                zip(self.tiles_paths,
                    self.polygons,
                    [[category_to_id_map[c] for c in cs] for cs in self.categories] if self.categories else [None, ]*len(self.tiles_paths),
                    self.other_attributes if self.other_attributes else [None, ] * len(self.tiles_paths),
                    [self.use_rle_for_labels, ] * len(self.tiles_paths)
                    )
            )))

        images_cocos = []
        detections_cocos = []
        for result in results:
            images_cocos.append(result["image_coco"])
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
                "images": images_cocos,
                "annotations": detections_cocos,
                "categories": id_to_category_map
            }, f, ensure_ascii=False, indent=2)

        print(f'Saved COCO dataset to {self.output_path}.')

    def _generate_tile_coco(self, tile_data):
        (tile_id,
         (tile_path, tile_polygons, tiles_polygons_category_ids, tiles_polygons_other_attributes, use_rle_for_labels)) = tile_data

        local_detections_coco = []

        assert Path(tile_path).exists(), "Please make sure to save the tiles/images before creating the COCO dataset."

        with rasterio.open(tile_path) as tile:
            tile_width, tile_height = tile.width, tile.height

        for i in range(len(tile_polygons)):
            detection = self._generate_label_coco(
                polygon=tile_polygons[i],
                tile_height=tile_height,
                tile_width=tile_width,
                tile_id=tile_id,
                use_rle_for_labels=use_rle_for_labels,
                category_id=tiles_polygons_category_ids[i] if tiles_polygons_category_ids else None,
                other_attributes_dict=tiles_polygons_other_attributes[i] if tiles_polygons_other_attributes else None
            )
            local_detections_coco.append(detection)
        return {
            "image_coco": {
                "id": tile_id,
                "width": tile_width,
                "height": tile_height,
                "file_name": str(tile_path.name),
            },
            "detections_coco": local_detections_coco
        }

    @staticmethod
    def _generate_label_coco(polygon: shapely.Polygon,
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

    def _generate_coco_categories(self):
        """
        Generate COCO categories from the unique label categories in the dataset.
        """
        categories_set = set([category for categories in self.categories for category in categories] if self.categories else [])
        if self.main_label_category_to_id_map:
            for c in categories_set:
                if c not in self.main_label_category_to_id_map:
                    raise Exception(f"Category '{c}' is not in the provided main_label_category_to_id_map.")
            category_to_id_map = self.main_label_category_to_id_map
            id_to_category_map = [{'id': category_to_id_map[category], 'name': category, 'supercategory': ''} for category in self.main_label_category_to_id_map.keys()]
        else:
            if self.categories:
                id_to_category_map = [{'id': i + 1, 'name': category, 'supercategory': ''} for i, category in
                                      enumerate(categories_set)]
                category_to_id_map = {category: i + 1 for i, category in enumerate(categories_set)}
            else:
                id_to_category_map = {}
                category_to_id_map = {}
                warnings.warn("The GeoDataFrame containing the labels doesn't contain a category column,"
                              " so labels won't have categories.")

        return id_to_category_map, category_to_id_map


def apply_affine_transform(geom: shapely.geometry, affine: rasterio.Affine):
    return transform(lambda x, y: affine * (x, y), geom)


def decode_rle_to_polygon(segmentation):
    # Decode the RLE
    mask = rle_segmentation_to_mask(segmentation)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize an empty list to hold the exterior coordinates of the polygons
    polygons = []

    for contour in contours:
        # Ensure the contour is of a significant size
        if len(contour) > 2:
            # Flatten the array and convert to a list of tuples
            contour = contour.flatten().tolist()
            # Reshape the contour to obtain a sequence of coordinates
            points = list(zip(contour[::2], contour[1::2]))
            polygons.append(Polygon(points))

    # Return the polygons (Note: This might be a list of polygons if there are multiple disconnected regions)
    if len(polygons) == 1:
        return polygons[0]  # Return the single polygon directly if there's only one
    else:
        return gpd.GeoSeries(polygons)  # Return a GeoSeries of Polygons if there are multiple


def coco_to_geojson(coco_json_path: str,
                    images_directory: str,
                    convert_to_crs_coordinates: bool,
                    geojson_output_path: str or None):
    # Load COCO JSON
    with open(coco_json_path, 'r') as file:
        coco_data = json.load(file)

    tiles_data = coco_data['images']
    annotations_data = coco_data['annotations']

    tiles_ids_to_tiles_map = {tile['id']: tile for tile in tiles_data}
    tiles_ids_to_annotations_map = {tile['id']: [] for tile in tiles_data}
    for annotation in annotations_data:
        tiles_ids_to_annotations_map[annotation['image_id']].append(annotation)

    # Getting the first tile CRS as the CRS that will be used for all tiles and their annotations
    if convert_to_crs_coordinates:
        common_crs = rasterio.open(f"{images_directory}/{coco_data['images'][0]['file_name']}").crs
    else:
        common_crs = None

    gdfs = []
    for tile_id in tiles_ids_to_tiles_map:
        tile_data = tiles_ids_to_tiles_map[tile_id]
        tile_annotations = tiles_ids_to_annotations_map[tile_id]

        polygons = []
        for annotation in tile_annotations:
            # Check if segmentation data is in RLE format; if so, decode it
            if annotation.get('is_rle_format', False) or isinstance(annotation['segmentation'], dict):
                polygon = decode_rle_to_polygon(segmentation=annotation['segmentation'])
            else:
                polygon = polygon_segmentation_to_polygon(segmentation=annotation['segmentation'])
            polygons.append(polygon)

        gdf = gpd.GeoDataFrame({
            'geometry': polygons,
            'tile_id': tile_id,
            'tile_path': tile_data['file_name'],
            'category_id': [annotation.get('category_id') for annotation in tile_annotations],
        })

        tile_src = rasterio.open(f"{images_directory}/{tile_data['file_name']}")

        gdf['geometry'] = gdf.apply(lambda row: apply_affine_transform(row['geometry'], tile_src.transform), axis=1)
        if convert_to_crs_coordinates:
            gdf.crs = tile_src.crs
            gdf = gdf.to_crs(common_crs)
        gdfs.append(gdf)

    all_polygons_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    all_polygons_gdf.set_geometry('geometry')
    all_polygons_gdf.crs = common_crs
    if geojson_output_path:
        all_polygons_gdf.to_file(geojson_output_path, driver='GeoJSON')
        print(f"Successfully converted the COCO json into a GeoJSON file saved at {geojson_output_path}.")

    return all_polygons_gdf

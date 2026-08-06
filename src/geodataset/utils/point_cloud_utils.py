import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import List, Union, Dict

import laspy
import shapely
from pycocotools.coco import COCO
from shapely import Polygon

from geodataset.utils import polygon_to_coco_rle_segmentation, polygon_to_coco_coordinates_segmentation


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
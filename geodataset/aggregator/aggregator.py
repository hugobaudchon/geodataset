import json
import warnings
from pathlib import Path
from typing import List, Dict
from warnings import warn

import pandas as pd
import rasterio
import shapely
from shapely import Polygon, MultiPolygon
import geopandas as gpd
from shapely.validation import make_valid
from tqdm import tqdm

from geodataset.utils import TileNameConvention, apply_affine_transform, COCOGenerator, apply_inverse_transform, \
    decode_coco_segmentation, fix_geometry_collection


class Aggregator:
    """
    Class to aggregate polygons from multiple tiles, with different pixel coordinate systems,
    into a single GeoDataFrame with a common CRS, applying the Non-Maximum Suppression (NMS) algorithm.

    This class should be instantiated using the 'from_coco' or 'from_polygons' class methods.
    """

    SUPPORTED_NMS_ALGORITHMS = ['iou', 'ioa-disambiguate']  # 'diou'
    SUPPORTED_SCORE_WEIGHTING_METHODS = ['weighted_arithmetic_mean', 'weighted_geometric_mean', 'weighted_harmonic_mean']

    def __init__(self,
                 output_path: str or Path,
                 polygons_gdf: gpd.GeoDataFrame,
                 scores_names: List[str],
                 other_attributes_names: List[str],
                 scores_weights: List[float],
                 tiles_extent_gdf: gpd.GeoDataFrame,
                 tile_ids_to_path: dict or None,
                 scores_weighting_method: str = 'weighted_geometric_mean',
                 min_centroid_distance_weight: float = None,
                 score_threshold: float = 0.1,
                 nms_threshold: float = 0.8,
                 nms_algorithm: str = 'iou',
                 best_geom_keep_area_ratio: float = 0.5,
                 pre_aggregated_output_path: str or Path = None):

        self.output_path = Path(output_path)
        self.polygons_gdf = polygons_gdf
        self.scores_names = scores_names
        self.other_attributes_names = other_attributes_names
        self.scores_weights = scores_weights
        self.tiles_extent_gdf = tiles_extent_gdf
        self.tile_ids_to_path = tile_ids_to_path
        self.scores_weighting_method = scores_weighting_method
        self.min_centroid_distance_weight = min_centroid_distance_weight
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.nms_algorithm = nms_algorithm
        self.best_geom_keep_area_ratio = best_geom_keep_area_ratio
        self.pre_aggregated_output_path = pre_aggregated_output_path

        if self.pre_aggregated_output_path:
            self.polygons_gdf.to_file(self.pre_aggregated_output_path, driver='GPKG')

        # If only 1 set of scores is provided, we set the score_weight to 1.0
        if len(self.scores_weights) == 1: self.scores_weights = [1.0]

        self._check_parameters()
        self._validate_polygons()
        self._prepare_scores()
        self._remove_low_score_polygons()
        self._apply_nms_algorithm()
        self._save_polygons()

    def _check_parameters(self):
        assert self.polygons_gdf.crs, "The provided polygons_gdf doesn't have a CRS."
        assert 'tile_id' in self.polygons_gdf, "The provided polygons_gdf doesn't have a 'tile_id' column."
        assert self.tiles_extent_gdf.crs, "The provided tiles_extent_gdf doesn't have a CRS."
        assert 'tile_id' in self.tiles_extent_gdf, "The provided tiles_extent_gdf doesn't have a 'tile_id' column."
        assert self.nms_algorithm in self.SUPPORTED_NMS_ALGORITHMS, \
            f"The nms_algorithm must be one of {self.SUPPORTED_NMS_ALGORITHMS}. Got {self.nms_algorithm}."

        assert self.output_path.suffix in ['.json', '.geojson', '.gpkg'], \
            "The output_path needs to end with either .json (coco output) or .geojson/.gpkg (geopackage output)."

        if self.output_path.suffix == '.json':
            assert self.tile_ids_to_path is not None, \
                "The tile_ids_to_path must be provided to save the polygons in COCO format."

        assert sum(self.scores_weights) == 1, "The sum of the scores_weights values must be equal to 1."
        assert self.scores_weighting_method in self.SUPPORTED_SCORE_WEIGHTING_METHODS, \
            f"The score_weighting_method must be one of {self.SUPPORTED_SCORE_WEIGHTING_METHODS}. " \

    @classmethod
    def from_coco(cls,
                  output_path: str or Path,
                  tiles_folder_path: str or Path,
                  coco_json_path: str or Path,
                  scores_names: List[str] = None,
                  other_attributes_names: List[str] = None,
                  scores_weights: List[float] = None,
                  scores_weighting_method: str = 'weighted_geometric_mean',
                  min_centroid_distance_weight: float = None,
                  score_threshold: float = 0.1,
                  nms_threshold: float = 0.8,
                  nms_algorithm: str = 'iou',
                  best_geom_keep_area_ratio: float=0.5,
                  pre_aggregated_output_path: str or Path = None):

        """
        Instanciate and run an Aggregator from a COCO JSON file.
        The polygons will be read from the coco_json_path, and the tiles will be read from the tiles_folder_path.

        Parameters
        ----------
        output_path: str or Path
            The filename where to save the aggregated polygons.
            '.gpkg', '.geojson' are supported, as well as '.json' for COCO format.
        tiles_folder_path: str or Path
            The folder where the tiles are stored.
        coco_json_path: str or Path
            The path to the COCO JSON file.
        scores_names: List[str]
            The names of the attributes in the COCO file annotations which should be used as scores.
            The code will look directly in the COCO annotations for these keys,
            and will also look in the 'other_attributes' key if the key is not found directly in the annotation.

            For example, if score_names = ['detection_score'], the 'detection_score' attribute can be in 2 different
            places in each COCO annotation::

                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "segmentation": {
                            "counts": "eNpjYBBgUD8rgjQmInZMmB0A",
                            "size": [224, 224]
                        },
                        "area": 10000.0,
                        "bbox": [100.0, 100.0, 100.0, 100.0],
                        "iscrowd": 0,
                        "detection_score": 0.8      # <- directly in the annotation
                        "other_attributes": {
                            "detection_score": 0.8  # <- in the 'other_attributes' key
                        }
                    },
                    ...
                ]
        other_attributes_names: List[str]
            The names of the attributes in the COCO file annotations which should be kept in the output result
            (but won't be used by the Aggregator itself). Same structure than "scores_names".
        scores_weights: List[float]
            The weights to apply to each score set. The scores_weights should sum to 1 and there should be as many
             weights as there are scores_names. See the 'scores_weighting_method' parameter for more details.
        scores_weighting_method: str
            The method to use to weight the different sets of scores. Supported values are ['weighted_arithmetic_mean',
            'weighted_geometric_mean', 'weighted_harmonic_mean'].

                - 'weighted_arithmetic_mean': The scores are simply averaged, but each score is weighted by its weight.
                - 'weighted_geometric_mean': The scores are multiplied, but each score is weighted by its weight.
                - 'weighted_harmonic_mean': The scores are averaged, but each score is weighted by its reciprocal.
        min_centroid_distance_weight: float or None
            The weight to apply to the polygon scores (after applying the scores_weights) based on the distance
             between the polygon centroid and its tile centroid. The smaller the value of min_centroid_distance_weight,
             the smaller the score will be for polygons at the edge of the tile.
        score_threshold: float
            The score threshold under which polygons will be removed, before applying the NMS algorithm.
        nms_threshold: float
            The threshold for the Non-Maximum Suppression algorithm. It serves different purposes depending on the
            'nms_algorithm' chosen:

                - 'iou': The threshold is used as the regular Intersection Over Union (IoU) threshold.
                  If 2 polygons have an IoU above this threshold, the one with the lowest score will be removed.
                - 'ioa-disambiguate': This is the threshold used to determine if a lesser-scored polygon should be
                discarded (entirely removed), or cut to remove its portion that overlaps with another higher-scored
                polygon. The equation is a bit different than IoU: it's the area of the intersection divided by
                the area of the lesser-scored polygon (Intersection / Area_lower_scored_polygon).

        nms_algorithm: str
            The Non-Maximum Suppression algorithm to use. Supported values are ['iou' and 'ioa-disambiguate'].

                - 'iou': This algorithm uses the Intersection Over Union (IoU) to suppress overlapping polygons.
                - 'ioa-disambiguate': Shouldn't be used for detection bounding boxes,
                  it was designed for instance segmentations.
                  In addition to using IoU, this algorithm attempts to disambiguate overlapping polygons by
                  cutting the intersecting parts and keeping only the largest resulting geometry, as long as it meets the
                  `best_geom_keep_area_ratio` threshold.
        best_geom_keep_area_ratio: float
            Only used when nms_algorithm is set to 'ioa-disambiguate'. When removing part of a polygon due to
            the intersection with another one, that polygon may end-up being cut into multiple smaller and disconnected
            geometries. The algorithm will only consider keeping the largest of those geometries, and only keep it if
            it makes up at least 'best_geom_keep_area_ratio' percent of the original polygon (in terms of area).
        pre_aggregated_output_path: str or Path
            If provided, the polygons will be saved in this file before applying the NMS algorithm.
            This is useful to debug the polygons before the NMS algorithm is applied.

        Returns
        -------
        Aggregator
        """

        if scores_names is None:
            scores_names = ['score']  # will try to find a 'score' attribute as it's required for NMS algorithm.

        if other_attributes_names is None:
            other_attributes_names = []  # by default other attributes are not mandatory

        if scores_weights:
            assert len(scores_names) == len(scores_weights), ("The scores_weights must have "
                                                              "the same length as the scores_names.")
        else:
            scores_weights = [1 / len(scores_names), ] * len(scores_names)

        all_polygons_gdf, all_tiles_extents_gdf, tile_ids_to_path = cls._from_coco(
            tiles_folder_path=tiles_folder_path,
            coco_json_path=coco_json_path,
            attributes_names=scores_names + other_attributes_names
        )

        return cls(output_path=output_path,
                   polygons_gdf=all_polygons_gdf,
                   scores_names=scores_names,
                   other_attributes_names=other_attributes_names,
                   scores_weights=scores_weights,
                   tiles_extent_gdf=all_tiles_extents_gdf,
                   scores_weighting_method=scores_weighting_method,
                   min_centroid_distance_weight=min_centroid_distance_weight,
                   score_threshold=score_threshold,
                   nms_threshold=nms_threshold,
                   nms_algorithm=nms_algorithm,
                   tile_ids_to_path=tile_ids_to_path,
                   best_geom_keep_area_ratio=best_geom_keep_area_ratio,
                   pre_aggregated_output_path=pre_aggregated_output_path)

    @classmethod
    def from_polygons(cls,
                      output_path: str or Path,
                      tiles_paths: List[str or Path],
                      polygons: List[List[Polygon]],
                      scores: List[List[float]] or Dict[str, List[List[float]]],
                      other_attributes: Dict[str, List[List[float]]],
                      scores_weights: Dict[str, float] = None,
                      scores_weighting_method: str = 'weighted_geometric_mean',
                      min_centroid_distance_weight: float = None,
                      score_threshold: float = 0.1,
                      nms_threshold: float = 0.8,
                      nms_algorithm: str = 'iou',
                      best_geom_keep_area_ratio: float = 0.5,
                      pre_aggregated_output_path: str or Path = None):

        """
        Instanciate and run an Aggregator from a list of polygons for each tile.

        Parameters
        ----------
        output_path: str or Path
            The filename where to save the aggregated polygons.
            '.gpkg', '.geojson' are supported, as well as '.json' for COCO format.
        tiles_paths: List[str or Path]
            The list of paths to the tiles.
        polygons: List[List[Polygon]]
            A list of lists of polygons, where each list of polygons corresponds to a tile.
        scores: List[List[float]] or Dict[str, List[List[float]]]
            It can be a list of lists of floats if only 1 set of scores is passed, where each list of floats corresponds
            to a tile.

            It can also be a dictionary of lists of lists of floats if multiple types of scores are passed, example::

                {
                    'detection_score': [[0.1, 0.2, ...], [0.2, 0.5, ...], ...],
                    'segmentation_score': [[0.4, 0.2, ...], [0.8, 0.9, ...], ...]
                }

            This is useful if you possibly have multiple scores for each polygon, such as detection confidence score
            and/or segmentation confidence score.
        other_attributes: Dict[str, List[List[float]]]
            Used to pass other polygons information, not used by the Aggregator but used for your downstream tasks.
            Structure is similar to "scores", a dict with keys being the attribute (=output gpkg columns) names and
            values being lists of lists, one value for each associated polygon.
        scores_weights: Dict[str, float]
            The weights to apply to each score set. The scores_weights should sum to 1 and there should be as many
                weights as there are scores_names. See the 'scores_weighting_method' parameter for more details.
        scores_weighting_method: str
            The method to use to weight the different sets of scores. Supported values are ['weighted_arithmetic_mean',
            'weighted_geometric_mean', 'weighted_harmonic_mean'].

                - 'weighted_arithmetic_mean': The scores are simply averaged, but each score is weighted by its weight.
                - 'weighted_geometric_mean': The scores are multiplied, but each score is weighted by its weight.
                - 'weighted_harmonic_mean': The scores are averaged, but each score is weighted by its reciprocal
        min_centroid_distance_weight: float or None
            The weight to apply to the polygon scores (after applying the scores_weights) based on the distance
            between the polygon centroid and its tile centroid. The smaller the value of min_centroid_distance_weight,
            the smaller the score will be for polygons at the edge of the tile.
        score_threshold: float
            The score threshold under which polygons will be removed, before applying the NMS algorithm.
        nms_threshold: float
            The threshold for the Non-Maximum Suppression algorithm. It serves different purposes depending on the
            'nms_algorithm' chosen:

                - 'iou': The threshold is used as the regular Intersection Over Union (IoU) threshold.
                  If 2 polygons have an IoU above this threshold, the one with the lowest score will be removed.
                - 'ioa-disambiguate': This is the threshold used to determine if a lesser-scored polygon should be
                discarded (entirely removed), or cut to remove its portion that overlaps with another higher-scored
                polygon. The equation is a bit different than IoU: it's the area of the intersection divided by
                the area of the lesser-scored polygon (Intersection / Area_lower_scored_polygon).

        nms_algorithm: str
            The Non-Maximum Suppression algorithm to use. Supported values are ['iou' and 'ioa-disambiguate'].

                - 'iou': This algorithm uses the Intersection Over Union (IoU) to suppress overlapping polygons.
                - 'ioa-disambiguate': Shouldn't be used for detection bounding boxes,
                  it was designed for instance segmentations.
                  In addition to using IoU, this algorithm attempts to disambiguate overlapping polygons by
                  cutting the intersecting parts and keeping only the largest resulting geometry, as long as it meets the
                  `best_geom_keep_area_ratio` threshold.
        best_geom_keep_area_ratio: float
            Only used when nms_algorithm is set to 'ioa-disambiguate'. When removing part of a polygon due to
            the intersection with another one, that polygon may end-up being cut into multiple smaller and disconnected
            geometries. The algorithm will only consider keeping the largest of those geometries, and only keep it if
            it makes up at least 'best_geom_keep_area_ratio' percent of the original polygon (in terms of area).
        pre_aggregated_output_path: str or Path
            If provided, the polygons will be saved in this file before applying the NMS algorithm.
            This is useful to debug the polygons before the NMS algorithm is applied.

        Returns
        -------
        Aggregator
        """

        assert len(tiles_paths) == len(polygons), ("The number of tiles_paths must be equal than the number of lists "
                                                   "in polygons (one list of polygons per tile).")

        ids = list(range(len(tiles_paths)))
        tile_ids_to_path = {k: v for k, v in zip(ids, tiles_paths)}
        tile_ids_to_polygons = {k: v for k, v in zip(ids, polygons)}

        if type(scores) is list:
            scores = {'score': scores}

        if scores_weights:
            assert set(scores_weights.keys()) == set(scores.keys()), ("The scores_weights keys must be "
                                                                      "the same as the scores keys.")
        else:
            scores_weights = {score_name: 1 / len(scores.keys()) for score_name in scores.keys()}\
                             if type(scores) is dict else {'score': 1}

        tile_ids_to_attributes = {}
        for tile_id in ids:
            tile_attributes = {}
            for score_name, score_values in scores.items():
                tile_attributes[score_name] = score_values[tile_id]
            for attribute_name, attribute_values in other_attributes.items():
                tile_attributes[attribute_name] = attribute_values[tile_id]
            tile_ids_to_attributes[tile_id] = tile_attributes

        all_polygons_gdf, all_tiles_extents_gdf = Aggregator._prepare_polygons(
            tile_ids_to_path=tile_ids_to_path,
            tile_ids_to_polygons=tile_ids_to_polygons,
            tile_ids_to_attributes=tile_ids_to_attributes
        )

        return cls(output_path=output_path,
                   polygons_gdf=all_polygons_gdf,
                   scores_names=list(scores.keys()),
                   other_attributes_names=list(other_attributes.keys()),
                   scores_weights=[scores_weights[score_name] for score_name in scores.keys()],
                   tiles_extent_gdf=all_tiles_extents_gdf,
                   scores_weighting_method=scores_weighting_method,
                   min_centroid_distance_weight=min_centroid_distance_weight,
                   score_threshold=score_threshold,
                   nms_threshold=nms_threshold,
                   nms_algorithm=nms_algorithm,
                   tile_ids_to_path=tile_ids_to_path,
                   best_geom_keep_area_ratio=best_geom_keep_area_ratio,
                   pre_aggregated_output_path=pre_aggregated_output_path)

    @staticmethod
    def _from_coco(tiles_folder_path: str or Path,
                   coco_json_path: str or Path,
                   attributes_names: List[str]):

        # Read the JSON file
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        # Reading the polygons
        tile_ids_to_polygons = {}
        tile_ids_to_attributes = {}
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']

            annotation_polygon = decode_coco_segmentation(annotation['segmentation'])

            attributes = {}
            warn_attributes_not_found = []
            for attribute_name in attributes_names:
                if attribute_name in annotation:
                    attributes[attribute_name] = annotation[attribute_name]
                elif ("other_attributes" in annotation
                      and annotation["other_attributes"]
                      and attribute_name in annotation["other_attributes"]):
                    attributes[attribute_name] = annotation["other_attributes"][attribute_name]
                else:
                    attributes[attribute_name] = 0
                    warn_attributes_not_found.append(attribute_name)

            if len(warn_attributes_not_found) > 0:
                warn(f"The following attributes keys could not be found in the annotation, or in the 'other_attributes'"
                     f" key of the annotation: {warn_attributes_not_found}")

            if image_id in tile_ids_to_polygons:
                tile_ids_to_polygons[image_id].append(annotation_polygon)
                for attribute_name in attributes.keys():
                    tile_ids_to_attributes[image_id][attribute_name].append(attributes[attribute_name])
            else:
                tile_ids_to_polygons[image_id] = [annotation_polygon]
                tile_ids_to_attributes[image_id] = {attribute_name: [attributes[attribute_name]] for attribute_name in attributes.keys()}

        tiles_folder_path = Path(tiles_folder_path)
        tile_ids_to_path = {}
        for image in coco_data['images']:
            image_path = tiles_folder_path / image['file_name']
            assert image_path.exists(), (f"Could not find the tile '{image['file_name']}'"
                                         f" in tiles_folder_path='{tiles_folder_path}'")
            tile_ids_to_path[image['id']] = image_path

        all_polygons_gdf, all_tiles_extents_gdf = Aggregator._prepare_polygons(
            tile_ids_to_path=tile_ids_to_path,
            tile_ids_to_polygons=tile_ids_to_polygons,
            tile_ids_to_attributes=tile_ids_to_attributes
        )

        return all_polygons_gdf, all_tiles_extents_gdf, tile_ids_to_path

    @staticmethod
    def _prepare_polygons(tile_ids_to_path: dict,
                          tile_ids_to_polygons: dict,
                          tile_ids_to_attributes: dict):
        images_without_crs = 0
        all_gdfs_polygons = []
        all_gdfs_tiles_extents = []
        for tile_id in tqdm(tile_ids_to_polygons.keys(), desc='Reverting tiles polygons to CRS coordinates'):
            image_path = tile_ids_to_path[tile_id]

            src = rasterio.open(image_path)

            if not src.crs or not src.transform:
                images_without_crs += 1
                continue

            # Making sure the Tile respects the convention
            TileNameConvention.parse_name(Path(image_path).name)

            tile_gdf = gpd.GeoDataFrame(geometry=tile_ids_to_polygons[tile_id])
            tile_gdf['geometry'] = tile_gdf['geometry'].astype(object).apply(
                lambda geom: apply_affine_transform(geom, src.transform)
            )
            tile_gdf.crs = src.crs
            tile_gdf['tile_id'] = tile_id
            for attribute_name in tile_ids_to_attributes[tile_id].keys():
                tile_gdf[attribute_name] = tile_ids_to_attributes[tile_id][attribute_name]

            all_gdfs_polygons.append(tile_gdf)

            bounds = src.bounds
            # Create a polygon from the bounds
            tile_extent_polygon = Polygon([
                (bounds.left, bounds.bottom),
                (bounds.left, bounds.top),
                (bounds.right, bounds.top),
                (bounds.right, bounds.bottom)
            ])
            gdf_tile_extent = gpd.GeoDataFrame(geometry=[tile_extent_polygon], crs=src.crs)
            gdf_tile_extent['tile_id'] = tile_id
            all_gdfs_tiles_extents.append(gdf_tile_extent)

        if images_without_crs > 0:
            warn(f"The aggregator skipped {images_without_crs} tiles without a CRS or transform.")

        tiles_crs = [gdf.crs for gdf in all_gdfs_tiles_extents]
        n_crs = len(set(tiles_crs))
        if n_crs > 1:
            common_crs = tiles_crs[0].crs
            warn(f"Multiple CRS were detected (n={n_crs}), so re-projecting all polygons of all tiles to the"
                 f" same common CRS '{common_crs}', chosen from one of the tiles.")
            all_gdfs_polygons = [gdf.to_crs(common_crs) for gdf in all_gdfs_polygons if gdf.crs != common_crs]
            all_gdfs_tiles_extents = [gdf.to_crs(common_crs) for gdf in all_gdfs_tiles_extents if gdf.crs != common_crs]

        all_polygons_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs_polygons, ignore_index=True))
        all_tiles_extents_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs_tiles_extents, ignore_index=True))

        return all_polygons_gdf, all_tiles_extents_gdf

    def _validate_polygons(self):
        def fix_geometry(geom):
            if not geom.is_valid:
                fixed_geom = make_valid(geom)
                return fixed_geom
            return geom

        self.polygons_gdf['geometry'] = self.polygons_gdf['geometry'].astype(object).apply(fix_geometry)

        # Remove geometries that are still not valid after fixing
        self.polygons_gdf = self.polygons_gdf[self.polygons_gdf.is_valid]

    def _prepare_scores(self):
        if self.scores_weighting_method == 'weighted_arithmetic_mean':
            self.polygons_gdf['aggregator_score'] = sum(
                [score_weight * self.polygons_gdf[score_name].fillna(0)
                 for score_name, score_weight in zip(self.scores_names, self.scores_weights)]
            )
        elif self.scores_weighting_method == 'weighted_geometric_mean':
            self.polygons_gdf['aggregator_score'] = 1.0
            for score_name, score_weight in zip(self.scores_names, self.scores_weights):
                self.polygons_gdf['aggregator_score'] *= (
                        self.polygons_gdf[score_name].fillna(0) ** score_weight
                )
            # Take the root based on the sum of the weights
            total_weight = sum(self.scores_weights)
            self.polygons_gdf['aggregator_score'] = self.polygons_gdf['aggregator_score'] ** (1 / total_weight)
        elif self.scores_weighting_method == 'weighted_harmonic_mean':
            weighted_reciprocal_sum = sum(
                [score_weight / (self.polygons_gdf[score_name].fillna(0) + 1e-12)
                 for score_name, score_weight in zip(self.scores_names, self.scores_weights)]
            )
            total_weight = sum(self.scores_weights)
            self.polygons_gdf['aggregator_score'] = total_weight / weighted_reciprocal_sum
        else:
            raise ValueError(f"Unsupported score weighting method: {self.scores_weighting_method}")

        if self.min_centroid_distance_weight is not None:
            # Calculate centroids for tiles and polygons
            tile_centroids = self.tiles_extent_gdf.set_index('tile_id').geometry.centroid
            self.polygons_gdf['polygon_centroid'] = self.polygons_gdf.geometry.centroid
            self.polygons_gdf['tile_centroid'] = self.polygons_gdf['tile_id'].map(tile_centroids)

            # Calculate distances between polygon centroids and tile centroids
            self.polygons_gdf['centroid_distance'] = self.polygons_gdf.apply(
                lambda row: row['polygon_centroid'].distance(row['tile_centroid']), axis=1
            )

            # Calculate max possible distance (tile diagonal)
            tile_diagonals = self.tiles_extent_gdf.set_index('tile_id').geometry.apply(
                lambda geom: geom.bounds
            ).apply(
                lambda bounds: ((bounds[2] - bounds[0]) ** 2 + (bounds[3] - bounds[1]) ** 2) ** 0.5
            )
            self.polygons_gdf['max_distance'] = self.polygons_gdf['tile_id'].map(tile_diagonals)

            # Calculate distance weight and adjust scores
            self.polygons_gdf['distance_weight'] = 1 - (
                    self.polygons_gdf['centroid_distance'] / self.polygons_gdf['max_distance']
            ) * (1 - self.min_centroid_distance_weight)
            self.polygons_gdf['aggregator_score'] *= self.polygons_gdf['distance_weight']

    def _remove_low_score_polygons(self):
        if self.score_threshold:
            if "aggregator_score" in self.polygons_gdf and self.polygons_gdf["aggregator_score"].any():
                n_before = len(self.polygons_gdf)
                self.polygons_gdf.drop(self.polygons_gdf[self.polygons_gdf["aggregator_score"] < self.score_threshold].index,
                                       inplace=True)
                n_after = len(self.polygons_gdf)
                print(f"Removed {n_before - n_after} out of {n_before} polygons"
                      f" as they were under the score_threshold={self.score_threshold}.")
            else:
                warn(f"Could not apply score_threshold={self.score_threshold} as scores were not found"
                     f" in the polygons_gdf. If you instanced the Aggregator from a COCO file,"
                     f" please make sure that ALL annotations have a 'score' key or a ['other_attributes']['score']"
                     f" nested keys (ex: 'detector_score' or 'segmenter_score')."
                     f" If you instanced the Aggregator from a list of polygons for each tile,"
                     f" make sure you also pass the associated list of scores for each tile.")

    @staticmethod
    def _calculate_iou_for_geometry(gdf, geometry):
        """
        Calculate the IoU between a single geometry and all geometries in a GeoDataFrame.

        Parameters:
        - gdf: A GeoDataFrame containing multiple geometries.
        - geometry: A single geometry to compare.

        Returns:
        - A pandas Series containing the IoU values between the single_row geometry and each geometry in gdf.
        """
        # Calculate intersection areas
        intersections = gdf.geometry.intersection(geometry).area

        # Calculate union areas
        unions = gdf.geometry.area + geometry.area - intersections

        # Calculate IoU values
        iou_values = intersections / unions

        return iou_values

    def _apply_iou_nms_algorithm(self):
        gdf = self.polygons_gdf.copy()
        gdf.sort_values(by='aggregator_score', ascending=False, inplace=True)

        intersect_gdf = gpd.sjoin(gdf, gdf, how='inner', predicate='intersects')
        intersect_gdf = intersect_gdf[intersect_gdf.index != intersect_gdf['index_right']]
        intersect_gdf_ids = intersect_gdf.groupby(intersect_gdf.index)['index_right'].unique()

        skip_ids = set()
        id_to_final_polygon = {}
        progress = tqdm(total=len(gdf), desc="Applying NMS-iou algorithm")
        for current_id in gdf.index:
            current = gdf.loc[current_id]
            progress.update(1)
            if current_id in skip_ids:
                continue
            if current.geometry.area == 0:
                skip_ids.add(current_id)
                continue
            if current_id in intersect_gdf_ids.index:
                intersecting_geometries_ids = intersect_gdf_ids.loc[current_id]
                intersecting_geometries_ids = [g_id for g_id in intersecting_geometries_ids if g_id not in skip_ids]
                ious = self._calculate_iou_for_geometry(gdf.loc[intersecting_geometries_ids], current.geometry)
                skip_ids.update(list(ious[ious > self.nms_threshold].index))

            id_to_final_polygon[current_id] = current.geometry
            skip_ids.add(current_id)

        keep_ids = list(id_to_final_polygon.keys())
        final_polygons = gpd.GeoDataFrame(geometry=[id_to_final_polygon[k] for k in keep_ids],
                                          index=keep_ids)

        progress.close()

        self.polygons_gdf.loc[final_polygons.index, 'geometry'] = final_polygons['geometry']
        drop_ids = list(skip_ids - set(keep_ids))
        self.polygons_gdf.drop(drop_ids, inplace=True, errors="ignore")

    def _apply_ioa_disambiguate_nms_algorithm(self):
        gdf = self.polygons_gdf.copy()
        gdf.sort_values(by='aggregator_score', ascending=False, inplace=True)

        gdf['geometry'] = gdf['geometry'].astype(object).apply(lambda x: fix_geometry_collection(x))
        gdf['geometry'] = gdf['geometry'].astype(object).apply(lambda x: self._check_remove_bad_geometries(x))
        gdf = gdf[gdf['geometry'].notnull()]

        intersect_gdf = gpd.sjoin(gdf, gdf, how='inner', predicate='intersects')
        intersect_gdf = intersect_gdf[intersect_gdf.index != intersect_gdf['index_right']]
        intersect_gdf_ids = intersect_gdf.groupby(intersect_gdf.index)['index_right'].unique()

        skip_ids = set()
        id_to_final_polygon = {}
        progress = tqdm(total=len(gdf), desc="Applying NMS-iou-disambiguate algorithm")
        for current_id in gdf.index:
            current = gdf.loc[current_id]
            progress.update(1)
            if current_id in skip_ids:
                continue

            if current_id in intersect_gdf_ids.index:
                intersecting_geometries_ids = intersect_gdf_ids.loc[current_id]
                intersecting_geometries_ids = [g_id for g_id in intersecting_geometries_ids if g_id not in skip_ids]
                if len(intersecting_geometries_ids) != 0:
                    for g_id in intersecting_geometries_ids:
                        # We have to re-compute the IoA as intersect_gdf might not be up-to-date anymore after some polygons were modified in previous iterations.
                        with warnings.catch_warnings(record=True) as w:
                            gdf.at[current_id, 'geometry'] = fix_geometry_collection(
                                gdf.at[current_id, 'geometry'])
                            try:
                                intersection = gdf.at[current_id, 'geometry'].intersection(gdf.at[g_id, 'geometry'])
                            except shapely.errors.GEOSException:
                                # 'invalid value encountered in intersection'
                                print('* Skipped polygon matching ids {}/{} for shapely intersection error. *'.format(current_id, g_id))

                        intersection = fix_geometry_collection(intersection)

                        if not self._check_remove_bad_geometries(intersection):
                            continue

                        # Instead of using IOU in the SegmenterAggregator, we only look at the intersection area over the lower-scored geometry area (IoA)
                        updated_intersection_over_geom_area = intersection.area / gdf.at[g_id, 'geometry'].area
                        if updated_intersection_over_geom_area > self.nms_threshold:
                            skip_ids.add(g_id)
                        else:
                            try:
                                gdf.at[current_id, 'geometry'] = gdf.at[current_id, 'geometry'].union(intersection)
                                new_geometry = gdf.at[g_id, 'geometry'].difference(intersection)
                            except shapely.errors.GEOSException:
                                # 'TopologyException: found non-noded intersection'
                                print('* Skipped polygon union between {} and {} for shapely union error. *'.format(current_id, g_id))
                                skip_ids.add(g_id)
                                continue

                            new_geometry = fix_geometry_collection(new_geometry)

                            if not new_geometry.is_valid:
                                new_geometry = make_valid(new_geometry)

                            if new_geometry.geom_type == 'MultiPolygon':
                                # If the largest Polygon represent more than best_geom_keep_area_ratio% of the total area,
                                # we only keep that Polygon
                                largest_polygon = max(new_geometry.geoms, key=lambda x: x.area)
                                if largest_polygon.area / new_geometry.area > self.best_geom_keep_area_ratio:
                                    gdf.at[g_id, 'geometry'] = largest_polygon
                                else:
                                    skip_ids.add(g_id)
                            elif new_geometry.area > 0:
                                gdf.at[g_id, 'geometry'] = new_geometry
                            else:
                                skip_ids.add(g_id)

            id_to_final_polygon[current_id] = current.geometry
            skip_ids.add(current_id)

        keep_ids = list(id_to_final_polygon.keys())
        final_polygons = gpd.GeoDataFrame(geometry=[id_to_final_polygon[k] for k in keep_ids],
                                          index=keep_ids)

        progress.close()

        self.polygons_gdf.loc[final_polygons.index, 'geometry'] = final_polygons['geometry']
        drop_ids = list(skip_ids - set(keep_ids))
        self.polygons_gdf.drop(drop_ids, inplace=True, errors="ignore")

    @staticmethod
    def _check_remove_bad_geometries(geometry):
        if not geometry.is_valid or geometry.area == 0:
            return None
        else:
            return geometry

    @staticmethod
    def _calculate_centroid_distance(gdf, geometry):
        """
        Calculate the distance between the centroid of each geometry in a GeoDataFrame and another single geometry.

        Parameters:
        - gdf: A GeoDataFrame containing multiple geometries with a 'centroid' column.
        - geometry: A single geometry to compare.

        Returns:
        - A pandas Series containing the distance values between the centroid of each geometry in gdf and the given geometry.
        """
        # Ensure the 'centroid' column exists, calculate if not present
        if 'centroid' not in gdf.columns:
            gdf['centroid'] = gdf.geometry.centroid

        # Calculate the centroid of the provided geometry
        geometry_centroid = geometry.centroid

        # Calculate distances
        distances = gdf['centroid'].distance(geometry_centroid)

        return distances

    def _apply_nms_algorithm(self):
        n_before = len(self.polygons_gdf)
        if self.nms_algorithm == "iou":
            self._apply_iou_nms_algorithm()
        if self.nms_algorithm == "ioa-disambiguate":
            self._apply_ioa_disambiguate_nms_algorithm()

            # TODO add support for adapative-nms: https://arxiv.org/pdf/1904.03629.pdf

        n_after = len(self.polygons_gdf)
        print(f"Removed {n_before - n_after} out of {n_before} polygons"
              f" by applying the Non-Maximum Suppression-{self.nms_algorithm} algorithm.")

    def _save_polygons(self):
        self._validate_polygons()

        if 'polygon_centroid' in self.polygons_gdf:
            self.polygons_gdf.drop(columns=['polygon_centroid'], inplace=True)
        if 'tile_centroid' in self.polygons_gdf:
            self.polygons_gdf.drop(columns=['tile_centroid'], inplace=True)

        self.polygons_gdf.set_geometry('geometry', inplace=True)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.output_path.suffix == '.geojson':
            self.polygons_gdf.to_file(str(self.output_path), driver="GeoJSON")
        elif self.output_path.suffix == '.gpkg':
            self.polygons_gdf.to_file(str(self.output_path), driver="GPKG")
        elif self.output_path.suffix == '.json':
            tiles_ids = self.polygons_gdf['tile_id'].unique()

            attributes_names = self.scores_names + self.other_attributes_names
            other_attributes = []
            for tile_id in tiles_ids:
                tile_attributes = []
                for label_index in self.polygons_gdf[self.polygons_gdf['tile_id'] == tile_id].index:
                    label_data = self.polygons_gdf.loc[label_index]
                    label_attributes = {'aggregator_score': label_data['aggregator_score']}
                    label_attributes.update({attribute_name: label_data[attribute_name].tolist() for attribute_name in attributes_names})
                    tile_attributes.append(label_attributes)

                other_attributes.append(tile_attributes)

            coco_generator = COCOGenerator(
                description=f"Aggregated polygons from multiple tiles.",
                tiles_paths=[self.tile_ids_to_path[tile_id] for tile_id in tiles_ids],
                polygons=[apply_inverse_transform(
                    polygons=self.polygons_gdf[self.polygons_gdf['tile_id'] == tile_id]['geometry'].tolist(),
                    raster_path=self.tile_ids_to_path[tile_id]) for tile_id in tiles_ids],
                scores=None,  # Using other_attributes instead
                categories=None,  # TODO add support for categories
                other_attributes=other_attributes,
                output_path=self.output_path,
                use_rle_for_labels=True,  # TODO make this a parameter to the class
                n_workers=5,  # TODO make this a parameter to the class
                coco_categories_list=None  # TODO make this a parameter to the class
            )
            coco_generator.generate_coco()
        else:
            raise Exception(
                "The output_path needs to end with either .json (coco output) or .geojson (geopackage output).")
        print(f"Saved {len(self.polygons_gdf)} polygons in file '{self.output_path}'.")


class DetectorAggregator:
    def __init__(self, *args, **kwargs):
        raise Exception('This class is now deprecated (since v0.1.4). Please use the Aggregator class instead.')

    @classmethod
    def from_coco(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    @classmethod
    def from_polygons(cls, *args, **kwargs):
        return cls(*args, **kwargs)


class SegmentationAggregator:
    def __init__(self, *args, **kwargs):
        raise Exception('This class is now deprecated (since v0.1.4). Please use the Aggregator class instead.')

    @classmethod
    def from_coco(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    @classmethod
    def from_polygons(cls, *args, **kwargs):
        return cls(*args, **kwargs)

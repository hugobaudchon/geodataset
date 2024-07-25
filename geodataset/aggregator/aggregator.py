import json
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict
from warnings import warn

import numpy as np
import pandas as pd
import rasterio
import shapely
from shapely import box, Polygon, MultiPolygon, GeometryCollection
import geopandas as gpd
from tqdm import tqdm

from geodataset.utils import TileNameConvention, apply_affine_transform, COCOGenerator, coco_rle_segmentation_to_polygon


class AggregatorBase(ABC):
    SUPPORTED_NMS_ALGORITHMS = ['iou']  # 'diou'

    def __init__(self,
                 output_path: str or Path,
                 polygons_gdf: gpd.GeoDataFrame,
                 scores_names: List[str],
                 scores_weights: List[float],
                 tiles_extent_gdf: gpd.GeoDataFrame,
                 tile_ids_to_path: dict or None,
                 score_threshold: float = 0.1,
                 nms_threshold: float = 0.8,
                 nms_algorithm: str = 'iou'):

        self.output_path = Path(output_path)
        self.polygons_gdf = polygons_gdf
        self.scores_names = scores_names
        self.scores_weights = scores_weights
        self.tiles_extent_gdf = tiles_extent_gdf
        self.tile_ids_to_path = tile_ids_to_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.nms_algorithm = nms_algorithm

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

        assert self.output_path.suffix in ['.json', '.geojson'], ("The output_path needs to end with either .json"
                                                                  " (coco output) or .geojson (geopackage output).")

        if self.output_path.suffix == '.json':
            assert self.tile_ids_to_path is not None, \
                "The tile_ids_to_path must be provided to save the polygons in COCO format."

    @staticmethod
    def _from_coco(polygon_type: str,
                   tiles_folder_path: Path,
                   coco_json_path: Path,
                   scores_names: List[str]):

        # Read the JSON file
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        # Reading the polygons
        tile_ids_to_polygons = {}
        tile_ids_to_scores = {}
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']

            if polygon_type == 'box':
                xmin = annotation['bbox'][0]
                xmax = annotation['bbox'][0] + annotation['bbox'][2]
                ymin = annotation['bbox'][1]
                ymax = annotation['bbox'][1] + annotation['bbox'][3]
                annotation_polygon = box(xmin, ymin, xmax, ymax)
            elif polygon_type == 'segmentation':
                annotation_polygon = coco_rle_segmentation_to_polygon(annotation['segmentation'])
            else:
                raise Exception(f"The polygon_type '{polygon_type}' is not supported. "
                                f"Supported values are ['box', 'segmentation'].")

            scores = {}
            warn_scores_not_found = []
            for score_name in scores_names:
                if score_name in annotation:
                    scores[score_name] = annotation["score_name"]
                elif ("other_attributes" in annotation
                      and annotation["other_attributes"]
                      and score_name in annotation["other_attributes"]):
                    scores[score_name] = annotation["other_attributes"][score_name]
                else:
                    scores[score_name] = 0
                    warn_scores_not_found.append(score_name)

            if len(warn_scores_not_found) > 0:
                warn(f"The following scores keys could not be found in the annotation, or in the 'other_attributes' key"
                     f" of the annotation: {warn_scores_not_found}")

            if image_id in tile_ids_to_polygons:
                tile_ids_to_polygons[image_id].append(annotation_polygon)
                for score_name in scores.keys():
                    tile_ids_to_scores[image_id][score_name].append(scores[score_name])
            else:
                tile_ids_to_polygons[image_id] = [annotation_polygon]
                tile_ids_to_scores[image_id] = {score_name: [scores[score_name]] for score_name in scores.keys()}

        tile_ids_to_path = {}
        for image in coco_data['images']:
            image_path = tiles_folder_path / image['file_name']
            assert image_path.exists(), (f"Could not find the tile '{image['file_name']}'"
                                         f" in tiles_folder_path='{tiles_folder_path}'")
            tile_ids_to_path[image['id']] = image_path

        all_polygons_gdf, all_tiles_extents_gdf = AggregatorBase._prepare_polygons(
            tile_ids_to_path=tile_ids_to_path,
            tile_ids_to_polygons=tile_ids_to_polygons,
            tile_ids_to_scores=tile_ids_to_scores)

        return all_polygons_gdf, all_tiles_extents_gdf, tile_ids_to_path

    @classmethod
    def from_polygons(cls,
                      output_path: Path,
                      tiles_paths: List[Path],
                      polygons: List[List[Polygon]],
                      scores: List[List[float]] or Dict[str, List[List[float]]],
                      scores_weights: Dict[str, float] = None,
                      score_threshold: float = 0.1,
                      nms_threshold: float = 0.8,
                      nms_algorithm: str = 'iou'):

        """
        Instanciate and run an Aggregator from a list of polygons for each tile.

        Parameters
        ----------
        output_path: str or Path
            The filename where to save the aggregated polygons.
            '.gpkg', '.geojson' are supported, as well as '.json' for COCO format.
        tiles_paths: List[Path]
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

            This is useful if you have multiple scores for each polygon, such as detection confidence score and
            segmentation confidence score.
        scores_weights: Dict[str, float]
            The weights to apply to each score column in the polygons_gdf. Weights are exponents.
            Only used if 'scores' is a dict and has more than 1 key (more than 1 set of scores).
            There keys of the dict should be the same as the keys of the 'scores' dict.

            The equation is::

                score = (score1 ** weight1) * (score2 ** weight2) * ...
        score_threshold: float
            The threshold under which polygons will be removed.
        nms_threshold: float
            The threshold for the Non-Maximum Suppression algorithm.
        nms_algorithm: str
            The Non-Maximum Suppression algorithm to use. Supported values are ['iou'].

        Returns
        -------
        AggregatorBase instance (DetectorAggregator or SegmentationAggregator...)
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
            assert all([w >= 1.0 for w in scores_weights.values()]), "All scores_weights values should be > 1."

        else:
            scores_weights = {score_name: 1 for score_name in scores.keys()} if type(scores) is dict else {'score': 1}

        tile_ids_to_scores = {}
        for tile_id in ids:
            id_scores = {}
            for score_name, score_values in scores.items():
                id_scores[score_name] = score_values[tile_id]
            tile_ids_to_scores[tile_id] = id_scores

        all_polygons_gdf, all_tiles_extents_gdf = AggregatorBase._prepare_polygons(
            tile_ids_to_path=tile_ids_to_path,
            tile_ids_to_polygons=tile_ids_to_polygons,
            tile_ids_to_scores=tile_ids_to_scores)

        return cls(output_path=output_path,
                   polygons_gdf=all_polygons_gdf,
                   scores_names=list(scores.keys()),
                   scores_weights=[scores_weights[score_name] for score_name in scores.keys()],
                   tiles_extent_gdf=all_tiles_extents_gdf,
                   score_threshold=score_threshold,
                   nms_threshold=nms_threshold,
                   nms_algorithm=nms_algorithm,
                   tile_ids_to_path=tile_ids_to_path)

    @staticmethod
    def _prepare_polygons(tile_ids_to_path: dict, tile_ids_to_polygons: dict, tile_ids_to_scores: dict):
        images_without_crs = 0
        all_gdfs_polygons = []
        all_gdfs_tiles_extents = []
        for tile_id in tile_ids_to_polygons.keys():
            image_path = tile_ids_to_path[tile_id]

            src = rasterio.open(image_path)

            if not src.crs or not src.transform:
                images_without_crs += 1
                continue

            # Making sure the Tile respects the convention
            TileNameConvention.parse_name(Path(image_path).name)

            gdf_polygons = gpd.GeoDataFrame(geometry=tile_ids_to_polygons[tile_id])
            gdf_polygons['geometry'] = gdf_polygons['geometry'].astype(object).apply(
                lambda geom: apply_affine_transform(geom, src.transform)
            )
            gdf_polygons.crs = src.crs
            gdf_polygons['tile_id'] = tile_id
            for score_name in tile_ids_to_scores[tile_id].keys():
                gdf_polygons[score_name] = tile_ids_to_scores[tile_id][score_name]
            all_gdfs_polygons.append(gdf_polygons)

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
                fixed_geom = geom.buffer(0)
                return fixed_geom
            return geom

        self.polygons_gdf['geometry'] = self.polygons_gdf['geometry'].astype(object).apply(fix_geometry)

    def _prepare_scores(self):
        self.polygons_gdf['score'] = [(s if s else 0) ** self.scores_weights[0] for s in
                                      self.polygons_gdf[self.scores_names[0]]]
        for score_name, score_weight in zip(self.scores_names[1:], self.scores_weights[1:]):
            self.polygons_gdf['score'] *= [(s if s else 0) ** score_weight for s in self.polygons_gdf[score_name]]

        # normalize the 'score' column between 0 and 1
        self.polygons_gdf['score'] = (self.polygons_gdf['score'] - self.polygons_gdf['score'].min()) / \
                                     (self.polygons_gdf['score'].max() - self.polygons_gdf['score'].min())

    def _remove_low_score_polygons(self):
        if self.score_threshold:
            if "score" in self.polygons_gdf and self.polygons_gdf["score"].any():
                n_before = len(self.polygons_gdf)
                self.polygons_gdf.drop(self.polygons_gdf[self.polygons_gdf["score"] < self.score_threshold].index,
                                       inplace=True)
                n_after = len(self.polygons_gdf)
                print(f"Removed {n_before - n_after} out of {n_before} polygons"
                      f" as they were under the score_threshold={self.score_threshold}.")
            else:
                warn(f"Could not apply score_threshold={self.score_threshold} as scores were not found"
                     f" in the polygons_gdf. If you instanced the Aggregator from a COCO file,"
                     f" please make sure that ALL annotations have a 'score' key or a ['other_attributes']['score']"
                     f" nested keys. If you instanced the Aggregator from a list of polygons for each tile,"
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
        if self.nms_algorithm == "diou":
            self._apply_diou_nms_algorithm()

            # TODO add support for adapative-nms: https://arxiv.org/pdf/1904.03629.pdf

        n_after = len(self.polygons_gdf)
        print(f"Removed {n_before - n_after} out of {n_before} polygons"
              f" by applying the Non-Maximum Suppression-{self.nms_algorithm} algorithm.")

    @abstractmethod
    def _apply_iou_nms_algorithm(self):
        pass

    @abstractmethod
    def _apply_diou_nms_algorithm(self):
        pass

    def _save_polygons(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.output_path.suffix == '.geojson':
            self.polygons_gdf.to_file(str(self.output_path), driver="GeoJSON")
        elif self.output_path.suffix == '.gpkg':
            self.polygons_gdf.to_file(str(self.output_path), driver="GPKG")
        elif self.output_path.suffix == '.json':
            tiles_ids = self.polygons_gdf['tile_id'].unique()

            other_attributes = []
            for tile_id in tiles_ids:
                tile_scores = []
                for label_index in self.polygons_gdf[self.polygons_gdf['tile_id'] == tile_id].index:
                    label_data = self.polygons_gdf.loc[label_index]
                    label_scores = {score_name: label_data[score_name] for score_name in self.scores_names}
                    label_scores['score'] = label_data['score']
                    tile_scores.append(label_scores)

                other_attributes.append(tile_scores)

            coco_generator = COCOGenerator(
                description=f"Aggregated polygons from multiple tiles.",
                tiles_paths=[self.tile_ids_to_path[tile_id] for tile_id in tiles_ids],
                polygons=[self._apply_inverse_transform(
                    polygons=self.polygons_gdf[self.polygons_gdf['tile_id'] == tile_id]['geometry'].tolist(),
                    tile_path=self.tile_ids_to_path[tile_id]) for tile_id in tiles_ids],
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

    @staticmethod
    def _apply_inverse_transform(polygons: List[Polygon], tile_path: Path):
        src = rasterio.open(tile_path)
        inverse_transform = ~src.transform
        polygons = [apply_affine_transform(polygon, inverse_transform) for polygon in polygons]
        return polygons


class DetectorAggregator(AggregatorBase):
    """
    Class to aggregate detection polygons from multiple tiles, with different pixel coordinate systems,
    into a single GeoDataFrame with a common CRS, applying the Non-Maximum Suppression (NMS) algorithm.

    This class should be instantiated using the 'from_coco' or 'from_polygons' class methods.
    """

    def __init__(self,
                 output_path: str or Path,
                 polygons_gdf: gpd.GeoDataFrame,
                 scores_names: List[str],
                 scores_weights: List[float],
                 tiles_extent_gdf: gpd.GeoDataFrame,
                 tile_ids_to_path: dict or None,
                 score_threshold: float = 0.1,
                 nms_threshold: float = 0.8,
                 nms_algorithm: str = 'iou'
                 ):
        super().__init__(output_path=output_path,
                         polygons_gdf=polygons_gdf,
                         scores_names=scores_names,
                         scores_weights=scores_weights,
                         tiles_extent_gdf=tiles_extent_gdf,
                         tile_ids_to_path=tile_ids_to_path,
                         score_threshold=score_threshold,
                         nms_threshold=nms_threshold,
                         nms_algorithm=nms_algorithm)

    @classmethod
    def from_coco(cls,
                  output_path: Path,
                  tiles_folder_path: Path,
                  coco_json_path: Path,
                  scores_names: List[str] = None,
                  scores_weights: List[float] = None,
                  score_threshold: float = 0.1,
                  nms_threshold: float = 0.8,
                  nms_algorithm: str = 'iou'):

        """
        Instanciate and run a DetectorAggregator from a COCO JSON file.
        The polygons will be read from the coco_json_path, and the tiles will be read from the tiles_folder_path.

        Parameters
        ----------
        output_path: str or Path
            The filename where to save the aggregated polygons.
            '.gpkg', '.geojson' are supported, as well as '.json' for COCO format.
        tiles_folder_path: Path
            The folder where the tiles are stored.
        coco_json_path: Path
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

        scores_weights: List[float]
            The weights to apply to each score column in the polygons_gdf. Weights are exponents.
            There should be as many weights as there are scores_names.

            The equation is::

                score = (score1 ** weight1) * (score2 ** weight2) * ...
        score_threshold: float
            The threshold under which polygons will be removed.
        nms_threshold: float
            The threshold for the Non-Maximum Suppression algorithm.
        nms_algorithm: str
            The NMS algorithm to use. Only 'iou' is currently supported.

        Returns
        -------
        DetectorAggregator
        """

        if scores_names is None:
            scores_names = ['score']

        if scores_weights:
            assert len(scores_names) == len(scores_weights), ("The scores_weights must have "
                                                              "the same length as the scores_names.")
        else:
            scores_weights = [1,] * len(scores_names)

        all_polygons_gdf, all_tiles_extents_gdf, tile_ids_to_path = cls._from_coco(
            polygon_type='box',
            tiles_folder_path=tiles_folder_path,
            coco_json_path=coco_json_path,
            scores_names=scores_names
        )

        return cls(output_path=output_path,
                   polygons_gdf=all_polygons_gdf,
                   scores_names=scores_names,
                   scores_weights=scores_weights,
                   tiles_extent_gdf=all_tiles_extents_gdf,
                   score_threshold=score_threshold,
                   nms_threshold=nms_threshold,
                   nms_algorithm=nms_algorithm,
                   tile_ids_to_path=tile_ids_to_path)

    def _apply_iou_nms_algorithm(self):
        gdf = self.polygons_gdf.copy()
        gdf.sort_values(by='score', ascending=False, inplace=True)

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

    def _apply_diou_nms_algorithm(self):
        raise NotImplementedError("This algorithm has to be updated to follow the optimizations made for iou_nms.")

        gdf = self.polygons_gdf.copy()
        gdf.sort_values(by='score', ascending=False, inplace=True)

        gdf["centroid"] = gdf.geometry.centroid
        max_distance = np.sqrt(((gdf.total_bounds[2] - gdf.total_bounds[0]) ** 2 +
                                (gdf.total_bounds[3] - gdf.total_bounds[1]) ** 2))

        intersect_gdf = gpd.sjoin(gdf, gdf, how='inner', predicate='intersects')
        intersect_gdf = intersect_gdf[intersect_gdf.index != intersect_gdf['index_right']]

        keep_ids = set()
        skip_ids = set()
        progress = tqdm(total=len(gdf), desc="Applying NMS-diou algorithm")
        while not gdf.empty:
            current = gdf.iloc[0]
            current_id = gdf.index[0]
            progress.update(1)
            if current_id in skip_ids:
                gdf.drop(current_id, inplace=True, errors="ignore")
                continue
            if len(gdf) > 1:
                intersecting_geometries_ids = intersect_gdf[intersect_gdf.index == current_id]['index_right'].unique()
                intersecting_geometries_ids = [g_id for g_id in intersecting_geometries_ids if g_id not in skip_ids]
                ious = self._calculate_iou_for_geometry(gdf.loc[intersecting_geometries_ids], current.geometry)
                center_dists = self._calculate_centroid_distance(gdf.loc[intersecting_geometries_ids], current.geometry)
                dious = ious - (center_dists / max_distance) ** 2
                skip_ids.update(list(dious[dious > self.nms_threshold].index))

            keep_ids.add(current_id)
            skip_ids.add(current_id)
            gdf.drop(current_id, inplace=True, errors="ignore")

        progress.close()
        drop_ids = list(skip_ids - keep_ids)
        self.polygons_gdf.drop(drop_ids, inplace=True, errors="ignore")


class SegmentationAggregator(AggregatorBase):
    """
    Class to aggregate segmentation polygons from multiple tiles, with different pixel coordinate systems,
    into a single GeoDataFrame with a common CRS, applying the Non-Maximum Suppression (NMS) algorithm.

    This class should be instantiated using the 'from_coco' or 'from_polygons' class methods.
    """
    def __init__(self,
                 output_path: Path,
                 polygons_gdf: gpd.GeoDataFrame,
                 scores_names: List[str],
                 scores_weights: List[float],
                 tiles_extent_gdf: gpd.GeoDataFrame,
                 tile_ids_to_path: dict or None,
                 score_threshold: float = 0.1,
                 nms_threshold: float = 0.8,
                 nms_algorithm: str = 'iou',
                 best_geom_keep_area_ratio: float = 0.8
                 ):

        self.best_geom_keep_area_ratio = best_geom_keep_area_ratio

        super().__init__(output_path=output_path,
                         polygons_gdf=polygons_gdf,
                         scores_names=scores_names,
                         scores_weights=scores_weights,
                         tiles_extent_gdf=tiles_extent_gdf,
                         tile_ids_to_path=tile_ids_to_path,
                         score_threshold=score_threshold,
                         nms_threshold=nms_threshold,
                         nms_algorithm=nms_algorithm)

    @classmethod
    def from_coco(cls,
                  output_path: Path,
                  tiles_folder_path: Path,
                  coco_json_path: Path,
                  scores_names: List[str] = None,
                  scores_weights: List[float] = None,
                  score_threshold: float = 0.1,
                  nms_threshold: float = 0.8,
                  nms_algorithm: str = 'iou'):

        """
        Instanciate and run a SegmentationAggregator from a COCO JSON file.
        The polygons will be read from the coco_json_path, and the tiles will be read from the tiles_folder_path.

        Parameters
        ----------
        output_path: str or Path
            The filename where to save the aggregated polygons.
            '.gpkg', '.geojson' are supported, as well as '.json' for COCO format.
        tiles_folder_path: Path
            The folder where the tiles are stored.
        coco_json_path: Path
            The path to the COCO JSON file.
        scores_names: List[str]
            The names of the attributes in the COCO file annotations which should be used as scores.
            The code will look directly in the COCO annotations for these keys,
            and will also look in the 'other_attributes' key if the key is not found directly in the annotation.

            For example, if score_names = ['segmentation_score'], the 'segmentation_score' attribute can be in 2
            different places in each COCO annotation::

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
                        "segmentation_score": 0.8      # <- directly in the annotation
                        "other_attributes": {
                            "segmentation_score": 0.8  # <- in the 'other_attributes' key
                        }
                    },
                    ...
                ]

        scores_weights: List[float]
            The weights to apply to each score column in the polygons_gdf. Weights are exponents.
            There should be as many weights as there are scores_names.
            This is only useful if there are more than 1 set of scores.

            The equation is:

            score = (score1 ** weight1) * (score2 ** weight2) * ...
        score_threshold: float
            The threshold under which polygons will be removed.
        nms_threshold: float
            The threshold for the Non-Maximum Suppression algorithm.
        nms_algorithm: str
            The NMS algorithm to use. Only 'iou' is currently supported.

        Returns
        -------
        SegmentationAggregator
        """

        if scores_names is None:
            scores_names = ['score']

        if scores_weights:
            assert len(scores_names) == len(scores_weights), ("The scores_weights must have "
                                                              "the same length as the scores_names.")
        else:
            scores_weights = [1,] * len(scores_names)

        all_polygons_gdf, all_tiles_extents_gdf, tile_ids_to_path = cls._from_coco(
            polygon_type='segmentation',
            tiles_folder_path=tiles_folder_path,
            coco_json_path=coco_json_path,
            scores_names=scores_names
        )

        return cls(output_path=output_path,
                   polygons_gdf=all_polygons_gdf,
                   scores_names=scores_names,
                   scores_weights=scores_weights,
                   tiles_extent_gdf=all_tiles_extents_gdf,
                   score_threshold=score_threshold,
                   nms_threshold=nms_threshold,
                   nms_algorithm=nms_algorithm,
                   tile_ids_to_path=tile_ids_to_path)

    def _apply_iou_nms_algorithm(self):
        gdf = self.polygons_gdf.copy()
        gdf.sort_values(by='score', ascending=False, inplace=True)

        gdf['geometry'] = gdf['geometry'].astype(object).apply(lambda x: self._check_geometry_collection(x))
        gdf['geometry'] = gdf['geometry'].astype(object).apply(lambda x: self._check_remove_bad_geometries(x))
        gdf = gdf[gdf['geometry'].notnull()]

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

            if current_id in intersect_gdf_ids.index:
                intersecting_geometries_ids = intersect_gdf_ids.loc[current_id]
                intersecting_geometries_ids = [g_id for g_id in intersecting_geometries_ids if g_id not in skip_ids]
                if len(intersecting_geometries_ids) != 0:
                    for g_id in intersecting_geometries_ids:
                        # We have to re-compute the IOU as intersect_gdf might not be up-to-date anymore after some polygons were modified in previous iterations.
                        with warnings.catch_warnings(record=True) as w:
                            gdf.at[current_id, 'geometry'] = self._check_geometry_collection(
                                gdf.at[current_id, 'geometry'])
                            try:
                                intersection = gdf.at[current_id, 'geometry'].intersection(gdf.at[g_id, 'geometry'])
                            except shapely.errors.GEOSException:
                                # 'invalid value encountered in intersection'
                                print('* Skipped polygon matching ids {}/{} for shapely intersection error. *'.format(current_id, g_id))

                        intersection = self._check_geometry_collection(intersection)

                        if not self._check_remove_bad_geometries(intersection):
                            continue

                        # Instead of using IOU in the SegmenterAggregator, we only look at the intersection area over the lower-scored geometry area
                        updated_intersection_over_geom_area = intersection.area / gdf.at[g_id, 'geometry'].area
                        if updated_intersection_over_geom_area > self.nms_threshold:
                            skip_ids.add(g_id)
                        else:
                            try:
                                gdf.at[current_id, 'geometry'] = gdf.at[current_id, 'geometry'].union(intersection)
                                new_geometry = gdf.at[g_id, 'geometry'].difference(intersection)
                            except shapely.errors.GEOSException:
                                # 'TopologyException: found non-noded intersection'
                                print('* Skipped polygon union between {} and {} '
                                      'for shapely union error. *'.format(current_id, g_id))

                            new_geometry = self.check_geometry_collection(new_geometry)

                            if not new_geometry.is_valid:
                                new_geometry = new_geometry.buffer(0)

                            if new_geometry.geom_type == 'MultiPolygon':
                                # If the largest Polygon represent more than 80% of the total area, we only keep that Polygon
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

    def _apply_diou_nms_algorithm(self):
        raise NotImplementedError("The DIOU NMS algorithm has to be implemented for polygon_type='segmentation'.")

    @staticmethod
    def _check_remove_bad_geometries(geometry):
        if not geometry.is_valid or geometry.area == 0:
            return None
        else:
            return geometry

    @staticmethod
    def _check_geometry_collection(geometry: shapely.Geometry):
        if geometry.geom_type == 'GeometryCollection':
            final_geoms = []
            # Iterate through each geometry in the collection
            for geom in geometry.geoms:
                if geom.geom_type == 'Polygon':
                    final_geoms.append(geom)
                elif geom.geom_type == 'LineString':
                    # Check if the LineString is closed and can be considered a polygon
                    if geom.is_ring:
                        # Convert the LineString to a Polygon
                        final_geoms.append(Polygon(geom))
            return MultiPolygon(final_geoms)
        else:
            return geometry

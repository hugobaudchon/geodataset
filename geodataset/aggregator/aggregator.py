import json
from pathlib import Path
from typing import List
from warnings import warn

import numpy as np
import pandas as pd
import rasterio
from shapely import box, Polygon
import geopandas as gpd
from tqdm import tqdm

from geodataset.utils import TileNameConvention, apply_affine_transform


class DetectionAggregator:
    SUPPORTED_NMS_ALGORITHMS = ['iou', 'diou']

    def __init__(self,
                 geojson_output_path: Path,
                 boxes_gdf: gpd.GeoDataFrame,
                 tiles_extent_gdf: gpd.GeoDataFrame,
                 score_threshold: float = 0.1,
                 nms_threshold: float = 0.8,
                 nms_algorithm: str = 'iou'):

        self.geojson_output_path = geojson_output_path
        self.boxes_gdf = boxes_gdf
        self.tiles_extent_gdf = tiles_extent_gdf
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.nms_algorithm = nms_algorithm

        assert self.boxes_gdf.crs, "The provided boxes_gdf doesn't have a CRS."
        assert 'tile_id' in self.boxes_gdf, "The provided boxes_gdf doesn't have a 'tile_id' column."
        assert self.tiles_extent_gdf.crs, "The provided tiles_extent_gdf doesn't have a CRS."
        assert 'tile_id' in self.tiles_extent_gdf, "The provided tiles_extent_gdf doesn't have a 'tile_id' column."
        assert self.nms_algorithm in self.SUPPORTED_NMS_ALGORITHMS, \
            f"The nms_algorithm must be one of {self.SUPPORTED_NMS_ALGORITHMS}. Got {self.nms_algorithm}."

        self.remove_low_score_boxes()
        self.apply_nms_algorithm()
        self.save_boxes()

    @classmethod
    def from_coco(cls,
                  geojson_output_path: Path,
                  tiles_folder_path: Path,
                  coco_json_path: Path,
                  score_threshold: float = 0.1,
                  nms_threshold: float = 0.8,
                  nms_algorithm: str = 'iou'):

        # Read the JSON file
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        # Reading the boxes
        tile_ids_to_boxes = {}
        tile_ids_to_scores = {}
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']

            xmin = annotation['bbox'][0]
            xmax = annotation['bbox'][0] + annotation['bbox'][2]
            ymin = annotation['bbox'][1]
            ymax = annotation['bbox'][1] + annotation['bbox'][3]
            annotation_box = box(xmin, ymin, xmax, ymax)

            if "score" in annotation:
                score = annotation["score"]
            elif ("other_attributes" in annotation
                  and annotation["other_attributes"]
                  and "score" in annotation["other_attributes"]):
                score = annotation["other_attributes"]["score"]
            else:
                score = None

            if image_id in tile_ids_to_boxes:
                tile_ids_to_boxes[image_id].append(annotation_box)
                tile_ids_to_scores[image_id].append(score)
            else:
                tile_ids_to_boxes[image_id] = [annotation_box]
                tile_ids_to_scores[image_id] = [score]

        tile_ids_to_path = {}
        for image in coco_data['images']:
            image_path = tiles_folder_path / image['file_name']
            assert image_path.exists(), (f"Could not find the tile '{image['file_name']}'"
                                         f" in tiles_folder_path='{tiles_folder_path}'")
            tile_ids_to_path[image['id']] = image_path

        all_boxes_gdf, all_tiles_extents_gdf = DetectionAggregator.prepare_boxes(tile_ids_to_path=tile_ids_to_path,
                                                                                 tile_ids_to_boxes=tile_ids_to_boxes,
                                                                                 tile_ids_to_scores=tile_ids_to_scores)

        return cls(geojson_output_path=geojson_output_path,
                   boxes_gdf=all_boxes_gdf,
                   tiles_extent_gdf=all_tiles_extents_gdf,
                   score_threshold=score_threshold,
                   nms_threshold=nms_threshold,
                   nms_algorithm=nms_algorithm)

    @classmethod
    def from_boxes(cls,
                   geojson_output_path: Path,
                   tiles_paths: List[Path],
                   boxes: List[List[box]],
                   scores: List[List[float]],
                   score_threshold: float = 0.1,
                   nms_threshold: float = 0.8,
                   nms_algorithm: str = 'iou'):

        assert len(tiles_paths) == len(boxes), ("The number of tiles_paths must be equal than the number of lists "
                                                "in boxes (one list of boxes per tile).")

        ids = list(range(len(tiles_paths)))
        tile_ids_to_path = {k: v for k, v in zip(ids, tiles_paths)}
        tile_ids_to_boxes = {k: v for k, v in zip(ids, boxes)}
        tile_ids_to_scores = {k: v for k, v in zip(ids, scores)}

        all_boxes_gdf, all_tiles_extents_gdf = DetectionAggregator.prepare_boxes(tile_ids_to_path=tile_ids_to_path,
                                                                                 tile_ids_to_boxes=tile_ids_to_boxes,
                                                                                 tile_ids_to_scores=tile_ids_to_scores)

        return cls(geojson_output_path=geojson_output_path,
                   boxes_gdf=all_boxes_gdf,
                   tiles_extent_gdf=all_tiles_extents_gdf,
                   score_threshold=score_threshold,
                   nms_threshold=nms_threshold,
                   nms_algorithm=nms_algorithm)

    @staticmethod
    def prepare_boxes(tile_ids_to_path: dict, tile_ids_to_boxes: dict, tile_ids_to_scores: dict):
        images_without_crs = 0
        all_gdfs_boxes = []
        all_gdfs_tiles_extents = []
        for tile_id in tile_ids_to_boxes.keys():
            image_path = tile_ids_to_path[tile_id]

            src = rasterio.open(image_path)

            if not src.crs or not src.transform:
                images_without_crs += 1
                continue

            # Making sure the Tile respects the convention
            TileNameConvention.parse_name(Path(image_path).name)

            gdf_boxes = gpd.GeoDataFrame(geometry=tile_ids_to_boxes[tile_id])
            gdf_boxes['geometry'] = gdf_boxes['geometry'].astype(object).apply(
                lambda geom: apply_affine_transform(geom, src.transform)
            )
            gdf_boxes.crs = src.crs
            gdf_boxes['tile_id'] = tile_id
            gdf_boxes['score'] = tile_ids_to_scores[tile_id]
            all_gdfs_boxes.append(gdf_boxes)

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
            warn(f"Multiple CRS were detected (n={n_crs}), so re-projecting all boxes of all tiles to the"
                 f" same common CRS '{common_crs}', chosen from one of the tiles.")
            all_gdfs_boxes = [gdf.to_crs(common_crs) for gdf in all_gdfs_boxes if gdf.crs != common_crs]
            all_gdfs_tiles_extents = [gdf.to_crs(common_crs) for gdf in all_gdfs_tiles_extents if gdf.crs != common_crs]

        all_boxes_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs_boxes, ignore_index=True))
        all_tiles_extents_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs_tiles_extents, ignore_index=True))

        return all_boxes_gdf, all_tiles_extents_gdf

    def remove_low_score_boxes(self):
        if self.score_threshold:
            if "score" in self.boxes_gdf and self.boxes_gdf["score"].any():
                n_before = len(self.boxes_gdf)
                self.boxes_gdf.drop(self.boxes_gdf[self.boxes_gdf["score"] < self.score_threshold].index, inplace=True)
                n_after = len(self.boxes_gdf)
                print(f"Removed {n_before-n_after} out of {n_before} boxes"
                      f" as they were under the score_threshold={self.score_threshold}.")
            else:
                warn(f"Could not apply score_threshold={self.score_threshold} as scores were not found"
                     f" in the boxes_gdf. If you instanced the Aggregator from a COCO file,"
                     f" please make sure that ALL annotations have a 'score' key or a ['other_attributes']['score']"
                     f" nested keys. If you instanced the Aggregator from a list of boxes for each tile,"
                     f" make sure you also pass the associated list of scores for each tile.")

    @staticmethod
    def calculate_iou_for_geometry(gdf, geometry):
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
    def calculate_centroid_distance(gdf, geometry):
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

    def apply_nms_algorithm(self):
        n_before = len(self.boxes_gdf)
        if self.nms_algorithm == "iou":
            self.apply_iou_nms_algorithm()
        if self.nms_algorithm == "diou":
            self.apply_diou_nms_algorithm()

                                                            # TODO add support for adapative-nms: https://arxiv.org/pdf/1904.03629.pdf

        n_after = len(self.boxes_gdf)
        print(f"Removed {n_before - n_after} out of {n_before} boxes"
              f" by applying the Non-Maximum Suppression-{self.nms_algorithm} algorithm.")

    def apply_iou_nms_algorithm(self):
        gdf = self.boxes_gdf.copy()
        gdf.sort_values(by='score', ascending=False, inplace=True)

        intersect_gdf = gpd.sjoin(gdf, gdf, how='inner', predicate='intersects')
        intersect_gdf = intersect_gdf[intersect_gdf.index != intersect_gdf['index_right']]

        keep_ids = set()
        skip_ids = set()
        progress = tqdm(total=len(gdf), desc="Applying NMS-iou algorithm")
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
                ious = self.calculate_iou_for_geometry(gdf.loc[intersecting_geometries_ids], current.geometry)
                skip_ids.update(list(ious[ious > self.nms_threshold].index))

            keep_ids.add(current_id)
            skip_ids.add(current_id)
            gdf.drop(current_id, inplace=True, errors="ignore")

        progress.close()
        drop_ids = list(skip_ids - keep_ids)
        self.boxes_gdf.drop(drop_ids, inplace=True, errors="ignore")

    def apply_diou_nms_algorithm(self):
        gdf = self.boxes_gdf.copy()
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
                ious = self.calculate_iou_for_geometry(gdf.loc[intersecting_geometries_ids], current.geometry)
                center_dists = self.calculate_centroid_distance(gdf.loc[intersecting_geometries_ids], current.geometry)
                dious = ious - (center_dists / max_distance) ** 2
                skip_ids.update(list(dious[dious > self.nms_threshold].index))

            keep_ids.add(current_id)
            skip_ids.add(current_id)
            gdf.drop(current_id, inplace=True, errors="ignore")

        progress.close()
        drop_ids = list(skip_ids - keep_ids)
        self.boxes_gdf.drop(drop_ids, inplace=True, errors="ignore")

    def save_boxes(self):
        self.boxes_gdf.to_file(str(self.geojson_output_path), driver="GeoJSON")
        print(f"Saved {len(self.boxes_gdf)} boxes in file '{self.geojson_output_path}'.")

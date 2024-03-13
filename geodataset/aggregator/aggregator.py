import json
from pathlib import Path
from typing import List
from warnings import warn

import pandas as pd
import rasterio
from shapely import box
import geopandas as gpd

from geodataset.utils import TileNameConvention, apply_affine_transform


class DetectionAggregator:
    def __init__(self,
                 labels_gdf: gpd.GeoDataFrame):

        self.labels_gdf = labels_gdf
        assert self.labels_gdf.crs, "The provided labels_gdf doesn't have a CRS."
        assert 'tile_id' in self.labels_gdf, "The provided labels_gdf doesn't have a 'tile_id' column."
        print(self.labels_gdf)

    @classmethod
    def from_coco(cls,
                  tiles_folder_path: Path,
                  coco_json_path: Path):

        # Read the JSON file
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        # Reading the boxes
        tile_ids_to_boxes = {}
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']

            xmin = annotation['bbox'][0]
            xmax = annotation['bbox'][0] + annotation['bbox'][2]
            ymin = annotation['bbox'][1]
            ymax = annotation['bbox'][1] + annotation['bbox'][3]
            annotation_box = box(xmin, ymin, xmax, ymax)

            if image_id in tile_ids_to_boxes:
                tile_ids_to_boxes[image_id].append(annotation_box)
            else:
                tile_ids_to_boxes[image_id] = [annotation_box]

        tile_ids_to_path = {}
        for image in coco_data['images']:
            image_path = tiles_folder_path / image['file_name']
            assert image_path.exists(), (f"Could not find the tile '{image['file_name']}'"
                                         f" in tiles_folder_path='{tiles_folder_path}'")
            tile_ids_to_path[image['id']] = image_path

        gdf_all_boxes = DetectionAggregator.prepare_boxes(tile_ids_to_path=tile_ids_to_path,
                                                          tile_ids_to_boxes=tile_ids_to_boxes)

        return cls(labels_gdf=gdf_all_boxes)

    @classmethod
    def from_boxes(cls,
                   tiles_paths: List[Path],
                   boxes: List[List[box]]):

        assert len(tiles_paths) == len(boxes), ("The number of tiles_paths must be equal than the number of lists "
                                                "in boxes (one list of boxes per tile).")

        ids = list(range(len(tiles_paths)))
        tile_ids_to_path = {k: v for k, v in zip(ids, tiles_paths)}
        tile_ids_to_boxes = {k: v for k, v in zip(ids, boxes)}

        gdf_all_boxes = DetectionAggregator.prepare_boxes(tile_ids_to_path=tile_ids_to_path,
                                                          tile_ids_to_boxes=tile_ids_to_boxes)

        return cls(labels_gdf=gdf_all_boxes)

    @staticmethod
    def prepare_boxes(tile_ids_to_path: dict, tile_ids_to_boxes: dict):
        images_without_crs = 0
        all_gdfs_boxes = []
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
            all_gdfs_boxes.append(gdf_boxes)

        if images_without_crs > 0:
            warn(f"The aggregator found {images_without_crs} tiles without a CRS or transform")

        gdfs_crs = [gdf.crs for gdf in all_gdfs_boxes]
        n_crs = len(set(gdfs_crs))
        if n_crs > 1:
            common_crs = gdfs_crs[0].crs
            warn(f"Multiple CRS were detected (n={n_crs}), so re-projecting all boxes of all tiles to the"
                 f" same common CRS '{common_crs}', chosen from one of the tiles.")
            all_gdfs_boxes = [gdf.to_crs(common_crs) for gdf in all_gdfs_boxes if gdf.crs != common_crs]

        gdf_all_boxes = gpd.GeoDataFrame(pd.concat(all_gdfs_boxes, ignore_index=True))

        return gdf_all_boxes


if __name__ == "__main__":
    aggregator = DetectionAggregator.from_coco(tiles_folder_path=Path("/home/hugobaudchon/Documents/Data/pre_processed/all_datasets/quebec_trees/2021_09_02_sbl_z1_rgb_cog/tiles"),
                                               coco_json_path=Path("/home/hugobaudchon/Documents/Data/pre_processed/all_datasets/quebec_trees/2021_09_02_sbl_z1_rgb_cog/2021_09_02_sbl_z1_rgb_cog_coco_gr0p05_train.json"))

    aggregator.labels_gdf.to_file("./test.geojson", driver='GeoJSON')




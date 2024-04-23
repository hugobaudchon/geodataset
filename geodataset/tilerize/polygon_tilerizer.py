from pathlib import Path
from typing import List

import geopandas as gpd

from geodataset.aoi import AOIFromPackageConfig
from geodataset.aoi.aoi_from_package import AOIFromPackageForPolygons
from geodataset.geodata import Raster
from geodataset.geodata.tile import TileSaver
from geodataset.labels import RasterPolygonLabels
from geodataset.utils import CocoNameConvention, COCOGenerator


class PolygonTilerizer:
    def __init__(self,
                 raster_path: Path,
                 labels_path: Path,
                 output_path: Path,
                 tile_size: int,
                 aois_config: AOIFromPackageConfig or None,
                 ground_resolution: float or None,
                 scale_factor: float or None,
                 use_rle_for_labels: bool = True,
                 min_intersection_ratio: float = 0.5,
                 main_label_category_column_name: str = None,
                 other_labels_attributes_column_names: List[str] = None,
                 coco_n_workers: int = 5,
                 coco_categories_list: list[dict] = None):

        self.raster_path = raster_path
        self.labels_path = labels_path
        self.output_path = output_path
        self.tile_size = tile_size
        self.aois_config = aois_config
        self.use_rle_for_labels = use_rle_for_labels
        self.min_intersection_ratio = min_intersection_ratio
        self.coco_n_workers = coco_n_workers
        self.coco_categories_list = coco_categories_list

        assert not (ground_resolution and scale_factor), ("Both a ground_resolution and a scale_factor were provided."
                                                          " Please only specify one.")
        self.scale_factor = scale_factor
        self.ground_resolution = ground_resolution

        self.raster = self._load_raster()
        self.labels = self._load_labels(main_label_category_column_name, other_labels_attributes_column_names)

        self.output_path = output_path / self.raster.product_name
        self.tiles_path = self.output_path / 'tiles'
        self.tiles_path.mkdir(parents=True, exist_ok=True)

    def _load_raster(self):
        raster = Raster(path=self.raster_path,
                        ground_resolution=self.ground_resolution,
                        scale_factor=self.scale_factor)
        return raster

    def _load_labels(self,
                     main_label_category_column_name: str or None,
                     other_labels_attributes_column_names: List[str] or None):

        labels = RasterPolygonLabels(path=self.labels_path,
                                     associated_raster=self.raster,
                                     main_label_category_column_name=main_label_category_column_name,
                                     other_labels_attributes_column_names=other_labels_attributes_column_names)

        return labels

    def _generate_aois_polygons(self):
        aoi_engine = AOIFromPackageForPolygons(
            aois_config=self.aois_config,
            associated_raster=self.raster,
            labels=self.labels,
        )

        aois_polygons, aois_labels = aoi_engine.get_aoi_polygons()

        return aois_polygons, aois_labels

    def _generate_aois_tiles_and_polygons(self):
        aois_polygons, aois_labels = self._generate_aois_polygons()

        final_aois_tiles = {aoi: [] for aoi in aois_polygons.keys()}
        final_aois_polygons = {aoi: [] for aoi in aois_polygons.keys()}
        for aoi, polygons in aois_polygons.items():
            for i, polygon_row in polygons.iterrows():
                polygon = polygon_row['geometry']

                polygon_tile, translated_polygon = self.raster.get_polygon_tile(
                    polygon=polygon,
                    polygon_id=polygon_row['geometry_id'],
                    tile_size=self.tile_size
                )

                final_aois_tiles[aoi].append(polygon_tile)

                gdf = gpd.GeoDataFrame(polygon_row.to_frame().transpose(), geometry='geometry')
                gdf['geometry'] = translated_polygon
                gdf['area'] = translated_polygon.area

                final_aois_polygons[aoi].append(gdf)

        return final_aois_tiles, final_aois_polygons

    def generate_coco_dataset(self):
        aois_tiles, aois_polygons = self._generate_aois_tiles_and_polygons()

        tile_saver = TileSaver(tiles_path=self.tiles_path, n_workers=self.coco_n_workers)

        coco_paths = {}
        for aoi in aois_tiles:
            if aoi == 'all' and len(aois_tiles.keys()) > 1:
                # don't save the 'all' tiles if aois were provided.
                continue

            tiles = aois_tiles[aoi]
            polygons = aois_polygons[aoi]

            if len(tiles) == 0:
                print(f"No tiles found for AOI {aoi}, skipping...")
                continue

            if len(tiles) == 0:
                print(f"No tiles found for AOI {aoi}. Skipping...")
                continue

            tiles_paths = [self.tiles_path / tile.generate_name() for tile in tiles]
            polygons_list = [x['geometry'].to_list() for x in polygons]
            categories_list = [x[self.labels.main_label_category_column_name].to_list() for x in polygons]\
                if self.labels.main_label_category_column_name else None
            other_attributes_dict_list = [{attribute: label[attribute].to_list() for attribute in
                                           self.labels.other_labels_attributes_column_names} for label in polygons]\
                if self.labels.other_labels_attributes_column_names else None
            other_attributes_dict_list = [[{k: d[k][i] for k in d} for i in range(len(next(iter(d.values()))))] for d in other_attributes_dict_list]\
                if self.labels.other_labels_attributes_column_names else None

            # Saving the tiles
            tile_saver.save_all_tiles(tiles)

            coco_output_file_path = self.output_path / CocoNameConvention.create_name(
                product_name=self.raster.product_name,
                ground_resolution=self.ground_resolution,
                scale_factor=self.scale_factor,
                fold=aoi
            )

            print(f"Generating COCO dataset for AOI {aoi}... "
                  f"(it might take a little while to save tiles and encode masks)")
            coco_generator = COCOGenerator(
                description=f"Dataset for the product {self.raster.product_name}"
                            f" with fold {aoi}"
                            f" and scale_factor {self.scale_factor}"
                            f" and ground_resolution {self.ground_resolution}.",
                tiles_paths=tiles_paths,
                polygons=polygons_list,
                scores=None,
                categories=categories_list,
                other_attributes=other_attributes_dict_list,
                output_path=coco_output_file_path,
                use_rle_for_labels=self.use_rle_for_labels,
                n_workers=self.coco_n_workers,
                coco_categories_list=self.coco_categories_list
            )

            coco_generator.generate_coco()
            coco_paths[aoi] = coco_output_file_path

        return coco_paths

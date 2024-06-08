from pathlib import Path
from typing import List, cast

import geopandas as gpd
from shapely import box
from tqdm import tqdm

from geodataset.aoi import AOIFromPackageConfig, AOIGeneratorConfig
from geodataset.aoi.aoi_from_package import AOIFromPackageForPolygons
from geodataset.geodata import Raster
from geodataset.geodata.tile import TileSaver, PolygonTile
from geodataset.labels import RasterPolygonLabels
from geodataset.utils import CocoNameConvention, COCOGenerator


class PolygonTilerizer:
    def __init__(self,
                 raster_path: Path,
                 labels_path: Path,
                 output_path: Path,
                 tile_size: int or None,
                 variable_tile_size_pixel_buffer: int or None,
                 aois_config: AOIFromPackageConfig or None,
                 ground_resolution: float or None,
                 scale_factor: float or None,
                 use_rle_for_labels: bool = True,
                 min_intersection_ratio: float = 0.5,
                 geopackage_layer_name: str = None,
                 main_label_category_column_name: str = None,
                 other_labels_attributes_column_names: List[str] = None,
                 coco_n_workers: int = 5,
                 coco_categories_list: list[dict] = None,
                 tile_batch_size: int = 1000):

        self.raster_path = raster_path
        self.labels_path = labels_path
        self.output_path = output_path
        self.tile_size = tile_size
        self.variable_tile_size_pixel_buffer = variable_tile_size_pixel_buffer
        self.aois_config = aois_config
        self.use_rle_for_labels = use_rle_for_labels
        self.min_intersection_ratio = min_intersection_ratio
        self.coco_n_workers = coco_n_workers
        self.coco_categories_list = coco_categories_list
        self.tile_batch_size = tile_batch_size

        assert not (ground_resolution and scale_factor), ("Both a ground_resolution and a scale_factor were provided."
                                                          " Please only specify one.")

        assert (tile_size or variable_tile_size_pixel_buffer), ("No tile_size or variable_tile_size_pixel_buffer provided."
                                                                " Please specify one.")
        assert not (tile_size and variable_tile_size_pixel_buffer), ("Both a tile_size and a variable_tile_size_pixel_buffer were provided."
                                                                     " Please only specify one.")

        self.scale_factor = scale_factor
        self.ground_resolution = ground_resolution

        self.raster = self._load_raster()
        self.labels = self._load_labels(
            geopackage_layer_name=geopackage_layer_name,
            main_label_category_column_name=main_label_category_column_name,
            other_labels_attributes_column_names=other_labels_attributes_column_names
        )

        self.output_path = output_path / self.raster.product_name
        self.tiles_folder_path = self.output_path / 'tiles'
        self.tiles_folder_path.mkdir(parents=True, exist_ok=True)

    def _load_raster(self):
        raster = Raster(path=self.raster_path,
                        ground_resolution=self.ground_resolution,
                        scale_factor=self.scale_factor)
        return raster

    def _load_labels(self,
                     geopackage_layer_name: str or None,
                     main_label_category_column_name: str or None,
                     other_labels_attributes_column_names: List[str] or None):

        labels = RasterPolygonLabels(path=self.labels_path,
                                     associated_raster=self.raster,
                                     geopackage_layer_name=geopackage_layer_name,
                                     main_label_category_column_name=main_label_category_column_name,
                                     other_labels_attributes_column_names=other_labels_attributes_column_names)

        return labels

    def _generate_aois_polygons(self):
        if self.aois_config is not None:
            if type(self.aois_config) is AOIGeneratorConfig:
                if len(self.aois_config.aois) > 1 or next(iter(self.aois_config.aois.values()))['percentage'] != 1:
                    raise Exception("Currently only supports inference on whole raster.")

                polygons = self.labels.geometries_gdf.copy()
                polygons['geometry_id'] = range(len(polygons))

                aois_polygons = {next(iter(self.aois_config.aois.keys())): polygons}
                # Get the raster extent and create a polygon and gdf for it

                raster_extent = box(0, 0, self.raster.metadata['width'], self.raster.metadata['height'])
                aois_gdf = {next(iter(self.aois_config.aois.keys())): gpd.GeoDataFrame(geometry=[raster_extent])}

            elif type(self.aois_config) is AOIFromPackageConfig:
                aoi_engine = AOIFromPackageForPolygons(labels=self.labels,
                                                       aois_config=cast(AOIFromPackageConfig, self.aois_config),
                                                       associated_raster=self.raster)

                aois_polygons, aois_gdf = aoi_engine.get_aoi_polygons()
            else:
                raise Exception(f'aois_config type unsupported: {type(self.aois_config)}')
        else:
            raise Exception("No AOI configuration provided.")

        return aois_polygons, aois_gdf

    def _generate_aois_tiles_and_polygons(self):
        aois_polygons, aois_labels = self._generate_aois_polygons()

        final_aois_tiles_paths = {aoi: [] for aoi in aois_polygons.keys()}
        final_aois_polygons = {aoi: [] for aoi in aois_polygons.keys()}
        tiles_batch = []
        for aoi, polygons in aois_polygons.items():
            for i, polygon_row in tqdm(polygons.iterrows(),
                                       f"Generating polygon tiles for AOI {aoi}...", total=len(polygons)):

                polygon = polygon_row['geometry']

                if self.tile_size:
                    tile_size = self.tile_size
                elif self.variable_tile_size_pixel_buffer:
                    # get the max height, width and put buffer of pixels around it (*2 to have tile_pixel_buffer on each side)
                    tile_size = max(polygon.bounds[2] - polygon.bounds[0],
                                    polygon.bounds[3] - polygon.bounds[1]) + self.variable_tile_size_pixel_buffer * 2
                else:
                    raise Exception("No tile size provided.")

                polygon_tile, translated_polygon = self.raster.get_polygon_tile(
                    polygon=polygon,
                    polygon_id=polygon_row['geometry_id'],
                    tile_size=tile_size
                )

                tiles_batch.append(polygon_tile)

                gdf = gpd.GeoDataFrame(polygon_row.to_frame().transpose(), geometry='geometry')
                gdf['geometry'] = translated_polygon
                gdf['area'] = translated_polygon.area

                final_aois_polygons[aoi].append(gdf)

                if len(tiles_batch) == self.tile_batch_size:
                    tiles_paths = self._save_tiles_batch(tiles_batch)
                    tiles_batch = []
                    final_aois_tiles_paths[aoi].extend(tiles_paths)

            if len(tiles_batch) > 0:
                tiles_paths = self._save_tiles_batch(tiles_batch)
                tiles_batch = []
                final_aois_tiles_paths[aoi].extend(tiles_paths)

        return final_aois_tiles_paths, final_aois_polygons

    def _save_tiles_batch(self, tiles: List[PolygonTile]):
        tiles_paths = [self.tiles_folder_path / tile.generate_name() for tile in tiles]
        tile_saver = TileSaver(tiles_path=self.tiles_folder_path, n_workers=self.coco_n_workers)
        tile_saver.save_all_tiles(tiles)

        return tiles_paths

    def generate_coco_dataset(self):
        aois_tiles_paths, aois_polygons = self._generate_aois_tiles_and_polygons()

        coco_paths = {}
        for aoi in aois_tiles_paths:
            if aoi == 'all' and len(aois_tiles_paths.keys()) > 1:
                # don't save the 'all' tiles if aois were provided.
                continue

            tiles_paths = aois_tiles_paths[aoi]
            polygons = aois_polygons[aoi]

            if len(tiles_paths) == 0:
                print(f"No tiles found for AOI {aoi}, skipping...")
                continue

            if len(tiles_paths) == 0:
                print(f"No tiles found for AOI {aoi}. Skipping...")
                continue

            polygons_list = [x['geometry'].to_list() for x in polygons]
            categories_list = [x[self.labels.main_label_category_column_name].to_list() for x in polygons]\
                if self.labels.main_label_category_column_name else None
            other_attributes_dict_list = [{attribute: label[attribute].to_list() for attribute in
                                           self.labels.other_labels_attributes_column_names} for label in polygons]\
                if self.labels.other_labels_attributes_column_names else None
            other_attributes_dict_list = [[{k: d[k][i] for k in d} for i in range(len(next(iter(d.values()))))] for d in other_attributes_dict_list]\
                if self.labels.other_labels_attributes_column_names else None

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

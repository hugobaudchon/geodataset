from pathlib import Path
from typing import List, cast

import geopandas as gpd
import pandas as pd
from shapely import box
from tqdm import tqdm

from geodataset.aoi import AOIFromPackageConfig, AOIGeneratorConfig
from geodataset.aoi.aoi_from_package import AOIFromPackageForPolygons
from geodataset.geodata import Raster, RasterPolygonTileMetadata
from geodataset.geodata.raster import RasterTileSaver
from geodataset.labels import RasterPolygonLabels
from geodataset.utils import CocoNameConvention, COCOGenerator
from geodataset.utils.file_name_conventions import AoiGeoPackageConvention


class RasterPolygonTilerizer:
    def __init__(self,
                 raster_path: str or Path,
                 labels_path: str or Path or None,
                 output_path: str or Path,
                 tile_size: int,
                 use_variable_tile_size: bool,
                 variable_tile_size_pixel_buffer: int or None,
                 labels_gdf: gpd.GeoDataFrame = None,
                 global_aoi: str or Path or gpd.GeoDataFrame = None,
                 aois_config: AOIFromPackageConfig = None,
                 ground_resolution: float = None,
                 scale_factor: float = None,
                 output_name_suffix: str = None,
                 use_rle_for_labels: bool = True,
                 min_intersection_ratio: float = 0.5,
                 geopackage_layer_name: str = None,
                 main_label_category_column_name: str = None,
                 other_labels_attributes_column_names: List[str] = None,
                 coco_n_workers: int = 5,
                 coco_categories_list: list[dict] = None,
                 tile_batch_size: int = 1000,
                 temp_dir: str or Path = './tmp'):

        self.raster_path = Path(raster_path)
        self.labels_path = Path(labels_path)
        self.tile_size = tile_size
        self.use_variable_tile_size = use_variable_tile_size
        self.variable_tile_size_pixel_buffer = variable_tile_size_pixel_buffer
        self.global_aoi = global_aoi
        self.aois_config = aois_config
        self.output_name_suffix = output_name_suffix
        self.use_rle_for_labels = use_rle_for_labels
        self.min_intersection_ratio = min_intersection_ratio
        self.coco_n_workers = coco_n_workers
        self.coco_categories_list = coco_categories_list
        self.tile_batch_size = tile_batch_size
        self.temp_dir = Path(temp_dir)

        assert not (ground_resolution and scale_factor), ("Both a ground_resolution and a scale_factor were provided."
                                                          " Please only specify one.")

        if use_variable_tile_size:
            assert variable_tile_size_pixel_buffer, "A variable_tile_size_pixel_buffer must be provided if use_variable_tile_size is True."

        self.scale_factor = scale_factor
        self.ground_resolution = ground_resolution

        self.raster = self._load_raster()
        self.labels = self._load_labels(
            labels_gdf=labels_gdf,
            geopackage_layer_name=geopackage_layer_name,
            main_label_category_column_name=main_label_category_column_name,
            other_labels_attributes_column_names=other_labels_attributes_column_names
        )

        self.output_path = Path(output_path) / self.raster.output_name
        self.tiles_folder_path = self.output_path / 'tiles'
        self.tiles_folder_path.mkdir(parents=True, exist_ok=True)

    def _load_raster(self):
        raster = Raster(path=self.raster_path,
                        output_name_suffix=self.output_name_suffix,
                        ground_resolution=self.ground_resolution,
                        scale_factor=self.scale_factor,
                        temp_dir=self.temp_dir)
        return raster

    def _load_labels(self,
                     labels_gdf: gpd.GeoDataFrame,
                     geopackage_layer_name: str or None,
                     main_label_category_column_name: str or None,
                     other_labels_attributes_column_names: List[str] or None):

        if self.labels_path and labels_gdf:
            raise ValueError("You can't provide both a labels_path and a labels_gdf.")
        elif self.labels_path:
            labels = RasterPolygonLabels(path=self.labels_path,
                                         associated_raster=self.raster,
                                         geopackage_layer_name=geopackage_layer_name,
                                         main_label_category_column_name=main_label_category_column_name,
                                         other_labels_attributes_column_names=other_labels_attributes_column_names)
        elif labels_gdf is not None:
            labels = RasterPolygonLabels(path=None,
                                         labels_gdf=labels_gdf,
                                         associated_raster=self.raster,
                                         main_label_category_column_name=main_label_category_column_name,
                                         other_labels_attributes_column_names=other_labels_attributes_column_names)
        else:
            raise ValueError("You must provide either a labels_path or a labels_gdf.")

        return labels

    def _generate_aois_polygons(self):
        if self.aois_config is not None:
            if type(self.aois_config) is AOIGeneratorConfig:
                if len(self.aois_config.aois) > 1 or next(iter(self.aois_config.aois.values()))['percentage'] != 1 or self.global_aoi is not None:
                    raise Exception("Currently only supports inference on whole raster.")

                polygons = self.labels.geometries_gdf.copy()
                polygons['geometry_id'] = range(len(polygons))

                aois_polygons = {next(iter(self.aois_config.aois.keys())): polygons}
                # Get the raster extent and create a polygon and gdf for it

                raster_extent = box(0, 0, self.raster.metadata['width'], self.raster.metadata['height'])
                aois_gdf = gpd.GeoDataFrame(geometry=[raster_extent])
                aois_gdf['aoi'] = next(iter(self.aois_config.aois.keys()))

            elif type(self.aois_config) is AOIFromPackageConfig:
                aoi_engine = AOIFromPackageForPolygons(labels=self.labels,
                                                       global_aoi=self.global_aoi,
                                                       aois_config=cast(AOIFromPackageConfig, self.aois_config),
                                                       associated_raster=self.raster)

                aois_polygons, aois_gdf = aoi_engine.get_aoi_polygons()
            else:
                raise Exception(f'aois_config type unsupported: {type(self.aois_config)}')
        else:
            raise Exception("No AOI configuration provided.")

        # Saving the AOIs to disk
        for aoi in aois_gdf['aoi'].unique():
            if aoi in aois_polygons and len(aois_polygons[aoi]) > 0:
                aoi_gdf = aois_gdf[aois_gdf['aoi'] == aoi]
                aoi_gdf = self.raster.revert_polygons_pixels_coordinates_to_crs(aoi_gdf)
                aoi_file_name = AoiGeoPackageConvention.create_name(
                    product_name=self.raster.output_name,
                    aoi=aoi,
                    ground_resolution=self.ground_resolution,
                    scale_factor=self.scale_factor
                )
                aoi_gdf.drop(columns=['id'], inplace=True, errors='ignore')
                aoi_gdf.to_file(self.output_path / aoi_file_name, driver='GPKG')
                print(f"Final AOI '{aoi}' saved to {self.output_path / aoi_file_name}.")

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

                polygon_tile, translated_polygon = self.raster.get_polygon_tile(
                    polygon=polygon,
                    polygon_id=polygon_row['geometry_id'],
                    polygon_aoi=aoi,
                    tile_size=self.tile_size,
                    use_variable_tile_size=self.use_variable_tile_size,
                    variable_tile_size_pixel_buffer=self.variable_tile_size_pixel_buffer
                )

                tiles_batch.append(polygon_tile)

                gdf = gpd.GeoDataFrame(polygon_row.to_frame().transpose(), geometry='geometry')
                gdf['geometry'] = translated_polygon
                gdf['area'] = translated_polygon.area

                final_aois_polygons[aoi].append(gdf)

                if len(tiles_batch) == self.tile_batch_size:
                    tiles_paths = self._save_tiles_batch(tiles_batch, aoi=aoi)
                    tiles_batch = []
                    final_aois_tiles_paths[aoi].extend(tiles_paths)

            if len(tiles_batch) > 0:
                tiles_paths = self._save_tiles_batch(tiles_batch, aoi=aoi)
                tiles_batch = []
                final_aois_tiles_paths[aoi].extend(tiles_paths)

        return final_aois_tiles_paths, final_aois_polygons

    def _save_tiles_batch(self, tiles: List[RasterPolygonTileMetadata], aoi: str):
        (self.tiles_folder_path / aoi).mkdir(parents=True, exist_ok=True)
        tiles_paths = [self.tiles_folder_path / aoi / tile.generate_name() for tile in tiles]
        tile_saver = RasterTileSaver(n_workers=self.coco_n_workers)
        tile_saver.save_all_tiles(tiles, output_folder=self.tiles_folder_path / aoi)

        return tiles_paths

    def generate_coco_dataset(self):
        aois_tiles_paths, aois_polygons = self._generate_aois_tiles_and_polygons()

        coco_paths = {}
        for aoi in aois_tiles_paths:
            if aoi == 'all' and len(aois_tiles_paths.keys()) > 1:
                # don't save the 'all' tiles if aois were provided.
                continue

            tiles_paths = aois_tiles_paths[aoi]
            polygons_gdfs = aois_polygons[aoi]

            if len(tiles_paths) == 0:
                print(f"No tiles found for AOI {aoi}, skipping...")
                continue

            # Combine the list of GeoDataFrames into one GeoDataFrame,
            # adding a 'tile_path' column from tiles_paths.
            combined_gdf_list = []
            for tile_path, gdf_tile in zip(tiles_paths, polygons_gdfs):
                temp_gdf = gdf_tile.copy()
                temp_gdf['tile_path'] = str(tile_path)
                combined_gdf_list.append(temp_gdf)
            combined_gdf = gpd.GeoDataFrame(pd.concat(combined_gdf_list, ignore_index=True))

            coco_output_file_path = self.output_path / CocoNameConvention.create_name(
                product_name=self.raster.output_name,
                ground_resolution=self.ground_resolution,
                scale_factor=self.scale_factor,
                fold=aoi
            )

            print(f"Generating COCO dataset for AOI {aoi}... "
                  f"(it might take a little while to save tiles and encode masks)")
            coco_generator = COCOGenerator.from_gdf(
                description=f"Dataset for the product {self.raster.output_name}"
                            f" with fold {aoi}"
                            f" and scale_factor {self.scale_factor}"
                            f" and ground_resolution {self.ground_resolution}.",
                gdf=combined_gdf,
                tiles_paths_column='tile_path',
                polygons_column='geometry',
                scores_column=None,
                categories_column=self.labels.main_label_category_column_name if self.labels.main_label_category_column_name else None,
                other_attributes_columns=self.labels.other_labels_attributes_column_names if self.labels.other_labels_attributes_column_names else None,
                output_path=coco_output_file_path,
                use_rle_for_labels=self.use_rle_for_labels,
                n_workers=self.coco_n_workers,
                coco_categories_list=self.coco_categories_list
            )

            coco_generator.generate_coco()
            coco_paths[aoi] = coco_output_file_path

        return coco_paths


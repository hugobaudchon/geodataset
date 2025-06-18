from pathlib import Path
from typing import List, cast, Optional

import geopandas as gpd
import pandas as pd
from shapely import box
from tqdm import tqdm

from geodataset.aoi import AOIFromPackageConfig, AOIGeneratorConfig, AOIConfig
from geodataset.aoi.aoi_base import DEFAULT_AOI_NAME
from geodataset.aoi.aoi_from_package import AOIFromPackageForPolygons
from geodataset.geodata import Raster, RasterTileMetadata
from geodataset.geodata.raster import RasterTileSaver
from geodataset.labels import RasterPolygonLabels
from geodataset.utils import CocoNameConvention, COCOGenerator
from geodataset.utils.file_name_conventions import AoiGeoPackageConvention


class RasterPolygonTilerizer:
    """
    This class is designed to create individual image tiles for each polygon in a GeoPackages/GeoJson/GeoDataFrame associated with a raster.

    Parameters
    ----------
    raster_path : str or pathlib.Path
        Path to the raster (.tif, .png...).
    labels_path : str or pathlib.Path or None
        Path to the labels. Supported formats are: .gpkg, .geojson, .shp, .xml, .csv.
    output_path : str or pathlib.Path
        Path to parent folder where to save the image tiles and associated labels.
    tile_size : int
        If use_variable_tile_size is set to True, then this parameter defines the maximum size of the tiles in pixels (tile_size, tile_size).
        If use_variable_tile_size is set to False, all polygon tiles will have the same size (tile_size, tile_size).
    use_variable_tile_size: bool
        Whether to use variable tile size. If True, the tile size will match the size of the polygon,
         with a buffer defined by variable_tile_size_pixel_buffer.
    variable_tile_size_pixel_buffer: int or None
        If use_variable_tile_size is True, this parameter defines the pixel buffer to add around the polygon when creating the tile.
    labels_gdf: geopandas.GeoDataFrame, optional
        A GeoDataFrame containing the labels. If provided, labels_path must be None.
    global_aoi : str or pathlib.Path or geopandas.GeoDataFrame, optional
        Path to the global AOI file, or directly a GeoDataFrame.
        If provided, only the tiles intersecting this AOI will be kept, even if some tiles are inside one of the aois
        in aois_config (if AOIFromPackageConfig).

        This parameter can be really useful to create a kfold dataset in association with an AOIGeneratorConfig config like this:

        aois_config = AOIGeneratorConfig(aois={
                'zone1': {'percentage': 0.2, 'position': 1, 'actual_name': f'train{kfold_id}'},
                'zone2': {'percentage': 0.2, 'position': 2, 'actual_name': f'train{kfold_id}'},
                'zone3': {'percentage': 0.2, 'position': 3, 'actual_name': f'valid{kfold_id}'},
                'zone4': {'percentage': 0.2, 'position': 4, 'actual_name': f'train{kfold_id}'},
                'zone5': {'percentage': 0.2, 'position': 5, 'actual_name': f'train{kfold_id}'}
            },
            aoi_type='band'
        )
    aois_config : :class:`~geodataset.aoi.AOIGeneratorConfig` or :class:`~geodataset.aoi.AOIFromPackageConfig` or None
        An instance of AOIConfig to use, or None if all tiles should be kept in a DEFAULT_AOI_NAME AOI.
    ground_resolution : float, optional
        The ground resolution in meter per pixel desired when loading the raster.
        Only one of ground_resolution and scale_factor can be set at the same time.
    scale_factor : float, optional
        Scale factor for rescaling the data (change pixel resolution).
        Only one of ground_resolution and scale_factor can be set at the same time.
    output_name_suffix : str, optional
        Suffix to add to the output file names.
    use_rle_for_labels : bool, optional
        Whether to use RLE encoding for the labels. If False, the labels will be saved as polygons.
    min_intersection_ratio : float, optional
        When finding the associated polygon labels to a tile, this ratio will specify the minimal required intersection
        ratio (intersecting_polygon_area / polygon_area) between a candidate polygon and the tile in order to keep this
        polygon as a label for that tile.
    geopackage_layer_name : str, optional
        The name of the layer in the geopackage file to use as labels. Only used if the labels_path is a .gpkg, .geojson
        or .shp file. Only useful when the labels geopackage file contains multiple layers.
    main_label_category_column_name : str, optional
        The name of the column in the labels file that contains the main category of the labels.
    other_labels_attributes_column_names : list of str, optional
        The names of the columns in the labels file that contains other attributes of the labels, which should be kept
        as a dictionary in the COCO annotations data.
    coco_n_workers : int, optional
        Number of workers to use when generating the COCO dataset.
        Useful when use_rle_for_labels=True as it is quite slow.
    coco_categories_list : list of dict, optional
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
    tile_batch_size : int, optional
        The number of polygon tiles to process in a single batch when saving them to disk.
    temp_dir : str or pathlib.Path
        Temporary directory to store the resampled Raster, if it is too big to fit in memory.
    """

    unique_polygon_id_column_name = 'polygon_id_geodataset'

    def __init__(self,
                 raster_path: str or Path,
                 labels_path: str or Path or None,
                 output_path: str or Path,
                 tile_size: int,
                 use_variable_tile_size: bool,
                 variable_tile_size_pixel_buffer: int or None,
                 labels_gdf: gpd.GeoDataFrame = None,
                 global_aoi: str or Path or gpd.GeoDataFrame = None,
                 aois_config: Optional[AOIConfig] = None,
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
        self.labels_path = Path(labels_path) if labels_path is not None else None
        self.tile_size = tile_size
        self.use_variable_tile_size = use_variable_tile_size
        self.variable_tile_size_pixel_buffer = variable_tile_size_pixel_buffer
        self.global_aoi = global_aoi
        self.aois_config = aois_config
        self.scale_factor = scale_factor
        self.ground_resolution = ground_resolution
        self.output_name_suffix = output_name_suffix
        self.use_rle_for_labels = use_rle_for_labels
        self.min_intersection_ratio = min_intersection_ratio
        self.coco_n_workers = coco_n_workers
        self.coco_categories_list = coco_categories_list
        self.tile_batch_size = tile_batch_size
        self.temp_dir = Path(temp_dir)

        self._check_parameters()
        self._check_aois_config()

        self.raster = self._load_raster()
        self.labels = self._load_labels(
            labels_gdf=labels_gdf,
            geopackage_layer_name=geopackage_layer_name,
            main_label_category_column_name=main_label_category_column_name,
            other_labels_attributes_column_names=other_labels_attributes_column_names
        )

        self.labels.geometries_gdf[self.unique_polygon_id_column_name] = range(1, len(self.labels.geometries_gdf) + 1)

        self.output_path = Path(output_path) / self.raster.output_name
        self.tiles_folder_path = self.output_path / 'tiles'
        self.tiles_folder_path.mkdir(parents=True, exist_ok=True)

    def _check_parameters(self):
        assert self.raster_path.exists(), \
            f"Raster file not found at {self.raster_path}."
        assert isinstance(self.tile_size, int) and self.tile_size > 0, \
            "The tile size must be and integer greater than 0."
        assert not (self.ground_resolution and self.scale_factor), \
            "Both a ground_resolution and a scale_factor were provided. Please only specify one."
        if self.use_variable_tile_size:
            assert self.variable_tile_size_pixel_buffer, "A variable_tile_size_pixel_buffer must be provided if use_variable_tile_size is True."

    def _check_aois_config(self):
        if self.aois_config is None:
            print("RasterPolygonTilerizer: No AOIs configuration provided."
                  f"Defaulting to a single '{DEFAULT_AOI_NAME}' AOI for all polygons.")
            self.aois_config = AOIGeneratorConfig(aois={DEFAULT_AOI_NAME: {'percentage': 1.0, 'position': 1}}, aoi_type='band')
        elif isinstance(self.aois_config, AOIGeneratorConfig) and not self.aois_config.aois: # Empty AOIGeneratorConfig
            print("RasterPolygonTilerizer: Empty AOIGeneratorConfig.aois."
                  f"Defaulting to a single '{DEFAULT_AOI_NAME}' AOI.")
            self.aois_config = AOIGeneratorConfig(aois={DEFAULT_AOI_NAME: {'percentage': 1.0, 'position': 1}}, aoi_type='band')
        else:
            self.aois_config = self.aois_config

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
            labels = RasterPolygonLabels(
                path=self.labels_path,
                associated_raster=self.raster,
                geopackage_layer_name=geopackage_layer_name,
                main_label_category_column_name=main_label_category_column_name,
                other_labels_attributes_column_names=other_labels_attributes_column_names
            )
        elif labels_gdf is not None:
            labels = RasterPolygonLabels(
                path=None,
                labels_gdf=labels_gdf,
                associated_raster=self.raster,
                main_label_category_column_name=main_label_category_column_name,
                other_labels_attributes_column_names=other_labels_attributes_column_names
            )
        else:
            raise ValueError("RasterPolygonTilerizer: You must"
                             "provide either a labels_path or a labels_gdf.")
        return labels

    def _generate_aois_polygons(self):
        polygons_for_aoi_processing = self.labels.geometries_gdf.copy()

        if isinstance(self.aois_config, AOIGeneratorConfig):
            is_simple_generate_all = (
                len(self.aois_config.aois) == 1 and
                next(iter(self.aois_config.aois.keys())).lower() in [DEFAULT_AOI_NAME.lower(), "infer"]
            )
            if is_simple_generate_all:
                aoi_name = next(iter(self.aois_config.aois.keys())) # Get the actual name ('train', 'infer', etc.)
                aois_polygons = {aoi_name: polygons_for_aoi_processing}
                raster_extent_poly = box(0, 0, self.raster.metadata['width'], self.raster.metadata['height'])
                aois_gdf = gpd.GeoDataFrame(
                    {'geometry': [raster_extent_poly], 'aoi': [aoi_name]},
                    crs=self.raster.metadata['crs']
                )
            else:
                # This case remains for truly complex AOIGeneratorConfigs not meant for polygon tiling
                raise NotImplementedError(
                    "Complex AOIGeneratorConfig (defining multiple spatial splits not named 'all'/'infer') "
                    "is not directly supported for RasterPolygonTilerizer when the goal is to tile existing polygons. "
                    "Use AOIFromPackageConfig to assign polygons to predefined vector AOIs, "
                    "or ensure AOIGeneratorConfig defines a single AOI named 'all'/'infer'/something of your choosing."
                )
        elif isinstance(self.aois_config, AOIFromPackageConfig):
            aoi_engine = AOIFromPackageForPolygons(
                labels=self.labels,
                global_aoi=self.global_aoi,
                aois_config=self.aois_config,
                associated_raster=self.raster
            )
            aois_polygons, aois_gdf = aoi_engine.get_aoi_polygons()
        else:
            # This should not be reached if __init__ correctly sets self.aois_config
            raise Exception(f'Internal error: aois_config type unsupported or None: {type(self.aois_config)}')    

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
            else:
                print(f"Skipping save of AOI boundary for '{aoi}'"
                      "as no polygons were assigned to it.")

        return aois_polygons, aois_gdf

    def _generate_aois_tiles_and_polygons(self):
        aois_polygons, _ = self._generate_aois_polygons()

        final_aois_tiles_paths = {aoi: [] for aoi in aois_polygons.keys()}
        final_aois_polygons = {aoi: [] for aoi in aois_polygons.keys()}
        tiles_batch = []

        for aoi, polygons in aois_polygons.items():
            for _, polygon_row in tqdm(polygons.iterrows(),
                                       f"Generating polygon tiles for AOI {aoi}...",
                                       total=len(polygons)):

                polygon = polygon_row['geometry']
                polygon_id = polygon_row[self.unique_polygon_id_column_name]

                polygon_tile, translated_polygon = self.raster.get_polygon_tile(
                    polygon=polygon,
                    polygon_id=polygon_id,
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

    def _save_tiles_batch(self, tiles: List[RasterTileMetadata], aoi: str):
        (self.tiles_folder_path / aoi).mkdir(parents=True, exist_ok=True)
        tiles_paths = [self.tiles_folder_path / aoi / tile.generate_name() for tile in tiles]
        tile_saver = RasterTileSaver(n_workers=self.coco_n_workers)
        tile_saver.save_all_tiles(
            tiles,
            output_folder=self.tiles_folder_path / aoi,
            apply_mask=False,  # We don't apply mask for polygon tiles and let user handle it if needed when loading it for training/inference
        )
        return tiles_paths

    def generate_coco_dataset(self):
        aois_tiles_paths, aois_polygons = self._generate_aois_tiles_and_polygons()
        coco_paths = {}

        for aoi in aois_tiles_paths:
            tiles_paths = aois_tiles_paths[aoi]
            polygons_gdfs = aois_polygons[aoi]

            if len(tiles_paths) == 0:
                print(f"No tiles found for AOI {aoi}, skipping...")
                continue

            if len(tiles_paths) != len(polygons_gdfs):
                raise ValueError(f"Mismatch in length of tile paths ({len(tiles_paths)}) "
                                 f"and polygon GDFs ({len(polygons_gdfs)}) for AOI {aoi}.")

            # Combine the list of GeoDataFrames into one GeoDataFrame,
            # adding a 'tile_path' column from tiles_paths.
            combined_gdf_list = []
            for tile_path, gdf_tile in zip(tiles_paths, polygons_gdfs):
                temp_gdf = gdf_tile.copy()
                temp_gdf['tile_path'] = str(tile_path)
                combined_gdf_list.append(temp_gdf)
            if not combined_gdf_list:
                print(f"No polygons to include in COCO for AOI {aoi}. Skipping.")
                continue
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

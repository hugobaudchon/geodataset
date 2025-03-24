from functools import partial
from typing import List
import geopandas as gpd
from pathlib import Path

import pandas as pd
from shapely import box, Polygon
from shapely.affinity import translate

from geodataset.aoi import AOIConfig
from geodataset.geodata import RasterTileMetadata
from geodataset.labels.raster_labels import RasterPolygonLabels

from geodataset.utils import save_aois_tiles_picture, CocoNameConvention, AoiTilesImageConvention, COCOGenerator
from geodataset.tilerize.raster_tilerizer import BaseDiskRasterTilerizer


class LabeledRasterTilerizer(BaseDiskRasterTilerizer):
    """
    This class is used to create image tiles from a raster and their associated labels from a .geojson, .gpkg or .csv file.
    COCO json files are generated for each AOI (or for the 'all' AOI).

    Parameters
    ----------
    raster_path : str or pathlib.Path
        Path to the raster (.tif, .png...).
    labels_path : str or pathlib.Path or None
        Path to the labels. Supported formats are: .gpkg, .geojson, .shp, .xml, .csv.
    output_path : str or pathlib.Path
        Path to parent folder where to save the image tiles and associated labels.
    tile_size : int
        The size of the tiles in pixels (tile_size, tile_size).
    tile_overlap : float
        The overlap between the tiles (0 <= overlap < 1).
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
        An instance of AOIConfig to use, or None if all tiles should be kept in an 'all' AOI.
    ground_resolution : float, optional
        The ground resolution in meter per pixel desired when loading the raster.
        Only one of ground_resolution and scale_factor can be set at the same time.
    scale_factor : float, optional
        Scale factor for rescaling the data (change pixel resolution).
        Only one of ground_resolution and scale_factor can be set at the same time.
    output_name_suffix : str, optional
        Suffix to add to the output file names.
    ignore_black_white_alpha_tiles_threshold : float, optional
        Threshold ratio of black, white or transparent pixels in a tile to skip it. Default is 0.8.
    use_rle_for_labels : bool, optional
        Whether to use RLE encoding for the labels. If False, the labels will be saved as polygons.
    min_intersection_ratio : float, optional
        When finding the associated polygon labels to a tile, this ratio will specify the minimal required intersection
        ratio (intersecting_polygon_area / polygon_area) between a candidate polygon and the tile in order to keep this
        polygon as a label for that tile.
    ignore_tiles_without_labels : bool, optional
        Whether to ignore (skip) tiles that don't have any associated labels.
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

    temp_dir : str or pathlib.Path
        Temporary directory to store the resampled Raster, if it is too big to fit in memory.

    """

    def __init__(self,
                 raster_path: str or Path,
                 labels_path: str or Path or None,
                 output_path: str or Path,
                 tile_size: int,
                 tile_overlap: float,
                 labels_gdf: gpd.GeoDataFrame = None,
                 global_aoi: str or Path or gpd.GeoDataFrame = None,
                 aois_config: AOIConfig = None,
                 ground_resolution: float = None,
                 scale_factor: float = None,
                 output_name_suffix: str = None,
                 ignore_black_white_alpha_tiles_threshold: float = 0.8,
                 use_rle_for_labels: bool = True,
                 min_intersection_ratio: float = 0.9,
                 ignore_tiles_without_labels: bool = False,
                 geopackage_layer_name: str = None,
                 main_label_category_column_name: str = None,
                 other_labels_attributes_column_names: List[str] = None,
                 coco_n_workers: int = 5,
                 coco_categories_list: list[dict] = None,
                 temp_dir: str or Path = './tmp'):

        super().__init__(
            raster_path=raster_path,
            output_path=output_path,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            global_aoi=global_aoi,
            aois_config=aois_config,
            ground_resolution=ground_resolution,
            scale_factor=scale_factor,
            output_name_suffix=output_name_suffix,
            ignore_black_white_alpha_tiles_threshold=ignore_black_white_alpha_tiles_threshold,
            temp_dir=temp_dir
        )

        self.labels_path = Path(labels_path) if labels_path else None
        self.use_rle_for_labels = use_rle_for_labels
        self.min_intersection_ratio = min_intersection_ratio
        self.ignore_tiles_without_labels = ignore_tiles_without_labels
        self.coco_n_workers = coco_n_workers
        self.coco_categories_list = coco_categories_list

        self.labels = self._load_labels(
            labels_gdf=labels_gdf,
            geopackage_layer_name=geopackage_layer_name,
            main_label_category_column_name=main_label_category_column_name,
            other_labels_attributes_column_names=other_labels_attributes_column_names
        )

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
            raise ValueError("You must provide either a labels_path or a labels_gdf.")

        return labels

    def _generate_aois_tiles(self):
        """
        Get the tiles for each AOI (or for the 'all' AOI) and the associated labels.
        """
        tiles = self._create_tiles()

        (intersecting_labels_raster_coords,
         intersecting_labels_tiles_coords) = self._find_associated_labels(tiles=tiles)

        # Keeping only the interesting tiles and creating a mapping from tile_ids to their labels
        labeled_tiles = []
        for tile in tiles:
            associated_labels = intersecting_labels_tiles_coords[
                intersecting_labels_tiles_coords['tile_id'] == tile.tile_id]
            if self.ignore_tiles_without_labels and len(associated_labels) == 0:
                continue
            else:
                labeled_tiles.append(tile)

        self._get_tiles_per_aoi(tiles=labeled_tiles)

    def _get_tiles_and_labels_per_aoi(self):
        assert self.aois_tiles is not None, "You must call the _get_aois_tiles method first."

        # As some tiles may have been duplicated/removed, we have to re-compute the associated labels
        # First, getting a list of tiles from every aois, except 'all' if there are multiple aois,
        # as 'all' would be a duplication of the other ones.
        tiles = [
            tile for key, tiles in self.aois_tiles.items()
            if not (key == 'all' and len(self.aois_tiles) > 1)
            for tile in tiles
        ]

        (intersecting_labels_raster_coords,
         intersecting_labels_tiles_coords) = self._find_associated_labels(tiles=tiles)

        # Intersect the labels with the aois, to make sure for a given tile, we only get labels inside its assigned AOI.
        intersecting_labels_raster_coords['label_id'] = intersecting_labels_raster_coords.index
        intersecting_labels_raster_coords['label_area'] = intersecting_labels_raster_coords.geometry.area
        intersecting_labels_aois_raster_coords = gpd.overlay(intersecting_labels_raster_coords, self.aois_gdf, how='intersection', keep_geom_type=False)
        intersecting_labels_aois_raster_coords['intersection_area'] = intersecting_labels_aois_raster_coords.geometry.area
        intersecting_labels_aois_raster_coords['intersection_ratio'] = intersecting_labels_aois_raster_coords['intersection_area'] / intersecting_labels_aois_raster_coords['label_area']
        intersecting_labels_aois_raster_coords = intersecting_labels_aois_raster_coords[intersecting_labels_aois_raster_coords['intersection_ratio'] > self.min_intersection_ratio]

        final_aois_tiles = {aoi: [] for aoi in list(self.aois_tiles.keys()) + ['all']}
        final_aois_labels = {aoi: [] for aoi in list(self.aois_tiles.keys()) + ['all']}
        for aoi in self.aois_tiles:
            for tile in self.aois_tiles[aoi]:
                # Use the AOI-cropped label geometries directly
                labels_crs_coords = intersecting_labels_aois_raster_coords[
                    (intersecting_labels_aois_raster_coords['tile_id'] == tile.tile_id) &
                    (intersecting_labels_aois_raster_coords['aoi'] == aoi)
                    ]
                labels_tiles_coords = labels_crs_coords.copy()

                # Translate from global to tile coordinates so that non-(0,0) offsets are handled.
                labels_tiles_coords['geometry'] = labels_tiles_coords.geometry.apply(
                    lambda geom: translate(geom, xoff=-tile.col, yoff=-tile.row)
                )

                # Removing boxes that have an area of 0.0
                labels_tiles_coords = labels_tiles_coords[labels_tiles_coords.geometry.area > 0.0]

                # Remove boxes where x2 - x1 <= 0.5 or y2 - y1 <= 0.5
                bounds = labels_tiles_coords.geometry.bounds
                labels_tiles_coords = labels_tiles_coords[
                    (bounds['maxx'] - bounds['minx'] > 0.5) &
                    (bounds['maxy'] - bounds['miny'] > 0.5)
                    ]

                if self.ignore_tiles_without_labels and len(labels_tiles_coords) == 0:
                    continue

                final_aois_tiles[aoi].append(tile)
                final_aois_labels[aoi].append(labels_tiles_coords)
                final_aois_tiles['all'].append(tile)
                final_aois_labels['all'].append(labels_tiles_coords)

        return final_aois_tiles, final_aois_labels

    def _find_associated_labels(self, tiles) -> (gpd.GeoDataFrame, gpd.GeoDataFrame):
        print("Finding the labels associated to each tile...")

        tile_ids = [tile.tile_id for tile in tiles]

        tiles_gdf = gpd.GeoDataFrame(data={'tile_id': tile_ids,
                                           'geometry': [box(tile.col,
                                                            tile.row,
                                                            tile.col + tile.metadata['width'],
                                                            tile.row + tile.metadata['height']) for tile in tiles]})
        labels_gdf = self.labels.geometries_gdf

        # Remove the tile_id column if it exists, as it is the result of some previous detection/segmentation runs
        if 'tile_id' in labels_gdf:
            labels_gdf.drop(columns='tile_id', inplace=True)

        labels_gdf['label_area'] = labels_gdf.geometry.area
        inter_polygons = gpd.overlay(tiles_gdf, labels_gdf, how='intersection', keep_geom_type=True)
        inter_polygons['area'] = inter_polygons.geometry.area
        inter_polygons['intersection_ratio'] = inter_polygons['area'] / inter_polygons['label_area']
        significant_polygons_inter = inter_polygons[inter_polygons['intersection_ratio'] > self.min_intersection_ratio]
        significant_polygons_inter.reset_index()

        def adjust_geometry(polygon: Polygon, tile: RasterTileMetadata):
            return translate(polygon, xoff=-tile.col, yoff=-tile.row)

        intersecting_labels_tiles_coords = significant_polygons_inter.copy()
        for tile_id, tile in zip(tile_ids, tiles):
            labels_indices = intersecting_labels_tiles_coords[
                intersecting_labels_tiles_coords['tile_id'] == tile_id].index
            adjusted_labels_tiles_coords = intersecting_labels_tiles_coords.loc[labels_indices, 'geometry'].astype(
                object).apply(
                partial(adjust_geometry, tile=tile))
            intersecting_labels_tiles_coords.loc[labels_indices, 'geometry'] = adjusted_labels_tiles_coords

        intersecting_labels_raster_coords = significant_polygons_inter

        return intersecting_labels_raster_coords, intersecting_labels_tiles_coords

    def _save_aoi_data(self,
                       aoi: str,
                       tiles: List[RasterTileMetadata],
                       tiles_paths_aoi: List[Path],
                       labels: list[gpd.GeoDataFrame],
                       save_tiles_folder: Path or None):
        if aoi == 'all' and len(self.aois_tiles.keys()) > 1:
            # don't save the 'all' tiles if aois were provided.
            return

        if len(tiles) == 0:
            print(f"No tiles found for AOI {aoi}. Skipping...")
            return

        # Saving the tiles
        if save_tiles_folder:
            save_tiles_folder.mkdir(parents=True, exist_ok=True)
            for tile in tiles:
                tile.save(output_folder=save_tiles_folder)

        coco_output_file_path = self.output_path / CocoNameConvention.create_name(
            product_name=self.raster.output_name,
            ground_resolution=self.ground_resolution,
            scale_factor=self.scale_factor,
            fold=aoi
        )

        # Combine the list of GeoDataFrames into one GeoDataFrame,
        # adding a 'tile_path' column from tiles_paths_aoi
        combined_gdf_list = []
        for tile_path, gdf_tile in zip(tiles_paths_aoi, labels):
            temp_gdf = gdf_tile.copy()
            temp_gdf['tile_path'] = str(tile_path)
            combined_gdf_list.append(temp_gdf)
        combined_gdf = gpd.GeoDataFrame(pd.concat(combined_gdf_list, ignore_index=True))

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
        return coco_output_file_path

    def generate_coco_dataset(self):
        """
        Generate the tiles and the COCO dataset(s) for each AOI (or for the 'all' AOI) and save everything to the disk.
        """
        self._generate_aois_tiles()
        aois_tiles, aois_labels = self._get_tiles_and_labels_per_aoi()

        # Updating the final tiles in case generate_additional_coco_dataset is called later
        self.aois_tiles = aois_tiles

        save_aois_tiles_picture(
            aois_tiles=self.aois_tiles,
            save_path=self.output_path / AoiTilesImageConvention.create_name(
                product_name=self.raster.output_name,
                ground_resolution=self.ground_resolution,
                scale_factor=self.scale_factor
            ),
            tile_coordinate_step=self.tile_coordinate_step
        )

        [print(f'No tiles found for AOI {aoi}.') for aoi in self.aois_config.actual_names
         if aoi not in self.aois_tiles or len(self.aois_tiles[aoi]) == 0]

        print('Saving the tiles and COCO json files...')
        coco_paths = {}
        for aoi in self.aois_tiles:
            tiles_folder_aoi = self.tiles_path / aoi
            tiles_paths_aoi = [tiles_folder_aoi / tile.generate_name() for tile in self.aois_tiles[aoi]]

            coco_paths[aoi] = self._save_aoi_data(
                aoi=aoi,
                tiles=self.aois_tiles[aoi],
                tiles_paths_aoi=tiles_paths_aoi,
                labels=aois_labels[aoi],
                save_tiles_folder=tiles_folder_aoi
            )

        return coco_paths

    def generate_additional_coco_dataset(self,
                                         labels_gdf: gpd.GeoDataFrame,
                                         aoi_name_mapping: dict,
                                         geopackage_layer_name: str or None = None,
                                         main_label_category_column_name: str or None = None,
                                         other_labels_attributes_column_names: List[str] or None = None):
        """
        Useful when you want to create a second dataset from another set of labels or predictions,
        while using the exact same tiles as before, without having to generate+save them another time.
        A mapping from original aoi names to new aoi names must be provided to avoid overwriting previous COCO datasets.
        Example: {'groundtruth': 'infer'} could be used if you want to first generate a ground truth COCO dataset
        and associated tiles using the generate_coco_dataset method, and then generate inference COCO for the same
        tiles with this generate_additional_coco_dataset method, in order to run some evaluation script afterward.
        """

        assert self.aois_tiles is not None, "You must call the generate_coco_dataset method first."

        self.ignore_tiles_without_labels = False  # we want to keep all tiles, even if they don't have labels

        # Loading the new labels
        self.labels = self._load_labels(
            labels_gdf=labels_gdf,
            geopackage_layer_name=geopackage_layer_name,
            main_label_category_column_name=main_label_category_column_name,
            other_labels_attributes_column_names=other_labels_attributes_column_names
        )

        _, aois_labels = self._get_tiles_and_labels_per_aoi()

        print('Saving the tiles and COCO json files...')
        coco_paths = {}
        for aoi in aoi_name_mapping.keys():
            if aoi not in self.aois_tiles:
                print(f'No tiles found for AOI {aoi}.')
                continue

            mapped_aoi_name = aoi_name_mapping[aoi]

            tiles_folder_aoi = self.tiles_path / aoi
            tiles_paths_aoi = [tiles_folder_aoi / tile.generate_name() for tile in self.aois_tiles[aoi]]

            coco_paths[mapped_aoi_name] = self._save_aoi_data(
                aoi=mapped_aoi_name,
                tiles=self.aois_tiles[aoi],
                tiles_paths_aoi=tiles_paths_aoi,
                labels=aois_labels[aoi],
                save_tiles_folder=False  # don't save tiles again
            )

        return coco_paths

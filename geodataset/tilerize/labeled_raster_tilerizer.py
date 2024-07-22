from functools import partial
from typing import List
import geopandas as gpd
from pathlib import Path

from shapely import box, Polygon
from shapely.affinity import translate

from geodataset.aoi import AOIConfig
from geodataset.geodata.tile import Tile
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
    labels_path : str or pathlib.Path
        Path to the labels. Supported formats are: .gpkg, .geojson, .shp, .xml, .csv.
    output_path : str or pathlib.Path
        Path to parent folder where to save the image tiles and associated labels.
    tile_size : int
        The size of the tiles in pixels (tile_size, tile_size).
    tile_overlap : float
        The overlap between the tiles (0 <= overlap < 1).
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
        A list of category dictionaries in COCO format. Exemple of 2 categories, one being the parent of the other:
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
                 raster_path: str or Path,
                 labels_path: str or Path,
                 output_path: str or Path,
                 tile_size: int,
                 tile_overlap: float,
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
                 coco_categories_list: list[dict] = None):

        super().__init__(raster_path=raster_path,
                         output_path=output_path,
                         tile_size=tile_size,
                         tile_overlap=tile_overlap,
                         aois_config=aois_config,
                         ground_resolution=ground_resolution,
                         scale_factor=scale_factor,
                         output_name_suffix=output_name_suffix,
                         ignore_black_white_alpha_tiles_threshold=ignore_black_white_alpha_tiles_threshold)

        self.labels_path = Path(labels_path)
        self.use_rle_for_labels = use_rle_for_labels
        self.min_intersection_ratio = min_intersection_ratio
        self.ignore_tiles_without_labels = ignore_tiles_without_labels
        self.coco_n_workers = coco_n_workers
        self.coco_categories_list = coco_categories_list

        self.labels = self._load_labels(
            geopackage_layer_name=geopackage_layer_name,
            main_label_category_column_name=main_label_category_column_name,
            other_labels_attributes_column_names=other_labels_attributes_column_names
        )

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

    def _get_tiles_and_labels_per_aoi(self):
        tiles = self._create_tiles()
        (intersecting_labels_raster_coords,
         intersecting_labels_tiles_coords) = self._find_associated_labels(tiles=tiles)

        # Keeping only the interesting tiles and creating a mapping from tile_ids to their labels
        labeled_tiles = []
        for tile in tiles:
            associated_labels = intersecting_labels_tiles_coords[intersecting_labels_tiles_coords['tile_id'] == tile.tile_id]
            if self.ignore_tiles_without_labels and len(associated_labels) == 0:
                continue
            else:
                labeled_tiles.append(tile)

        # Assigning the tiles to AOIs
        aois_tiles, aois_gdf = self._get_tiles_per_aoi(tiles=labeled_tiles)

        # Intersect the labels with the aois, to make sure for a given tile, we only get labels inside its assigned AOI.
        intersecting_labels_raster_coords['label_id'] = intersecting_labels_raster_coords.index
        intersecting_labels_raster_coords['label_area'] = intersecting_labels_raster_coords.geometry.area
        intersecting_labels_aois_raster_coords = gpd.overlay(intersecting_labels_raster_coords, aois_gdf, how='intersection', keep_geom_type=False)
        intersecting_labels_aois_raster_coords['intersection_area'] = intersecting_labels_aois_raster_coords.geometry.area
        intersecting_labels_aois_raster_coords['intersection_ratio'] = intersecting_labels_aois_raster_coords['intersection_area'] / intersecting_labels_aois_raster_coords['label_area']
        intersecting_labels_aois_raster_coords = intersecting_labels_aois_raster_coords[intersecting_labels_aois_raster_coords['intersection_ratio'] > self.min_intersection_ratio]

        final_aois_tiles = {aoi: [] for aoi in list(aois_tiles.keys()) + ['all']}
        final_aois_labels = {aoi: [] for aoi in list(aois_tiles.keys()) + ['all']}
        for aoi in aois_tiles:
            for tile in aois_tiles[aoi]:
                labels_crs_coords = intersecting_labels_aois_raster_coords[intersecting_labels_aois_raster_coords['tile_id'] == tile.tile_id]
                labels_crs_coords = labels_crs_coords[labels_crs_coords['aoi'] == aoi]
                labels_ids = labels_crs_coords['label_id'].tolist()
                labels_tiles_coords = intersecting_labels_tiles_coords[intersecting_labels_tiles_coords.index.isin(labels_ids)]

                # removing boxes that have an area of 0.0
                labels_tiles_coords = labels_tiles_coords[labels_tiles_coords.geometry.area > 0.0]

                # remove boxes where x2 - x1 <= 0.5 or y2 - y1 <= 0.5, as they are too small and will cause issues when rounding the coordinates (area=0)
                labels_tiles_coords = labels_tiles_coords[(labels_tiles_coords.geometry.bounds['maxx'] - labels_tiles_coords.geometry.bounds['minx']) > 0.5]
                labels_tiles_coords = labels_tiles_coords[(labels_tiles_coords.geometry.bounds['maxy'] - labels_tiles_coords.geometry.bounds['miny']) > 0.5]

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

        def adjust_geometry(polygon: Polygon, tile: Tile):
            return translate(polygon, xoff=-tile.col, yoff=-tile.row)

        intersecting_labels_tiles_coords = significant_polygons_inter.copy()
        for tile_id, tile in zip(tile_ids, tiles):
            labels_indices = intersecting_labels_tiles_coords[intersecting_labels_tiles_coords['tile_id'] == tile_id].index
            adjusted_labels_tiles_coords = intersecting_labels_tiles_coords.loc[labels_indices, 'geometry'].astype(object).apply(
                partial(adjust_geometry, tile=tile))
            intersecting_labels_tiles_coords.loc[labels_indices, 'geometry'] = adjusted_labels_tiles_coords

        intersecting_labels_raster_coords = significant_polygons_inter

        return intersecting_labels_raster_coords, intersecting_labels_tiles_coords

    def generate_coco_dataset(self):
        """
        Generate the tiles and the COCO dataset(s) for each AOI (or for the 'all' AOI) and save everything to the disk.
        """
        aois_tiles, aois_labels = self._get_tiles_and_labels_per_aoi()

        save_aois_tiles_picture(aois_tiles=aois_tiles,
                                save_path=self.output_path / AoiTilesImageConvention.create_name(
                                    product_name=self.raster.output_name,
                                    ground_resolution=self.ground_resolution,
                                    scale_factor=self.scale_factor
                                ),
                                tile_coordinate_step=self.tile_coordinate_step)

        print('Saving the tiles and COCO json files...')
        coco_paths = {}
        for aoi in aois_tiles:
            if aoi == 'all' and len(aois_tiles.keys()) > 1:
                # don't save the 'all' tiles if aois were provided.
                continue

            tiles = aois_tiles[aoi]
            labels = aois_labels[aoi]

            if len(tiles) == 0:
                print(f"No tiles found for AOI {aoi}, skipping...")
                continue

            if len(tiles) == 0:
                print(f"No tiles found for AOI {aoi}. Skipping...")
                continue

            tiles_paths = [self.tiles_path / tile.generate_name() for tile in tiles]
            polygons = [x['geometry'].to_list() for x in labels]
            categories_list = [x[self.labels.main_label_category_column_name].to_list() for x in labels]\
                if self.labels.main_label_category_column_name else None
            other_attributes_dict_list = [{attribute: label[attribute].to_list() for attribute in
                                           self.labels.other_labels_attributes_column_names} for label in labels]\
                if self.labels.other_labels_attributes_column_names else None
            other_attributes_dict_list = [[{k: d[k][i] for k in d} for i in range(len(next(iter(d.values()))))] for d in other_attributes_dict_list]\
                if self.labels.other_labels_attributes_column_names else None

            # Saving the tiles
            for tile in aois_tiles[aoi]:
                tile.save(output_folder=self.tiles_path)

            coco_output_file_path = self.output_path / CocoNameConvention.create_name(
                product_name=self.raster.output_name,
                ground_resolution=self.ground_resolution,
                scale_factor=self.scale_factor,
                fold=aoi
            )

            coco_generator = COCOGenerator(
                description=f"Dataset for the product {self.raster.output_name}"
                            f" with fold {aoi}"
                            f" and scale_factor {self.scale_factor}"
                            f" and ground_resolution {self.ground_resolution}.",
                tiles_paths=tiles_paths,
                polygons=polygons,
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

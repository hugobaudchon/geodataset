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
from geodataset.tilerize import BaseDiskRasterTilerizer


class LabeledRasterTilerizer(BaseDiskRasterTilerizer):

    def __init__(self,
                 raster_path: Path,
                 labels_path: Path,
                 output_path: Path,
                 tile_size: int,
                 tile_overlap: float,
                 aois_config: AOIConfig = None,
                 ground_resolution: float = None,
                 scale_factor: float = None,
                 use_rle_for_labels: bool = True,
                 min_intersection_ratio: float = 0.9,
                 ignore_tiles_without_labels: bool = False,
                 ignore_black_white_alpha_tiles_threshold: float = 0.8,
                 main_label_category_column_name: str = None,
                 other_labels_attributes_column_names: List[str] = None,
                 coco_n_workers: int = 5,
                 main_label_category_to_id_map: dict = None):
        """
        raster_path: Path,
            Path to the raster (.tif, .png...).
        labels_path: Path,
            Path to the labels (.geojson, .gpkg, .csv...).
        output_path: Path,
            Path to parent folder where to save the image tiles and associated labels.
        tile_size: int,
            The wanted size of the tiles (tile_size, tile_size).
        tile_overlap: float,
            The overlap between the tiles (should be 0 <= overlap < 1).
        aois_config: AOIConfig or None,
            An instance of AOIConfig to use, or None if all tiles should be kept in an 'all' AOI.
        ground_resolution: float,
            The ground resolution in meter per pixel desired when loading the raster.
            Only one of ground_resolution and scale_factor can be set at the same time.
        scale_factor: float,
            Scale factor for rescaling the data (change pixel resolution).
            Only one of ground_resolution and scale_factor can be set at the same time.
        use_rle_for_labels: bool,
            Whether to use RLE encoding for the labels. If False, the labels will be saved as polygons.
        intersection_ratio: float,
            When finding the associated labels to a tile, this ratio will specify the minimal required intersection
            ratio between a candidate polygon and the tile in order to keep this polygon as a label for that tile.
        ignore_tiles_without_labels: bool,
            Whether to ignore (skip) tiles that don't have any associated labels.
        ignore_black_white_alpha_tiles_threshold: bool,
            Whether to ignore (skip) mostly black or white (>ignore_black_white_alpha_tiles_threshold%) tiles.
        coco_n_workers: int,
            Number of workers to use when generating the COCO dataset. Useful when use_rle_for_labels=True as it is quite slow.
        main_label_category_to_id_map: dict,
            A mapping from the main label category names to their corresponding ids (will be used when generating the COCO files).
        """
        super().__init__(raster_path=raster_path,
                         output_path=output_path,
                         tile_size=tile_size,
                         tile_overlap=tile_overlap,
                         aois_config=aois_config,
                         ground_resolution=ground_resolution,
                         scale_factor=scale_factor,
                         ignore_black_white_alpha_tiles_threshold=ignore_black_white_alpha_tiles_threshold)

        self.labels_path = labels_path
        self.use_rle_for_labels = use_rle_for_labels
        self.min_intersection_ratio = min_intersection_ratio
        self.ignore_tiles_without_labels = ignore_tiles_without_labels
        self.coco_n_workers = coco_n_workers
        self.main_label_category_to_id_map = main_label_category_to_id_map

        self.labels = self._load_labels(main_label_category_column_name, other_labels_attributes_column_names)

    def _load_labels(self,
                     main_label_category_column_name: str or None,
                     other_labels_attributes_column_names: List[str] or None):

        labels = RasterPolygonLabels(path=self.labels_path,
                                     associated_raster=self.raster,
                                     main_label_category_column_name=main_label_category_column_name,
                                     other_labels_attributes_column_names=other_labels_attributes_column_names)

        return labels

    def _get_tiles_and_labels_per_aoi(self):
        tiles = self._create_tiles()
        intersecting_labels = self._find_associated_labels(tiles=tiles)

        # Keeping only the interesting tiles and creating a mapping from tile_ids to their labels
        labeled_tiles = []
        tile_id_to_labels = {}
        for tile in tiles:
            associated_labels = intersecting_labels[intersecting_labels['tile_id'] == tile.tile_id]
            if self.ignore_tiles_without_labels and len(associated_labels) == 0:
                continue
            else:
                labeled_tiles.append(tile)
                tile_id_to_labels[tile.tile_id] = associated_labels

        # Assigning the tiles to AOIs
        aois_tiles = self._get_tiles_per_aoi(tiles=labeled_tiles)

        # Assign the labels to AOIs tiles using the mapping previously created
        aois_labels = {aoi: [] for aoi in aois_tiles.keys()}
        for aoi in aois_tiles:
            for tile in aois_tiles[aoi]:
                aois_labels[aoi].append(tile_id_to_labels[tile.tile_id])

        return aois_tiles, aois_labels

    def _find_associated_labels(self, tiles) -> gpd.GeoDataFrame:
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

        for tile_id, tile in zip(tile_ids, tiles):
            labels_indices = significant_polygons_inter[significant_polygons_inter['tile_id'] == tile_id].index
            adjusted_geometries = significant_polygons_inter.loc[labels_indices, 'geometry'].astype(object).apply(
                partial(adjust_geometry, tile=tile))
            significant_polygons_inter.loc[labels_indices, 'geometry'] = adjusted_geometries

        return significant_polygons_inter

    def generate_coco_dataset(self):
        aois_tiles, aois_labels = self._get_tiles_and_labels_per_aoi()

        save_aois_tiles_picture(aois_tiles=aois_tiles,
                                save_path=self.output_path / AoiTilesImageConvention.create_name(
                                    product_name=self.product_name,
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

            coco_output_file_path = self.output_path / CocoNameConvention.create_name(product_name=self.product_name,
                                                                                      ground_resolution=self.ground_resolution,
                                                                                      scale_factor=self.scale_factor,
                                                                                      fold=aoi)

            coco_generator = COCOGenerator(
                description=f"Dataset for the product {self.product_name}"
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
                main_label_category_to_id_map=self.main_label_category_to_id_map
            )

            coco_generator.generate_coco()
            coco_paths[aoi] = coco_output_file_path

        return coco_paths

import json
import warnings
from functools import partial
from typing import List
import geopandas as gpd
from pathlib import Path

from shapely import box, Polygon
from shapely.affinity import translate

from geodataset.aoi import AOIConfig
from geodataset.geodata.tile import Tile
from geodataset.labels.raster_labels import RasterPolygonLabels

from datetime import date

from geodataset.utils import polygon_to_coco_coordinates, polygon_to_coco_rle_mask, save_aois_tiles_picture
from geodataset.tilerize import BaseRasterTilerizer


class LabeledRasterTilerizer(BaseRasterTilerizer):
    SUPPORTED_TASKS = ['detection', 'segmentation']

    def __init__(self,
                 dataset_name: str,
                 raster_path: Path,
                 labels_path: Path,
                 output_path: Path,
                 task: str,
                 scale_factor: float = 1.0,
                 use_rle_for_labels: bool = True,
                 min_intersection_ratio: float = 0.9,
                 ignore_tiles_without_labels: bool = False,
                 ignore_black_white_alpha_tiles_threshold: float = 0.8,
                 main_label_category_column_name: str = None,
                 other_labels_attributes_column_names: List[str] = None):
        """
        raster_path: Path,
            Path to the raster (.tif, .png...).
        labels_path: Path,
            Path to the labels (.geojson, .gpkg, .csv...).
        output_path: Path,
            Path to parent folder where to save the image tiles and associated labels.
        scale_factor: float,
            Scale factor for rescaling the data (change pixel resolution).
        intersection_ratio: float,
            When finding the associated labels to a tile, this ratio will specify the minimal required intersection
            ratio between a candidate polygon and the tile in order to keep this polygon as a label for that tile.
        ignore_tiles_without_labels: bool,
            Whether to ignore (skip) tiles that don't have any associated labels.
        ignore_black_white_alpha_tiles_threshold: bool,
            Whether to ignore (skip) mostly black or white (>ignore_black_white_alpha_tiles_threshold%) tiles.
        """
        super().__init__(dataset_name=dataset_name,
                         raster_path=raster_path,
                         output_path=output_path,
                         scale_factor=scale_factor,
                         ignore_black_white_alpha_tiles_threshold=ignore_black_white_alpha_tiles_threshold)

        self.labels_path = labels_path
        self.use_rle_for_labels = use_rle_for_labels
        self.min_intersection_ratio = min_intersection_ratio
        self.ignore_tiles_without_labels = ignore_tiles_without_labels
        self.task = task

        assert task in self.SUPPORTED_TASKS, f'The task \'{task}\' is not in the supported tasks {self.SUPPORTED_TASKS}.'

        self.labels = self._load_labels(main_label_category_column_name, other_labels_attributes_column_names)

    def _load_labels(self,
                     main_label_category_column_name: str or None,
                     other_labels_attributes_column_names: List[str] or None):
        if self.task == 'detection':
            labels = RasterPolygonLabels(path=self.labels_path,
                                         associated_raster=self.raster,
                                         task='detection',
                                         scale_factor=self.scale_factor,
                                         main_label_category_column_name=main_label_category_column_name,
                                         other_labels_attributes_column_names=other_labels_attributes_column_names)
        elif self.task == 'segmentation':
            labels = RasterPolygonLabels(path=self.labels_path,
                                         associated_raster=self.raster,
                                         task='segmentation',
                                         scale_factor=self.scale_factor,
                                         main_label_category_column_name=main_label_category_column_name,
                                         other_labels_attributes_column_names=other_labels_attributes_column_names)
        else:
            raise NotImplementedError('An unsupported \'task\' value was provided.')

        return labels

    def _find_associated_labels(self, tiles) -> gpd.GeoDataFrame:

        # We only get the labels for the 'all' key (all tiles) and then assign them to correct aois,
        # in order to reduce computation cost.

        tile_ids = [tile.tile_id for tile in tiles]

        tiles_gdf = gpd.GeoDataFrame(data={'tile_id': tile_ids,
                                           'geometry': [box(tile.col,
                                                            tile.row,
                                                            tile.col + tile.metadata['width'],
                                                            tile.row + tile.metadata['height']) for tile in tiles]})
        labels_gdf = self.labels.labels_gdf
        labels_gdf['label_area'] = labels_gdf.geometry.area
        inter_polygons = gpd.overlay(tiles_gdf, labels_gdf, how='intersection')
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

    def _generate_coco_categories(self):
        """
        Generate COCO categories from the unique label categories in the dataset.
        """
        if self.labels.main_label_category_column_name:
            unique_categories = set(self.labels.labels_gdf[self.labels.main_label_category_column_name])
            categories = [{'id': i + 1, 'name': category, 'supercategory': ''} for i, category in
                          enumerate(unique_categories)]
            self.category_id_map = {category: i + 1 for i, category in enumerate(unique_categories)}
        else:
            categories = {}
            warnings.warn("The GeoDataFrame containing the labels doesn't contain a category column,"
                          " so labels won't have categories.")
        return categories

    def _generate_coco_images_and_labels_annotations(self,
                                                     tiles: List[Tile],
                                                     intersecting_labels: gpd.GeoDataFrame):
        images_coco = []
        annotations_coco = []
        annotation_id = 1

        for tile in tiles:
            associated_labels = intersecting_labels[intersecting_labels['tile_id'] == tile.tile_id]
            if self.ignore_tiles_without_labels and len(associated_labels) == 0:
                continue

            # Directly generate COCO image data from the Tile object
            images_coco.append(tile.to_coco())

            for _, label in associated_labels.iterrows():
                coco_annotation = self._generate_label_coco(label=label, tile=tile, tile_id=tile.tile_id)
                coco_annotation['id'] = annotation_id
                annotation_id += 1
                annotations_coco.append(coco_annotation)

        return images_coco, annotations_coco

    def _generate_label_coco(self, label: gpd.GeoDataFrame, tile: Tile, tile_id: int) -> dict:
        if self.use_rle_for_labels:
            # Convert the polygon to a COCO RLE mask
            segmentation = polygon_to_coco_rle_mask(polygon=label['geometry'],
                                                    tile_height=tile.metadata['height'],
                                                    tile_width=tile.metadata['width'])
        else:
            # Convert the polygon's exterior coordinates to the format expected by COCO
            segmentation = polygon_to_coco_coordinates(polygon=label['geometry'])

            # Calculate the area of the polygon
        area = label['geometry'].area

        # Get the bounding box in COCO format: [x, y, width, height]
        bbox = list(label['geometry'].bounds)
        bbox_coco_format = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

        # Finding the main category if any
        if self.labels.main_label_category_column_name:
            category_id = self.category_id_map[label[self.labels.main_label_category_column_name]]
        else:
            category_id = None

        # Finding the other attributes if any
        if self.labels.other_labels_attributes_column_names:
            other_attributes_dict = {}
            for attribute in self.labels.other_labels_attributes_column_names:
                other_attributes_dict[attribute] = label[attribute]
        else:
            other_attributes_dict = None

        # Generate COCO annotation data from each associated label
        coco_annotation = {
            "segmentation": [segmentation],  # COCO expects a list of polygons, each a list of coordinates
            "is_rle_format": self.use_rle_for_labels,
            "area": area,
            "iscrowd": 0,  # Assuming this polygon represents a single object (not a crowd)
            "image_id": tile_id,
            "bbox": bbox_coco_format,
            "category_id": category_id,
            "other_attributes": other_attributes_dict
        }

        return coco_annotation

    def generate_coco_dataset(self, tile_size=1024, overlap=0, start_counter_tile=1, aois_config: AOIConfig = None):
        aois_tiles = self._get_tiles_per_aoi(aois_config=aois_config,
                                             tile_size=tile_size,
                                             overlap=overlap,
                                             start_counter_tile=start_counter_tile)

        tile_coordinate_step = int((1 - overlap) * tile_size)
        save_aois_tiles_picture(aois_tiles=aois_tiles,
                                save_path=self.output_path / 'aois_tiles.png',
                                tile_coordinate_step=tile_coordinate_step)

        for aoi in aois_tiles:
            if aoi == 'all' and len(aois_tiles.keys()) > 1:
                # don't save the 'all' tiles if aois were provided.
                continue

            tiles = aois_tiles[aoi]
            aoi_tiles_intersecting_labels = self._find_associated_labels(tiles=tiles)

            categories_coco = self._generate_coco_categories()

            images_coco, annotations_coco = self._generate_coco_images_and_labels_annotations(
                tiles,
                intersecting_labels=aoi_tiles_intersecting_labels
            )

            # Assemble the COCO dataset
            coco_dataset = {
                "info": {
                    "description": f"{self.dataset_name}",
                    "dataset_name": self.dataset_name,
                    "version": "1.0",
                    "year": str(date.today().year),
                    "date_created": str(date.today())
                },
                "licenses": [
                    # add license?
                ],
                "images": images_coco,
                "annotations": annotations_coco,
                "categories": categories_coco
            }

            aoi_tiles_output_folder = self.tiles_path / aoi
            aoi_tiles_output_folder.mkdir(exist_ok=False)
            # Save the tile images
            for tile in aois_tiles[aoi]:
                tile.save(output_folder=aoi_tiles_output_folder)

            coco_output_file_path = self.output_path / f'{self.dataset_name}_{aoi}_coco.json'
            # Save the COCO dataset to a JSON file
            with coco_output_file_path.open('w') as f:
                json.dump(coco_dataset, f, ensure_ascii=False, indent=2)

            print(f"The COCO dataset for AOI '{aoi}' has been saved to {self.output_path}")

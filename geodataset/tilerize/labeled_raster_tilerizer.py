import json
import warnings
from functools import partial
from typing import Tuple, List
import numpy as np
import rasterio
import geopandas as gpd
from pathlib import Path

from shapely import box, Polygon
from shapely.affinity import translate

from geodataset.geodata import Raster
from geodataset.geodata.tile import Tile
from geodataset.labels.raster_labels import RasterPolygonLabels

from datetime import date

from geodataset.utils.geometry import polygon_to_coco_rle_mask, polygon_to_coco_coordinates


class LabeledRasterTilerizer:
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
                 ignore_mostly_black_or_white_tiles: bool = True):
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
        ignore_mostly_black_or_white_tiles: bool,
            Whether to ignore (skip) mostly black or white (>50%) tiles.
        """
        self.dataset_name = dataset_name
        self.raster_path = raster_path
        self.labels_path = labels_path
        self.scale_factor = scale_factor
        self.use_rle_for_labels = use_rle_for_labels
        self.min_intersection_ratio = min_intersection_ratio
        self.ignore_tiles_without_labels = ignore_tiles_without_labels
        self.ignore_mostly_black_or_white_tiles = ignore_mostly_black_or_white_tiles
        self.task = task

        assert task in self.SUPPORTED_TASKS, f'The task \'{task}\' is not in the supported tasks {self.SUPPORTED_TASKS}.'

        self.output_path = output_path
        self.tiles_path = self.output_path / self.dataset_name / 'tiles'
        self.tiles_path.mkdir(parents=True, exist_ok=True)
        self.coco_json_path = self.output_path / self.dataset_name / f'{self.dataset_name}_coco.json'

        (self.raster,
         self.labels) = self._load_data()

    def _load_data(self):
        raster = Raster(path=self.raster_path,
                        scale_factor=self.scale_factor)

        if self.task == 'detection':
            labels = RasterPolygonLabels(path=self.labels_path,
                                         associated_raster=raster,
                                         task='detection',
                                         scale_factor=self.scale_factor)
        elif self.task == 'segmentation':
            labels = RasterPolygonLabels(path=self.labels_path,
                                         associated_raster=raster,
                                         task='segmentation',
                                         scale_factor=self.scale_factor)
        else:
            raise NotImplementedError('An unsupported \'task\' value was provided.')

        return raster, labels

    def _create_tiles(self,
                      tile_size: int,
                      overlap: float,
                      start_counter_tile: int) -> List[Tuple[int, Tile]]:
        width = self.raster.metadata['width']
        height = self.raster.metadata['height']
        print('Raster size: ', (height, width))
        print('Desired tile size: ', tile_size)
        print('Creating tiles and finding their associated labels...')
        samples = []
        for row in range(0, height, int((1 - overlap) * tile_size)):
            print(f'\t Row {row}/{height}')
            for col in range(0, width, int((1 - overlap) * tile_size)):
                window = rasterio.windows.Window(col, row, tile_size, tile_size)
                tile = self.raster.get_tile(window=window,
                                            dataset_name=self.dataset_name)
                if self.ignore_mostly_black_or_white_tiles:
                    # If it's >50% black pixels or white pixels, just continue. No point segmenting it.
                    if np.sum(tile.data == 0) / (tile_size * tile_size * self.raster.metadata['count']) > 0.5:
                        continue
                    if np.sum(tile.data == 255) / (tile_size * tile_size * self.raster.metadata['count']) > 0.5:
                        continue

                samples.append((start_counter_tile, tile))
                start_counter_tile += 1

        return samples

    def _find_associated_labels(self, samples) -> gpd.GeoDataFrame:
        tile_ids = [sample[0] for sample in samples]
        tiles = [sample[1] for sample in samples]
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
        if 'category' in self.labels.labels_gdf:
            unique_categories = set(self.labels.labels_gdf['category'])
            categories = [{'id': i + 1, 'name': category, 'supercategory': ''} for i, category in
                          enumerate(unique_categories)]
            self.category_id_map = {category: i + 1 for i, category in enumerate(unique_categories)}
        else:
            categories = {}
            warnings.warn("The GeoDataFrame containing the labels doesn't contain a category column,"
                          " so labels won't have categories.")
        return categories

    def _generate_coco_images_and_labels_annotations(self,
                                                     samples: List[Tuple[int, Tile]],
                                                     intersecting_labels: gpd.GeoDataFrame):
        images_coco = []
        annotations_coco = []
        annotation_id = 1

        for tile_id, tile in samples:
            associated_labels = intersecting_labels[intersecting_labels['tile_id'] == tile_id]
            if self.ignore_tiles_without_labels and len(associated_labels) == 0:
                continue

            # Directly generate COCO image data from the Tile object
            images_coco.append(tile.to_coco(image_id=tile_id))

            for _, label in associated_labels.iterrows():
                coco_annotation = self._generate_label_coco(label=label, tile=tile, tile_id=tile_id)
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

        # Generate COCO annotation data from each associated label
        coco_annotation = {
            "segmentation": [segmentation],  # COCO expects a list of polygons, each a list of coordinates
            "is_rle_format": self.use_rle_for_labels,
            "area": area,
            "iscrowd": 0,  # Assuming this polygon represents a single object (not a crowd)
            "image_id": tile_id,
            "bbox": bbox_coco_format,
            "category_id": self.category_id_map[label['category']] if 'category' in label else None,
        }

        return coco_annotation

    def generate_coco_dataset(self, tile_size=1024, overlap=0, start_counter_tile=1):
        samples = self._create_tiles(tile_size=tile_size, overlap=overlap, start_counter_tile=start_counter_tile)
        intersecting_labels = self._find_associated_labels(samples=samples)

        categories_coco = self._generate_coco_categories()
        (images_coco,
         annotations_coco) = self._generate_coco_images_and_labels_annotations(samples,
                                                                               intersecting_labels=intersecting_labels)

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

        # Save the tile images
        for tile_id, tile in samples:
            tile.save(output_folder=self.tiles_path)

        # Save the COCO dataset to a JSON file
        with self.coco_json_path.open('w') as f:
            json.dump(coco_dataset, f, ensure_ascii=False, indent=2)

        print(f"COCO dataset has been saved to {self.coco_json_path}")

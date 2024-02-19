import json
from typing import Tuple, List
import numpy as np
import rasterio
from pathlib import Path

from geodataset.geodata import Raster
from geodataset.geodata.label import PolygonLabel
from geodataset.geodata.tile import Tile
from geodataset.labels import RasterDetectionLabels, RasterSegmentationLabels

from datetime import date


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
        self.tiles_path = self.output_path / 'tiles'
        self.tiles_path.mkdir(parents=True, exist_ok=True)

        (self.raster,
         self.labels) = self._load_data()

    def _load_data(self):
        raster = Raster(path=self.raster_path,
                        scale_factor=self.scale_factor)

        if self.task == 'detection':
            labels = RasterDetectionLabels(path=self.labels_path,
                                           associated_raster=raster,
                                           scale_factor=self.scale_factor)
        elif self.task == 'segmentation':
            labels = RasterSegmentationLabels(path=self.labels_path,
                                              associated_raster=raster,
                                              scale_factor=self.scale_factor)
        else:
            raise NotImplementedError('An unsupported \'task\' value was provided.')

        return raster, labels

    def create_tiles(self, tile_size=1024, overlap=0) -> List[Tuple[Tile, List[PolygonLabel]]]:
        width = self.raster.metadata['width']
        height = self.raster.metadata['height']
        print('Raster size: ', (width, height))
        print('Desired tile size: ', tile_size)
        print('Saving tiles')
        samples = []
        for row in range(0, width, int((1 - overlap) * tile_size)):
            print(f'\t Row {row}/{width}')
            for col in range(0, height, int((1 - overlap) * tile_size)):
                window = rasterio.windows.Window(col, row, tile_size, tile_size)
                tile = self.raster.get_tile(window=window,
                                            dataset_name=self.dataset_name)
                if self.ignore_mostly_black_or_white_tiles:
                    # If it's >50% black pixels or white pixels, just continue. No point segmenting it.
                    if np.sum(tile.data == 0) / (tile_size * tile_size * self.raster.metadata['count']) > 0.5:
                        continue
                    if np.sum(tile.data == 255) / (tile_size * tile_size * self.raster.metadata['count']) > 0.5:
                        continue

                associated_labels = self.labels.find_associated_labels(window=window,
                                                                       min_intersection_ratio=self.min_intersection_ratio)

                if self.ignore_tiles_without_labels and not associated_labels:
                    continue

                samples.append((tile, associated_labels))

        return samples

    def _generate_coco_categories(self):
        """
        Generate COCO categories from the unique label categories in the dataset.
        """
        unique_categories = set(label.category for label in self.labels.labels)  # Assuming self.labels.labels is a list of PolygonLabel objects
        categories = [{'id': i + 1, 'name': category, 'supercategory': ''} for i, category in enumerate(unique_categories)]
        self.category_id_map = {category: i + 1 for i, category in enumerate(unique_categories)}  # For mapping category names to IDs
        return categories

    def _generate_coco_images_annotations(self, tiles_info, start_image_id=1):
        images_coco = []
        annotations_coco = []
        annotation_id = 1

        for image_id, (tile, associated_labels) in enumerate(tiles_info, start=start_image_id):
            # Directly generate COCO image data from the Tile object
            images_coco.append(tile.to_coco(image_id=image_id))

            for label in associated_labels:
                # Generate COCO annotation data from each associated label
                coco_annotation = label.to_coco(image_id=image_id,
                                                category_id_map=self.category_id_map,
                                                use_rle=self.use_rle_for_labels,
                                                associated_image_size=(tile.data.shape[1], tile.data.shape[2]))
                coco_annotation['id'] = annotation_id
                annotations_coco.append(coco_annotation)
                annotation_id += 1

        return images_coco, annotations_coco

    def generate_coco_dataset(self, tile_size=1024, overlap=0, start_counter_tile=0):
        samples = self.create_tiles(tile_size=tile_size, overlap=overlap)

        categories_coco = self._generate_coco_categories()
        images_coco, annotations_coco = self._generate_coco_images_annotations(samples, start_image_id=start_counter_tile)

        # Assemble the COCO dataset
        coco_dataset = {
            "info": {
                "description": f"{self.dataset_name} Dataset",
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
        for tile, labels in samples:
            tile.save(output_folder=self.tiles_path)

        # Save the COCO dataset to a JSON file
        coco_json_path = self.output_path / f'{self.dataset_name}_coco.json'
        with coco_json_path.open('w') as f:
            json.dump(coco_dataset, f, ensure_ascii=False, indent=2)

        print(f"COCO dataset has been saved to {coco_json_path}")

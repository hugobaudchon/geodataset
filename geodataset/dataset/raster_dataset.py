import json
from abc import ABC, abstractmethod
from warnings import warn
from pathlib import Path
from typing import List

import albumentations
import numpy as np
import rasterio
from shapely import box

from geodataset.utils import rle_segmentation_to_bbox, polygon_segmentation_to_bbox, CocoNameConvention, \
    collate_fn_raster_detection_dataset


class LabeledRasterCocoDataset(ABC):
    def __init__(self,
                 fold: str,
                 root_path: Path,
                 transform: albumentations.core.composition.Compose = None):
        """
        Parameters:
        - fold: str, the dataset fold to load (e.g., 'train', 'valid', 'test'...).
        - root_path: pathlib.Path, the root directory of the dataset.
        - transform: albumentations.core.composition.Compose, a composition of transformations to apply to the tiles.
        """
        self.fold = fold
        self.root_path = Path(root_path)
        self.transform = transform
        self.tiles = {}
        self.tiles_path_to_id_mapping = {}  # Mapping old image IDs to new ones to ensure uniqueness
        self.cocos_detected = []

        self._load_coco_datasets(directory=self.root_path)
        self._find_tiles_paths(directory=self.root_path)
        self._remove_tiles_not_found()

        if len(self.cocos_detected) == 0:
            raise Exception(f"No COCO datasets for fold {self.fold} were found in the specified root folder.")
        elif len(self.cocos_detected) > 0 and len(self) == 0:
            raise Exception(f"Could not find find any tiles associated with the COCO files found."
                            f" COCO files found: {self.cocos_detected}.")

        print(f"Found {len(self.tiles)} tiles and {sum([len(self.tiles[x]['labels']) for x in self.tiles])} labels"
              f" for fold {self.fold}.")

    def _load_coco_datasets(self, directory: Path):
        """
        Loads the dataset by traversing the directory tree and loading relevant COCO JSON files.
        """

        for path in directory.iterdir():
            if path.is_file() and path.name.endswith(f".json"):
                try:
                    product_name, scale_factor, fold = CocoNameConvention.parse_name(path.name)
                    if fold == self.fold:
                        self._load_coco_json(json_path=path)
                except ValueError:
                    return
            elif path.is_dir() and path.name != 'tiles':
                self._load_coco_datasets(directory=path)

    def _load_coco_json(self, json_path: Path):
        """
        Loads a COCO JSON file and its associated tiles.

        Parameters:
        - json_path: pathlib.Path, the path to the COCO JSON file.
        """

        with open(json_path) as f:
            coco_data = json.load(f)
        self._reindex_coco_data(coco_data=coco_data)
        self.cocos_detected.append(json_path)

    def _reindex_coco_data(self, coco_data: dict):
        """
        Re-indexes the COCO data to ensure unique tile and annotation IDs.

        Parameters:
        - coco_data: dict, the COCO dataset loaded from a JSON file.
        """
        coco_tiles = {}
        for image in coco_data['images']:
            tile_name = image['file_name']

            if tile_name not in self.tiles_path_to_id_mapping:
                new_tile_id = len(self.tiles)
                self.tiles_path_to_id_mapping[tile_name] = new_tile_id
                self.tiles[new_tile_id] = {
                    'name': tile_name,
                    'labels': []
                }

            # Keeping track of the original tile id from the COCO dataset to the tile name
            coco_tiles[image['id']] = tile_name

        for annotation in coco_data['annotations']:
            # Finding the associated tile name given the tile id from the COCO dataset
            tile_name = coco_tiles[annotation['image_id']]
            # Finding the tile id in this python dataset, not the tile id in the original COCO dataset
            tile_id = self.tiles_path_to_id_mapping[tile_name]
            self.tiles[tile_id]['labels'].append(annotation)

    def _find_tiles_paths(self, directory: Path):
        """
        Loads the dataset by traversing the directory tree and loading relevant COCO JSON files.
        """

        for path in directory.iterdir():
            if path.is_dir() and path.name == 'tiles':
                for tile_path in path.iterdir():
                    if tile_path.name in self.tiles_path_to_id_mapping:
                        tile_id = self.tiles_path_to_id_mapping[tile_path.name]
                        if 'path' in self.tiles[tile_id] and self.tiles[tile_id]['path'] != tile_path:
                            raise Exception(f"At least two tiles under the root directory {self.root_path} have the same"
                                            f" name, which is ambiguous. Make sure all tiles have unique names. The 2"
                                            f" ambiguous tiles are {tile_path} and {self.tiles[tile_id]['path']}.")
                        self.tiles[tile_id]['path'] = tile_path
            elif path.is_dir() and path.name != 'tiles':
                self._find_tiles_paths(directory=path)

    def _remove_tiles_not_found(self):
        original_tiles_number = len(self.tiles)
        tile_ids_to_delete = []
        for tile_id, tile in self.tiles.items():
            if 'path' not in tile:
                tile_ids_to_delete.append(tile_id)

        for tile_id in tile_ids_to_delete:
            del self.tiles[tile_id]

        new_tiles_number = len(self.tiles)

        if original_tiles_number != new_tiles_number:
            warn(f"Had to remove {original_tiles_number - new_tiles_number} tiles out of {original_tiles_number}"
                 f" as they could not be found in the root folder or its sub-folders.")

    @abstractmethod
    def __getitem__(self, idx: int):
        pass

    def __len__(self):
        """
        Returns the total number of tiles in the dataset.

        Returns:
        - An integer count of the tiles.
        """
        return len(self.tiles)

    @staticmethod
    @abstractmethod
    def get_collate_fn():
        pass


class DetectionLabeledRasterCocoDataset(LabeledRasterCocoDataset):
    def __init__(self, fold: str, root_path: Path):
        super().__init__(fold=fold, root_path=root_path)

    def __getitem__(self, idx: int):
        """
        Retrieves a tile and its annotations by index, applying any specified transformations.

        Parameters:
        - idx: int, the index of the tile to retrieve.

        Returns:
        - A tuple containing the transformed tile and its annotations.
        """
        tile_info = self.tiles[idx]

        with rasterio.open(tile_info['path']) as tile_file:
            tile = tile_file.read([1, 2, 3])  # Reading the first three bands

        labels = tile_info['labels']
        bboxes = []

        for label in labels:
            if 'bbox' in label:
                # Directly use the provided bbox
                bbox_coco = label['bbox']
                bbox = box(*[bbox_coco[0], bbox_coco[1], bbox_coco[0] + bbox_coco[2], bbox_coco[1] + bbox_coco[3]])
            else:
                segmentation = label['segmentation']
                if ('is_rle_format' in label and label['is_rle_format']) or isinstance(segmentation, dict):
                    # RLE format
                    bbox = rle_segmentation_to_bbox(segmentation)
                elif ('is_rle_format' in label and not label['is_rle_format']) or isinstance(segmentation, list):
                    # Polygon (coordinates) format
                    bbox = polygon_segmentation_to_bbox(segmentation)
                else:
                    raise NotImplementedError("Could not find the segmentation type (RLE vs polygon coordinates).")

            bboxes.append(bbox.bounds)

        if self.transform:
            transformed = self.transform(image=tile, bboxes=bboxes)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
        else:
            transformed_image = tile
            transformed_bboxes = bboxes

        return transformed_image, {'boxes': transformed_bboxes, 'labels': [1,] * len(transformed_bboxes)}

    @staticmethod
    def get_collate_fn():
        return collate_fn_raster_detection_dataset

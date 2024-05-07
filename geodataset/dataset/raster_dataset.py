import numpy as np
from typing import List
from pathlib import Path

import albumentations
import rasterio
from shapely import box

from geodataset.dataset.base_dataset import BaseDataset, BaseLabeledRasterCocoDataset
from geodataset.utils import rle_segmentation_to_bbox, polygon_segmentation_to_bbox, rle_segmentation_to_mask


class DetectionLabeledRasterCocoDataset(BaseLabeledRasterCocoDataset):
    def __init__(self, fold: str, root_path: Path or List[Path], transform: albumentations.core.composition.Compose = None):
        super().__init__(fold=fold, root_path=root_path, transform=transform)

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

            bboxes.append(np.array([int(x) for x in bbox.bounds]))

        category_ids = np.array([0 if label['category_id'] is None else label['category_id']
                                 for label in labels])

        if self.transform:
            transformed = self.transform(image=tile.transpose((1, 2, 0)),
                                         bboxes=bboxes,
                                         labels=category_ids)
            transformed_image = transformed['image'].transpose((2, 0, 1))
            transformed_bboxes = transformed['bboxes']
            transformed_category_ids = transformed['labels']
        else:
            transformed_image = tile
            transformed_bboxes = bboxes
            transformed_category_ids = category_ids

        transformed_image = transformed_image / 255  # normalizing
        # getting the areas of the boxes, assume pascal_voc box format
        area = np.array([(bboxe[3] - bboxe[1]) * (bboxe[2] - bboxe[0]) for bboxe in transformed_bboxes])
        # suppose all instances are not crowd
        iscrowd = np.zeros((len(transformed_bboxes),))
        # get tile id
        image_id = np.array([idx])
        # group annotations info
        transformed_bboxes = {'boxes': transformed_bboxes, 'labels': transformed_category_ids,
                              'area': area, 'iscrowd': iscrowd, 'image_id': image_id}

        return transformed_image, transformed_bboxes


class SegmentationLabeledRasterCocoDataset(BaseLabeledRasterCocoDataset):
    def __init__(self, fold: str, root_path: Path or List[Path], transform: albumentations.core.composition.Compose = None):
        super().__init__(fold=fold, root_path=root_path, transform=transform)

    def __getitem__(self, idx: int):
        """
        Retrieves a tile and its segmentations/masks by index, applying any specified transformations.

        Parameters:
        - idx: int, the index of the tile to retrieve.

        Returns:
        - A tuple containing the transformed tile and its segmentations/masks.
        """
        tile_info = self.tiles[idx]

        with rasterio.open(tile_info['path']) as tile_file:
            tile = tile_file.read([1, 2, 3])  # Reading the first three bands

        labels = tile_info['labels']
        masks = []

        for label in labels:
            if 'segmentation' in label:
                segmentation = label['segmentation']
                if ('is_rle_format' in label and label['is_rle_format']) or isinstance(segmentation, dict):
                    # RLE format
                    mask = rle_segmentation_to_mask(segmentation)
                else:
                    raise NotImplementedError("Please make sure that the masks are encoded using RLE.")

                masks.append(mask)

        category_ids = np.array([0 if label['category_id'] is None else label['category_id']
                                 for label in labels])

        if self.transform:
            transformed = self.transform(image=tile.transpose((1, 2, 0)),
                                         mask=np.stack(masks, axis=0),
                                         labels=category_ids)
            transformed_image = transformed['image'].transpose((2, 0, 1))
            transformed_masks = [mask for mask in transformed['mask']]
            transformed_category_ids = transformed['labels']
        else:
            transformed_image = tile
            transformed_masks = masks
            transformed_category_ids = category_ids

        transformed_image = transformed_image / 255  # normalizing
        area = np.array([np.sum(mask) for mask in masks])
        # suppose all instances are not crowd
        iscrowd = np.zeros((len(transformed_masks),))
        # get tile id
        image_id = np.array([idx])
        transformed_masks = {'masks': transformed_masks, 'labels': transformed_category_ids,
                             'area': area, 'iscrowd': iscrowd, 'image_id': image_id}

        return transformed_image, transformed_masks


class UnlabeledRasterDataset(BaseDataset):
    def __init__(self,
                 fold: str,
                 root_path: Path or List[Path],
                 transform: albumentations.core.composition.Compose = None):
        """
        Parameters:
        - fold: str, the dataset fold to load (e.g., 'train', 'valid', 'test'...).
        - root_path: pathlib.Path, the root directory of the dataset.
        - transform: albumentations.core.composition.Compose, a composition of transformations to apply to the tiles.
        """
        self.fold = fold
        self.root_path = root_path
        self.transform = transform
        self.tile_paths = []

        if isinstance(self.root_path, Path):
            self.root_path = [self.root_path]

        self._find_tiles_paths(directories=self.root_path)

    def _find_tiles_paths(self, directories: List[Path]):
        """
        Loads the dataset by traversing the directory tree and loading relevant COCO JSON files.
        """

        for directory in directories:
            if directory.is_dir() and directory.name == 'tiles':
                for path in directory.iterdir():
                    if path.suffix == ".tif":
                        self.tile_paths.append(path)

            if directory.is_dir():
                for path in directory.iterdir():
                    if path.is_dir():
                        self._find_tiles_paths(directories=[path])

    def __getitem__(self, idx: int):
        """
        Retrieves a tile and its annotations by index, applying any specified transformations.

        Parameters:
        - idx: int, the index of the tile to retrieve.

        Returns:
        - A tuple containing the transformed tile and its annotations.
        """
        tile_path = self.tile_paths[idx]

        with rasterio.open(tile_path) as tile_file:
            tile = tile_file.read([1, 2, 3])  # Reading the first three bands

        if self.transform:
            transformed = self.transform(image=tile.transpose((1, 2, 0)))
            transformed_image = transformed['image'].transpose((2, 0, 1))
        else:
            transformed_image = tile

        transformed_image = transformed_image / 255  # normalizing

        return transformed_image

    def __len__(self):
        """
        Returns the total number of tiles in the dataset.

        Returns:
        - An integer count of the tiles.
        """
        return len(self.tile_paths)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

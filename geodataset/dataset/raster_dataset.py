import numpy as np
from typing import List
from pathlib import Path

import albumentations
import rasterio
from shapely import box

from geodataset.dataset.base_dataset import BaseDataset, BaseLabeledRasterCocoDataset
from geodataset.utils import decode_coco_segmentation


class DetectionLabeledRasterCocoDataset(BaseLabeledRasterCocoDataset):
    """
    A dataset class that loads COCO datasets and their associated tiles (images). It will recursively search for COCO
    json files and .tif tiles in the specified root folder and its sub-folders. The COCO json files should follow the
    naming convention defined in the :class:`~geodataset.utils.CocoNameConvention` class. COCO jsons generated by this
    library should automatically follow this convention.

    Can be used for object detection tasks, where the annotations are bounding boxes OR segmentations (in this case
    this class will only use the bounding box of the segmentation).

    It can directly be used with a torch.utils.data.DataLoader.

    Parameters
    ----------
    fold: str
        The dataset fold to load (e.g., 'train', 'valid', 'test'...).
    root_path: str or List[str] or pathlib.Path or List[pathlib.Path]
        The root directory of the dataset.
    transform: albumentations.core.composition.Compose
        A composition of transformations to apply to the tiles and their associated annotations
        (applied in __getitem__).
    """
    def __init__(self,
                 fold: str,
                 root_path: str or List[str] or Path or List[Path],
                 transform: albumentations.core.composition.Compose = None,
                 box_padding_percentage: float = 0.0,
                 force_binary_class=None):
        super().__init__(fold=fold, root_path=root_path, transform=transform)
        self.box_padding_percentage = box_padding_percentage
        self.force_binary_class = force_binary_class

    def __getitem__(self, idx: int):
        """
        Retrieves a tile and its annotations by index, applying the transform passed to the constructor of the class,
        if any. It also normalizes the tile data between 0 and 1.

        Parameters
        ----------
        idx: int
            The index of the tile to retrieve

        Returns
        -------
        tuple of (numpy.ndarray, dict)
            The transformed tile (image) data, normalized between 0 and 1, and a dictionary containing the annotations
            and metadata of the tile. The dictionary has the following keys:

            - **boxes** (list of numpy.ndarray): A list of bounding boxes for the annotations.
            - **labels** (numpy.ndarray): An array of category ids for the annotations (same length as 'boxes').
            - **area** (list of float): A list of areas for the bounding boxes annotations (same length as 'boxes').
            - **iscrowd** (numpy.ndarray): An array of zeros (same length as 'boxes'). Currently not implemented.
            - **image_id** (numpy.ndarray): A single-value array containing the index of the tile.
        """
        tile_info = self.tiles[idx]

        with rasterio.open(tile_info['path']) as tile_file:
            tile = tile_file.read([1, 2, 3])  # Reading the first three bands

        labels = tile_info['labels']
        bboxes = []

        for label in labels:
            bbox = decode_coco_segmentation(label, 'bbox')

            if self.box_padding_percentage:
                minx, miny, maxx, maxy = bbox.bounds
                width = maxx - minx
                height = maxy - miny
                padding_x = width * (self.box_padding_percentage / 100)
                padding_y = height * (self.box_padding_percentage / 100)

                new_minx = max(0, minx - padding_x)
                new_miny = max(0, miny - padding_y)
                new_maxx = min(tile.shape[1], maxx + padding_x)
                new_maxy = min(tile.shape[2], maxy + padding_y)

                bbox = box(new_minx, new_miny, new_maxx, new_maxy)

            bboxes.append(np.array([int(x) for x in bbox.bounds]))

        if self.force_binary_class:
            category_ids = np.array([1 for _ in labels])
        else:
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
    """
    A dataset class that loads COCO datasets and their associated tiles (images). It will recursively search for COCO
    json files and .tif tiles in the specified root folder and its sub-folders. The COCO json files should follow the
    naming convention defined in the :class:`~geodataset.utils.CocoNameConvention` class. COCO jsons generated by this
    library should automatically follow this convention.

    Can be used for semantic segmentation tasks, where the annotations are segmentations.

    It can directly be used with a torch.utils.data.DataLoader.

    Parameters
    ----------
    fold: str
        The dataset fold to load (e.g., 'train', 'valid', 'test'...).
    root_path: str or List[str] or pathlib.Path or List[pathlib.Path]
        The root directory of the dataset.
    transform: albumentations.core.composition.Compose
        A composition of transformations to apply to the tiles and their associated annotations
        (applied in __getitem__).
    """
    def __init__(self,
                 fold: str,
                 root_path: str or List[str] or Path or List[Path],
                 transform: albumentations.core.composition.Compose = None,
                 force_binary_class=None):
        super().__init__(fold=fold, root_path=root_path, transform=transform)
        self.force_binary_class = force_binary_class

    def __getitem__(self, idx: int):
        """
        Retrieves a tile and its annotations by index, applying the transform passed to the constructor of the class,
        if any. It also normalizes the tile data between 0 and 1.

        Parameters
        ----------
        idx: int
            The index of the tile to retrieve

        Returns
        -------
        tuple of (numpy.ndarray, dict)
            The transformed tile (image) data, normalized between 0 and 1, and a dictionary containing the annotations
            and metadata of the tile. The dictionary has the following keys:

            - **masks** (list of numpy.ndarray): A list of segmentation masks for the annotations.
            - **labels** (numpy.ndarray): An array of category ids for the annotations (same length as 'masks').
            - **area** (list of float): A list of areas for the segmentation masks annotations (same length as 'masks').
            - **iscrowd** (numpy.ndarray): An array of zeros (same length as 'masks'). Currently not implemented.
            - **image_id** (numpy.ndarray): A single-value array containing the index of the tile.
        """
        tile_info = self.tiles[idx]

        with rasterio.open(tile_info['path']) as tile_file:
            tile = tile_file.read([1, 2, 3])  # Reading the first three bands

        labels = tile_info['labels']
        masks = []

        for label in labels:
            if 'segmentation' in label:
                mask = decode_coco_segmentation(label, 'mask')
                masks.append(mask)

        if self.force_binary_class:
            category_ids = np.array([1 for _ in labels])
        else:
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


class InstanceSegmentationLabeledRasterCocoDataset(BaseLabeledRasterCocoDataset):
    """
    A dataset class that loads COCO datasets and their associated tiles (images). It will recursively search for COCO
    json files and .tif tiles in the specified root folder and its sub-folders. The COCO json files should follow the
    naming convention defined in the :class:`~geodataset.utils.CocoNameConvention` class. COCO jsons generated by this
    library should automatically follow this convention.

    Can be used for instance segmentation, object detection and both tasks, where the annotations are segmentation masks
    and bounding boxes.

    It can directly be used with a torch.utils.data.DataLoader.

    Parameters
    ----------
    fold: str
        The dataset fold to load (e.g., 'train', 'valid', 'test'...).
    root_path: str or List[str] or pathlib.Path or List[pathlib.Path]
        The root directory of the dataset.
    transform: albumentations.core.composition.Compose
        A composition of transformations to apply to the tiles and their associated annotations
        (applied in __getitem__).
    """
    def __init__(self,
                 fold: str,
                 root_path: str or List[str] or Path or List[Path],
                 transform: albumentations.core.composition.Compose = None,
                 box_padding_percentage: float = 0.0,
                 force_binary_class=None):
        super().__init__(fold=fold, root_path=root_path, transform=transform)
        self.box_padding_percentage = box_padding_percentage
        self.force_binary_class = force_binary_class

    def __getitem__(self, idx: int):
        """
        Retrieves a tile and its annotations by index, applying the transform passed to the constructor of the class,
        if any. It also normalizes the tile data between 0 and 1.

        Parameters
        ----------
        idx: int
            The index of the tile to retrieve

        Returns
        -------
        tuple of (numpy.ndarray, dict)
            The transformed tile (image) data, normalized between 0 and 1, and a dictionary containing the annotations
            and metadata of the tile. The dictionary has the following keys:

            - **masks** (list of numpy.ndarray): A list of segmentation masks for the annotations.
            - **boxes** (list of numpy.ndarray): A list of bounding boxes for the annotations.
            - **labels** (numpy.ndarray): An array of category ids for the annotations (same length as 'masks').
            - **area** (list of float): A list of areas for the segmentation masks annotations (same length as 'masks').
            - **iscrowd** (numpy.ndarray): An array of zeros (same length as 'masks'). Currently not implemented.
            - **image_id** (numpy.ndarray): A single-value array containing the index of the tile.
        """
        tile_info = self.tiles[idx]

        with rasterio.open(tile_info['path']) as tile_file:
            tile = tile_file.read([1, 2, 3])  # Reading the first three bands

        labels = tile_info['labels']
        masks = []
        bboxes = []

        for label in labels:
            bbox = decode_coco_segmentation(label, 'bbox')
            mask = decode_coco_segmentation(label, 'mask')

            if self.box_padding_percentage:
                minx, miny, maxx, maxy = bbox.bounds
                width = maxx - minx
                height = maxy - miny
                padding_x = width * (self.box_padding_percentage / 100)
                padding_y = height * (self.box_padding_percentage / 100)

                new_minx = max(0, minx - padding_x)
                new_miny = max(0, miny - padding_y)
                new_maxx = min(tile.shape[0], maxx + padding_x)
                new_maxy = min(tile.shape[1], maxy + padding_y)

                bbox = box(new_minx, new_miny, new_maxx, new_maxy)

            bboxes.append(np.array([int(x) for x in bbox.bounds]))
            masks.append(mask)

        if self.force_binary_class:
            category_ids = np.array([1 for _ in labels])
        else:
            category_ids = np.array([0 if label['category_id'] is None else label['category_id']
                                     for label in labels])

        if self.transform:
            transformed = self.transform(image=tile.transpose((1, 2, 0)),
                                         mask=np.stack(masks, axis=2),
                                         bboxes=bboxes,
                                         labels=category_ids)
            transformed_image = transformed['image'].transpose((2, 0, 1))
            transformed_bboxes = transformed['bboxes']
            transformed_masks =  transformed['mask'].transpose((2, 0, 1))
            transformed_category_ids = transformed['labels']
        else:
            transformed_image = tile
            transformed_bboxes = bboxes
            transformed_masks = masks
            transformed_category_ids = category_ids

        transformed_image = transformed_image / 255  # normalizing
        # If needed, areas of the boxes, assume pascal_voc box format:
        # area = np.array([(bboxe[3] - bboxe[1]) * (bboxe[2] - bboxe[0]) for bboxe in transformed_bboxes])
        # We use area of masks
        area = np.array([np.sum(mask) for mask in masks])
        # suppose all instances are not crowd
        iscrowd = np.zeros((len(transformed_masks),))
        # get tile id
        image_id = np.array([idx])
        transformed_targets = {'masks': transformed_masks, 'boxes': transformed_bboxes,
                               'labels': transformed_category_ids, 'area': area, 'iscrowd': iscrowd,
                               'image_id': image_id}

        return transformed_image, transformed_targets


class UnlabeledRasterDataset(BaseDataset):
    """
    A dataset class for loading unlabeled raster tiles.
    It will recursively search for all '.tif' files in the specified root and its sub-folders.

    It can directly be used with a torch.utils.data.DataLoader.

    Parameters
    ----------
    fold: str
        The dataset fold to load (e.g., 'train', 'valid', 'test'...).
        **This parameter is not used in this class, but is kept for consistency with the other dataset classes.**
    root_path: str or List[str] or pathlib.Path or List[pathlib.Path]
        The root directory of the dataset.
    transform: albumentations.core.composition.Compose
        A composition of transformations to apply to the tiles and their associated annotations
        (applied in __getitem__).
    """
    def __init__(self,
                 root_path: str or List[str] or Path or List[Path],
                 transform: albumentations.core.composition.Compose = None,
                 fold: str = None):
        self.fold = fold
        self.root_path = root_path
        self.transform = transform
        self.tile_paths = []

        if isinstance(self.root_path, (str, Path)):
            self.root_path = [self.root_path]

        self.root_path = [Path(x) for x in self.root_path]

        self._find_tiles_paths(directories=self.root_path)

        print(f"Found {len(self.tile_paths)} tiles for fold {self.fold}.")

    def _find_tiles_paths(self, directories: List[Path]):
        """
        Loads the dataset by traversing the directory tree and loading relevant tiles metadata.
        """

        for directory in directories:
            if self.fold is not None:
                # If a fold is specified, load only the tiles for that fold
                if directory.is_dir() and directory.name == 'tiles':
                    fold_directory = (directory / self.fold)
                    # Datasets may not contain all splits
                    if fold_directory.exists():
                        for path in fold_directory.iterdir():
                            # Iterate within the corresponding split folder
                            if path.suffix == ".tif":
                                self.tile_paths.append(path)
            else:
                # If no fold is specified, load all tiles
                for path in directory.iterdir():
                    # Iterate within the corresponding split folder
                    if path.suffix == ".tif":
                        self.tile_paths.append(path)

            if directory.is_dir():
                for path in directory.iterdir():
                    if path.is_dir():
                        self._find_tiles_paths(directories=[path])

    def __getitem__(self, idx: int):
        """
        Retrieves a tile and its annotations by index, applying the transform passed to the constructor of the class,
        if any. It also normalizes the tile data between 0 and 1.

        Parameters
        ----------
        idx: int
            The index of the tile to retrieve

        Returns
        -------
        numpy.ndarray
            The transformed tile (image) data, normalized between 0 and 1.
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

        Returns
        -------
        int
            The number of tiles in the dataset.
        """
        return len(self.tile_paths)

    def __iter__(self):
        """
        Iterates over the tiles in the dataset.
        """
        for i in range(len(self)):
            yield self[i]

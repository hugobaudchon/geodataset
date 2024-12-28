import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union
from warnings import warn

import albumentations

from geodataset.utils import CocoNameConvention, PointCloudCocoNameConvention, find_tiles_paths


class BaseDataset(ABC):
    """
    Abstract class for a dataset. Requires the implementation of methods needed for use with PyTorch's DataLoader:
    __getitem__, __len__ and __iter__.
    """
    @abstractmethod
    def __getitem__(self, idx: int):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass


class BaseLabeledRasterCocoDataset(BaseDataset, ABC):
    """
    Abstract class for a dataset that loads COCO datasets and their associated tiles (images).
    It will recursively search for COCO json files and image tiles (.tif, .png...) in the specified root folder and its sub-folders.
    The COCO json files should follow the naming convention defined in the :class:`utils.CocoNameConvention` class.
    COCO jsons generated by this library should automatically follow this convention.

    This class implements the __len__ and __iter__ methods, and requires the implementation of the __getitem__ method.
    It is a great starting point to create your own custom dataset, if the ones provided in this library (ex: :class:`~geodataset.dataset.DetectionLabeledRasterCocoDataset`, :class:`~geodataset.dataset.SegmentationLabeledRasterCocoDataset`, ...) do not fit
    your needs.

    Parameters
    ----------
    fold: str
        The dataset fold to load (e.g., 'train', 'valid', 'test'...).
    root_path: str or List[str] or pathlib.Path or List[pathlib.Path]
        The root directory of the dataset.
    transform: albumentations.core.composition.Compose
        A composition of transformations to apply to the tiles.
    """
    def __init__(self,
                 fold: str,
                 root_path: str or List[str] or Path or List[Path],
                 transform: albumentations.core.composition.Compose = None):
        self.fold = fold
        self.root_path = root_path
        self.transform = transform
        self.tiles = {}
        self.tiles_path_to_id_mapping = {}
        self.category_id_to_metadata_mapping = {}
        self.cocos_detected = []

        if isinstance(self.root_path, (str, Path)):
            self.root_path = [self.root_path]

        self.root_path = [Path(x) for x in self.root_path]

        self._load_coco_datasets(directories=self.root_path)
        self._find_tiles_paths(directories=self.root_path)
        self._remove_tiles_not_found()
        self._filter_tiles_without_box()

        if len(self.cocos_detected) == 0:
            raise Exception(f"No COCO datasets for fold '{self.fold}' were found in the specified root folder.")
        elif len(self.cocos_detected) > 0 and len(self.tiles) == 0:
            raise Exception(f"Could not find find any tiles associated with the COCO files found."
                            f" COCO files found: {self.cocos_detected}.")

        print(f"Found {len(self.tiles)} tiles and {sum([len(self.tiles[x]['labels']) for x in self.tiles])} labels"
              f" for fold {self.fold}.")

    def _load_coco_datasets(self, directories: List[Path]):
        """
        Loads the dataset by traversing the directory tree and loading relevant COCO JSON files.
        """
        for directory in directories:
            for path in directory.iterdir():
                if path.is_file() and path.name.endswith(f".json"):
                    try:
                        product_name, scale_factor, ground_resolution, fold = CocoNameConvention.parse_name(path.name)
                        if fold == self.fold:
                            self._load_coco_json(json_path=path)
                    except ValueError:
                        return
                elif path.is_dir() and path.name != 'tiles':
                    self._load_coco_datasets(directories=[path])

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
                    'height': image['height'],
                    'width': image['width'],
                    'labels': []
                }

            # Keeping track of the original tile id from the COCO dataset to the tile name
            coco_tiles[image['id']] = tile_name

        for category in coco_data['categories']:
            category_id = category['id']
            if category_id not in self.category_id_to_metadata_mapping:
                self.category_id_to_metadata_mapping[category_id] = category
            elif category_id in self.category_id_to_metadata_mapping and self.category_id_to_metadata_mapping[category_id]['name'] != category['name']:
                raise Exception(f"Category with id {category_id} is duplicated in the COCO dataset,"
                                f" or the same category id exists in different COCO datasets present in the root folder.")

        for annotation in coco_data['annotations']:
            # Finding the associated tile name given the tile id from the COCO dataset
            tile_name = coco_tiles[annotation['image_id']]
            # Finding the tile id in this python dataset, not the tile id in the original COCO dataset
            tile_id = self.tiles_path_to_id_mapping[tile_name]
            if not annotation['category_id']:
                # Making sure any annotation without category is assigned None as category
                annotation['category_id'] = None
            self.tiles[tile_id]['labels'].append(annotation)

    def _find_tiles_paths(self, directories: List[Path]):
        """
        Loads the dataset by traversing the directory tree and loading relevant COCO JSON files.
        """
        for directory in directories:
            tiles_names_to_paths = find_tiles_paths([directory], extensions=['tif', 'png', 'jpg', 'jpeg'])

            for tile_name, tile_path in tiles_names_to_paths.items():
                if tile_name in self.tiles_path_to_id_mapping:
                    tile_id = self.tiles_path_to_id_mapping[tile_name]
                    if 'path' in self.tiles[tile_id] and self.tiles[tile_id]['path'] != tile_path:
                        raise Exception(
                            f"At least two tiles under the root directories {self.root_path} have the same"
                            f" name, which is ambiguous. Make sure all tiles have unique names. The 2"
                            f" ambiguous tiles are {tile_path} and {self.tiles[tile_id]['path']}.")
                    self.tiles[tile_id]['path'] = tile_path

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
            # Need to reindex the tile ids for __getitem__
            self._reindex_tiles()

    def _filter_tiles_without_box(self):
        # Remove tiles without annotation for training
        original_tiles_number = len(self.tiles)

        tile_ids_without_labels = []
        for tile_id, tile in self.tiles.items():
            if len(tile['labels']) == 0:
                tile_ids_without_labels.append(tile_id)

        for tile_id in tile_ids_without_labels:
            del self.tiles[tile_id]

        new_tiles_number = len(self.tiles)

        if original_tiles_number != new_tiles_number:
            warn(f"Had to remove {original_tiles_number - new_tiles_number} tiles out of {original_tiles_number}"
                 f" as they do not contain annotation in the training set.")
            # Need to reindex the tile ids for __getitem__
            self._reindex_tiles()

    def _reindex_tiles(self):
        reindexed_tiles = {}
        for i, (_, tile) in enumerate(self.tiles.items()):
            reindexed_tiles[i] = tile
        self.tiles = reindexed_tiles

    def __len__(self):
        """
        Returns the total number of tiles in the dataset.

        Returns:
        - An integer count of the tiles.
        """
        return len(self.tiles)

    def __iter__(self):
        """
        Iterates over the tiles in the dataset.
        """
        for i in range(len(self)):
            yield self[i]


class BaseLabeledPointCloudCocoDataset(BaseDataset, ABC):
    def __init__(self,
                 fold: str,
                 root_path: Union[str, List[str], Path, List[Path]],
                 ):
        self.fold = fold
        self.root_path = root_path
        self.tiles = {}
        self.tiles_path_to_id_mapping = {}
        self.category_id_to_metadata_mapping = {}
        self.cocos_detected = []

        if isinstance(self.root_path, (str, Path)):
            self.root_path = [self.root_path]

        self.root_path = [Path(x) for x in self.root_path]

        self._load_coco_datasets(directories=self.root_path)
        self._find_tiles_paths(directories=self.root_path)
        self._remove_tiles_not_found()
        self._filter_tiles_without_box()

        if len(self.cocos_detected) == 0:
            raise Exception(f"No COCO datasets for fold '{self.fold}' were found in the specified root folder.")
        elif len(self.cocos_detected) > 0 and len(self.tiles) == 0:
            raise Exception(f"Could not find find any tiles associated with the COCO files found."
                            f" COCO files found: {self.cocos_detected}.")

        print(f"Found {len(self.tiles)} tiles and {sum([len(self.tiles[x]['labels']) for x in self.tiles])} labels"
              f" for fold {self.fold}.")

    def _load_coco_datasets(self, directories: List[Path]):
        """
        Loads the dataset by traversing the directory tree and loading relevant COCO JSON files.
        """
        for directory in directories:
            for path in directory.iterdir():
                if path.is_file() and path.name.endswith(f".json") and "pc_coco" in path.name:
                    product_name, scale_factor, ground_resolution, voxel_size, fold = PointCloudCocoNameConvention.parse_name(
                        path.name)
                    if fold == self.fold:
                        self._load_coco_json(json_path=path)
                elif path.is_dir() and path.name != 'tiles' and not path.name.startswith("pc_tiles"):
                    self._load_coco_datasets(directories=[path])

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
        for image in coco_data['point_cloud']:

            tile_name = image['file_name']

            if tile_name not in self.tiles_path_to_id_mapping:
                new_tile_id = len(self.tiles)
                self.tiles_path_to_id_mapping[tile_name] = new_tile_id
                self.tiles[new_tile_id] = {
                    'name': tile_name,
                    'height': image['height'],
                    'width': image['width'],
                    'labels': [],
                    "min_x": image['min_x'],
                    "max_x": image['max_x'],
                    "min_y": image['min_y'],
                    "max_y": image['max_y'],
                }

            # Keeping track of the original tile id from the COCO dataset to the tile name
            coco_tiles[image['id']] = tile_name

        for category in coco_data['categories']:
            category_id = category['id']
            if category_id not in self.category_id_to_metadata_mapping:
                self.category_id_to_metadata_mapping[category_id] = category
            elif category_id in self.category_id_to_metadata_mapping and \
                    self.category_id_to_metadata_mapping[category_id]['name'] != category['name']:
                raise Exception(f"Category with id {category_id} is duplicated in the COCO dataset,"
                                f" or the same category id exists in different COCO datasets present in the root folder.")

        for annotation in coco_data['annotations']:
            # Finding the associated tile name given the tile id from the COCO dataset
            tile_name = coco_tiles[annotation['image_id']]
            # Finding the tile id in this python dataset, not the tile id in the original COCO dataset
            tile_id = self.tiles_path_to_id_mapping[tile_name]
            if not annotation['category_id']:
                # Making sure any annotation without category is assigned None as category
                annotation['category_id'] = None
            self.tiles[tile_id]['labels'].append(annotation)

    def _find_tiles_paths(self, directories: List[Path]):
        """
        Loads the dataset by traversing the directory tree and loading relevant COCO JSON files.
        """

        for directory in directories:
            if directory.is_dir() and directory.name.startswith("pc_tiles"):
                for tile_path in directory.iterdir():
                    if tile_path.name in self.tiles_path_to_id_mapping:
                        tile_id = self.tiles_path_to_id_mapping[tile_path.name]
                        if 'path' in self.tiles[tile_id] and self.tiles[tile_id]['path'] != tile_path:
                            raise Exception(
                                f"At least two tiles under the root directories {self.root_path} have the same"
                                f" name, which is ambiguous. Make sure all tiles have unique names. The 2"
                                f" ambiguous tiles are {tile_path} and {self.tiles[tile_id]['path']}.")
                        self.tiles[tile_id]['path'] = tile_path

            if directory.is_dir():
                for path in directory.iterdir():
                    if path.is_dir():
                        self._find_tiles_paths(directories=[path])

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
            # Need to reindex the tile ids for __getitem__
            self._reindex_tiles()

    def _filter_tiles_without_box(self):
        # Remove tiles without annotation for training
        original_tiles_number = len(self.tiles)

        tile_ids_without_labels = []
        for tile_id, tile in self.tiles.items():
            if len(tile['labels']) == 0:
                tile_ids_without_labels.append(tile_id)

        for tile_id in tile_ids_without_labels:
            del self.tiles[tile_id]

        new_tiles_number = len(self.tiles)

        if original_tiles_number != new_tiles_number:
            warn(f"Had to remove {original_tiles_number - new_tiles_number} tiles out of {original_tiles_number}"
                 f" as they do not contain annotation in the training set.")
            # Need to reindex the tile ids for __getitem__
            self._reindex_tiles()

    def _reindex_tiles(self):
        reindexed_tiles = {}
        for i, (_, tile) in enumerate(self.tiles.items()):
            reindexed_tiles[i] = tile
        self.tiles = reindexed_tiles

    def __len__(self):
        """
        Returns the total number of tiles in the dataset.

        Returns:
        - An integer count of the tiles.
        """
        return len(self.tiles)

    def __iter__(self):
        """
        Iterates over the tiles in the dataset.
        """
        for i in range(len(self)):
            yield self[i]
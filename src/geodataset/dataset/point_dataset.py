from geodataset.dataset.base_dataset import BaseLabeledPointCloudCocoDataset
from typing import Union, List
from pathlib import Path
import albumentations
import open3d as o3d
from geodataset.dataset.base_dataset import BaseDataset
import pandas as pd


class SegmentationLabeledPointCloudCocoDataset(BaseLabeledPointCloudCocoDataset):
    """
    A dataset class that loads COCO datasets and their associated tiles (point clouds). 

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
                 root_path: Union[str, List[str], Path, List[Path]],
                 label_type: str,
                 transform: albumentations.core.composition.Compose = None,
                 ):
        
        super().__init__(fold=fold, root_path=root_path)

        self.label_type = label_type
        self.transform = transform

        assert label_type in ['semantic', 'instance'], f"Invalid label type: {label_type}. Must be either 'semantic' or 'instance'."

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

        pcd =  o3d.t.io.read_point_cloud(tile_info['path'].as_posix())

        position = pcd.point.positions.numpy()
        semantic_labels = getattr(pcd.point, f"{self.label_type}_labels").numpy()

        #TODO: Might be needed later
        # labels = tile_info['labels']
        # bboxes = []

        # for label in labels:
        #     if 'bbox' in label:
        #         # Directly use the provided bbox
        #         bbox_coco = label['bbox']
        #         bbox = box(*[bbox_coco[0], bbox_coco[1], bbox_coco[0] + bbox_coco[2], bbox_coco[1] + bbox_coco[3]])
        #     else:
        #         segmentation = label['segmentation']
        #         if ('is_rle_format' in label and label['is_rle_format']) or isinstance(segmentation, dict):
        #             # RLE format
        #             bbox = coco_rle_segmentation_to_bbox(segmentation)
        #         elif ('is_rle_format' in label and not label['is_rle_format']) or isinstance(segmentation, list):
        #             # Polygon (coordinates) format
        #             bbox = coco_coordinates_segmentation_to_bbox(segmentation)
        #         else:
        #             raise NotImplementedError("Could not find the segmentation type (RLE vs polygon coordinates).")

        #     bboxes.append(np.array([int(x) for x in bbox.bounds]))


        if self.transform:
            position, semantic_labels = self.transform((position, semantic_labels))

        return position, semantic_labels

class BasePointDataset(BaseDataset):
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
                 root_path: Union[str, List[str], Path, List[Path]],
                 fold: str = None, 
                 extension: str = 'pcd'):
        
        self.fold = fold
        self.root_path = root_path
        self.tile_paths = []
        self.extension = extension
        self.metadata_df = pd.DataFrame(columns = ["tile_name","min_x","max_x","min_y","max_y","crs"])

        if isinstance(self.root_path, (str, Path)):
            self.root_path = [self.root_path]

        self.root_path = [Path(x) for x in self.root_path]

        self._find_tiles_paths(directories=self.root_path)

        print(f"Found {len(self.tile_paths)} tiles for {fold}.")

    def _find_tiles_paths(self, directories: List[Path]):
        """
        Loads the dataset by traversing the directory tree and loading relevant COCO JSON files.
        """

        for directory in directories:
            if directory.is_dir() and 'pc_tiles' in directory.name:
                for file in directory.iterdir():
                    if file.suffix == '.csv':
                        df = pd.read_csv(file)
                        df = df[df["tile_name"].str.contains(self.fold)]
                        if all(x == y for x, y in zip(df.columns, self.metadata_df.columns)):
                            if self.metadata_df.empty:
                                self.metadata_df = df
                            else:
                                self.metadata_df = pd.concat([self.metadata_df,df], axis=0)
                # Iterate within the 'pc_tiles' folder
                fold_directory = (directory / self.fold)
                # Datasets may not contain all splits
                if fold_directory.exists():
                    for path in fold_directory.iterdir():
                        # Iterate within the corresponding split folder
                        if path.suffix == f".{self.extension}":
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

        pcd =  o3d.t.io.read_point_cloud(tile_path) 
        
        return pcd

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

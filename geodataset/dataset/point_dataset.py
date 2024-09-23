from geodataset.dataset.base_dataset import BaseLabeledPointCloudCocoDataset
from typing import Union, List
from pathlib import Path
import albumentations
import open3d as o3d
import numpy as np
from geodataset.utils import coco_rle_segmentation_to_bbox, coco_coordinates_segmentation_to_bbox, coco_rle_segmentation_to_mask
from shapely import box


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
        
        print(f"transforms not being used" )
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
            transformed_positions, transformed_semantic_labels = self.transform((position, semantic_labels))
            position = transformed_positions
            semantic_labels = transformed_semantic_labels

        return position, semantic_labels

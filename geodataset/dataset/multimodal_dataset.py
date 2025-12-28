import random
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import laspy
import numpy as np
import rasterio

from geodataset.dataset.base_dataset import BaseLabeledRasterCocoDataset
from geodataset.utils import decode_coco_segmentation


class LabeledMultiModalCocoDataset(BaseLabeledRasterCocoDataset):
    """
    A dataset class for classification tasks using polygon-based tiles from raster and point cloud data. Loads COCO
    datasets and their associated tiles.

    It can directly be used with a torch.utils.data.DataLoader.

    Parameters
    ----------
    fold: str
        The dataset fold to load (e.g., 'train', 'valid', 'test'...)
    root_path: Union[str, List[str], Path, List[Path]]
        The root directory of the dataset
    augment: An optional augmentation pipeline for the dataset
    other_attributes_names_to_pass: List[str]
        A list of the names of some other COCO annotations attributes to return when iterating over the dataset
         (like a global_id, confidence_score...).
    """

    def __init__(
        self,
        dataset_name: str,
        fold: str,
        root_path: Union[str, List[str], Path, List[Path]],
        modalities: List[str] = ["image", "point_cloud", "dsm"],
        tasks: List[str] = ["classification"],
        voxel_size: Optional[float] = None,
        num_points_file: Optional[int] = None,
        downsample_method_file: Optional[str] = None,
        augment: Optional = None,
        num_points: Optional[int] = None,
        few_shot_path: Optional[str] = None,
        num_downsample_seeds: Optional[int] = 1,
        exclude_classes: Optional[List[int]] = None,
        outlier_removal: Optional[str] = None,
        height_attr: Optional[str] = None,
    ):
        available_tasks = {"classification", "height", "segmentation"}
        if not all(t in available_tasks for t in tasks):
            raise ValueError(
                f"Unsupported task in {tasks}. Allowed tasks are: {available_tasks}"
            )
        if "height" in tasks:
            self.height_attr = height_attr
            other_attributes_names_to_pass = [height_attr]
        else:
            other_attributes_names_to_pass = None
        super().__init__(
            fold=fold,
            root_path=root_path,
            transform=None,
            other_attributes_names_to_pass=other_attributes_names_to_pass,
            keep_unlabeled=False,
            few_shot_path=few_shot_path,
            exclude_classes=exclude_classes,
        )
        self.dataset_name = dataset_name
        self.modalities = modalities
        self.tasks = tasks
        self.augment = augment
        self.num_points = num_points
        self.num_downsample_seeds = num_downsample_seeds
        self.outlier_removal = outlier_removal

        # Add point cloud tile paths to tiles
        for tile in self.tiles.values():
            labels = tile["labels"]
            labels[0]["category_id"]
            if "point_cloud" in self.modalities:
                tile["point_cloud_path"] = self._derive_point_cloud_path(
                    tile["path"],
                    voxel_size,
                    num_points_file,
                    downsample_method_file,
                    outlier_removal,
                )
            if "dsm" in self.modalities:
                tile["dsm_path"] = self._derive_dsm_path(tile["path"])

    def _derive_dsm_path(self, raster_path: Union[str, Path]) -> Path:
        if self.dataset_name == "quebec_plantations":
            dsm_path = Path(str(raster_path).replace("_rgb", "_dsm"))
            return dsm_path
        elif self.dataset_name == "quebec_trees":
            # Look for pattern: YYYY_MM_DD_<site>_z<number>
            m = re.search(
                r"(\d{4})_(\d{2})_(\d{2})_([a-zA-Z]+)_(z\d+)_rgb", raster_path.name
            )
            if not m:
                raise ValueError(
                    f"Could not parse site/date pattern from {raster_path}"
                )
            date_str = f"{m[1]}{m[2]}{m[3]}"  # e.g. 20210902
            site_str = f"{m[4]}{m[5]}"  # e.g. sblz2
            new_prefix = f"{date_str}_{site_str}_p4rtk_dsm"
            old_prefix = f"{m[1]}_{m[2]}_{m[3]}_{m[4]}_{m[5]}_rgb"
            dsm_path = Path(str(raster_path).replace(old_prefix, new_prefix))
            return dsm_path
        elif self.dataset_name == "bci":
            dsm_path = Path(str(raster_path).replace("_orthomosaic", "_dsm"))
            return dsm_path
        else:
            raise ValueError(f"Unknown dataset {self.dataset_name}")

    def _derive_point_cloud_path(
        self,
        raster_path: Union[str, Path],
        voxel_size: float,
        num_points_file: int,
        downsample_method_file: str,
        outlier_removal: Optional[str] = None,
    ) -> Path:
        """
        Convert:
          .../<scene>/<scene>_rgb/tiles/<fold>/<scene>_rgb_tile_<fold>_sf1p0_<coords>.tif
        to:
          .../<scene>/<scene>_pcd/pc_tiles_0.01/<fold>/<scene>_pg_highdis_pc_tile_<fold>_sf1p0_vs0p01_<coords>.laz
        """
        p = Path(raster_path)
        # Expect structure: .../<scene>/<scene>_rgb/tiles/<fold>/<filename>.tif
        fold_dir = p.parent.name
        tiles_dir = p.parent.parent
        scene_rgb_dir = tiles_dir.parent
        scene_root = scene_rgb_dir.parent
        scene_rgb_name = scene_rgb_dir.name
        if self.dataset_name == "quebec_plantations":
            scene_pcd_dir_name = scene_rgb_name.replace("_rgb", "_pcd")
        elif self.dataset_name == "quebec_trees":
            # Look for pattern: YYYY_MM_DD_<site>_z<number>
            m = re.search(
                r"(\d{4})_(\d{2})_(\d{2})_([a-zA-Z]+)_(z\d+)_rgb", scene_rgb_name
            )
            if not m:
                raise ValueError(
                    f"Could not parse site/date pattern from {scene_rgb_name}"
                )
            date_str = f"{m[1]}{m[2]}{m[3]}"  # e.g. 20210902
            site_str = f"{m[4]}{m[5]}"  # e.g. sblz2
            new_prefix = f"{date_str}_{site_str}_pcd"
            old_prefix = f"{m[1]}_{m[2]}_{m[3]}_{m[4]}_{m[5]}_rgb"
            scene_pcd_dir_name = Path(
                str(scene_rgb_name).replace(old_prefix, new_prefix)
            )
        elif self.dataset_name == "bci":
            scene_pcd_dir_name = scene_rgb_name.replace("_orthomosaic", "_pcd")
        else:
            raise ValueError(f"Unknown dataset {self.dataset_name}")
        # Format voxel size for dir and filename parts

        def fmt_voxel_dir(v: Union[str, float, int]) -> str:
            if isinstance(v, (float, int)):
                s = f"{v:.8f}".rstrip("0").rstrip(".")
            else:
                s = str(v)
            return s

        if voxel_size is not None and num_points_file is None:
            voxel_dir = f"_{fmt_voxel_dir(voxel_size)}"  # e.g. "_0.01"
            voxel_token = voxel_dir.replace(".", "p")
            voxel_token = voxel_token.replace("_", "vs")  # e.g. "vs0p01"
        elif voxel_size is None and num_points_file is not None:
            voxel_dir = f"_{num_points_file}"
            voxel_token = voxel_dir.replace("_", "np")  # e.g. "np2048"
            if downsample_method_file == "random":
                voxel_dir += "_r"  # add _r in directory name
        elif voxel_size is None and num_points_file is None:
            voxel_dir = ""
            voxel_token = ""
        else:
            raise ValueError(
                "Only one of voxel_size or num_points_file should be provided"
            )
        if outlier_removal is not None:
            voxel_dir += f"_or_{outlier_removal}"
            parts = outlier_removal.split("_")
            processed_parts = []
            for part in parts:
                if "k" in part or "z" in part:
                    processed_parts.append("o" + part)
                else:
                    processed_parts.append(part)
            voxel_token += "_" + "_".join(processed_parts)
        pcd_base = scene_root / scene_pcd_dir_name / f"pc_tiles{voxel_dir}" / fold_dir
        stem = p.stem  # original filename without .tif
        if self.dataset_name == "quebec_plantations":
            new_stem = stem.replace("_rgb_tile_", "_pg_pc_tile_", 1)
        elif self.dataset_name == "quebec_trees":
            new_stem = stem.replace("_rgb_tile_", "_pc_tile_", 1)
        elif self.dataset_name == "bci":
            new_stem = stem.replace("_orthomosaic_tile_", "_cloud_pc_tile_", 1)
        if len(voxel_token) > 0:
            # Insert voxel-size tag after the sf… chunk (e.g., _sf1p0_)
            new_stem = re.sub(r"(_sf[^_]+_)", rf"\1{voxel_token}_", new_stem, count=1)
        new_name = new_stem + ".laz"
        pcd_path = pcd_base / new_name
        return pcd_path

    def _get_image_data(
        self, tile_info: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reads and normalizes the RGB image tile."""
        with rasterio.open(tile_info["path"]) as tile_file:
            # Read first 3 bands (RGB) or handle grayscale
            if tile_file.count >= 3:
                img = tile_file.read([1, 2, 3])
            else:
                img = tile_file.read()
                if img.shape[0] == 1:  # If single band, replicate to 3 channels
                    img = np.repeat(img, 3, axis=0)
            transform = tile_file.transform
            resolution = tile_file.res

        # Normalize the image data (handle 8-bit vs 16-bit)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            raise ValueError(
                f"Unexpected image dtype {img.dtype} in {tile_info['path']}"
            )
        return img, transform, resolution

    def _get_point_cloud_path(self, tile_info: Dict[str, Any]) -> Path:
        """Determines the correct point cloud path, applying downsampling/seeding logic."""
        original_path = Path(tile_info["point_cloud_path"])
        path_parts = list(original_path.parts)
        path_target = path_parts[9]

        seed_idx = 0
        if self.num_downsample_seeds > 1:
            seed_idx = random.randrange(self.num_downsample_seeds)

        path_parts[9] = f"{path_target}_s{seed_idx}"
        return Path(*path_parts)

    def _process_point_cloud(
        self, tile_info: Dict[str, Any], img_size: int, transform: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Loads, geo-references, scales, and samples the point cloud."""
        inv_transform = ~transform  # Inverse transform: world → pixel

        # Calculate xy_scale
        pixel_size_x = abs(transform.a)
        pixel_size_y = abs(transform.e)
        rtol = 0.05
        if np.isclose(pixel_size_x, pixel_size_y, rtol=rtol):
            xy_scale = 0.5 * (pixel_size_x + pixel_size_y)
        else:
            raise ValueError(
                f"Pixel sizes are not close enough: {pixel_size_x} vs {pixel_size_y}"
            )

        if transform.b != 0 or transform.d != 0:
            warnings.warn(
                f"Affine has rotation/shear (b={transform.b}, d={transform.d}) for {tile_info['path']}"
            )

        point_cloud_path = self._get_point_cloud_path(tile_info)

        with laspy.open(point_cloud_path) as point_cloud_file:
            las = point_cloud_file.read()

        x, y, z = las.x, las.y, np.asarray(las.z, dtype=np.float32)
        cols, rows = inv_transform * (x, y)

        # Scale coordinates (normalized to [-1, 1] for x/y and z)
        # Note: z is offset by its mean before scaling.
        half_img_size = img_size / 2
        rows_scaled = (rows - half_img_size) / half_img_size
        cols_scaled = (cols - half_img_size) / half_img_size
        z_scaled = (z - z.mean()) / xy_scale / half_img_size

        # Normalize RGB colors
        r, g, b = las.red / 65535.0, las.green / 65535.0, las.blue / 65535.0

        points = np.vstack(
            (rows_scaled, cols_scaled, z_scaled, r, g, b), dtype=np.float32
        ).T

        # Filter points outside the tile boundary (with 2-pixel tolerance)
        H, W = img_size, img_size
        valid_mask = (
            (rows >= -2.5) & (rows < H + 1.5) & (cols >= -2.5) & (cols < W + 1.5)
        )
        points = points[valid_mask]

        # Apply final point cloud sampling if required
        if self.num_points is not None:
            N = points.shape[0]
            if self.num_points < N:
                indices = np.random.choice(
                    N, self.num_points, replace=(N >= self.num_points)
                )
                points = points[indices]
            elif self.num_points > N:
                raise ValueError(
                    f"Tile {tile_info['path']} has only {N} points, fewer than requested {self.num_points}"
                )

        return points, xy_scale

    def _get_dsm_data(
        self, tile_info: Dict[str, Any]
    ) -> Tuple[np.ndarray, float, float]:
        """Reads, normalizes, and scales the Digital Surface Model (DSM)."""
        dsm_path = tile_info["dsm_path"]
        with rasterio.open(dsm_path) as dsm_file:
            dsm = dsm_file.read().astype(np.float32)

        # Normalize the DSM [0, 1]
        dsm_offset = np.nanmin(dsm)
        dsm_scale = np.nanmax(dsm) - dsm_offset
        dsm = (dsm - dsm_offset) / dsm_scale

        return dsm, dsm_offset, dsm_scale

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Retrieves a tile and its associated data (Image, Point Cloud, DSM) and targets by index, applying transforms
        if specified."""
        tile_info = self.tiles[idx]
        xy_scale = None
        dsm_offset = None
        dsm_scale = None

        # Load Image Data
        img, transform, resolution = self._get_image_data(tile_info)
        img_size = img.shape[1]

        # Prepare Mask Data (if segmentation task is active)
        mask = None
        if "segmentation" in self.tasks:
            # Assuming decode_coco_segmentation is available
            mask = decode_coco_segmentation(tile_info["labels"][0], "mask")
            mask = mask[np.newaxis, :, :]  # [1, H, W]

        # Load/Process Point Cloud Data (if modality is active)
        points = None
        if "point_cloud" in self.modalities:
            points, xy_scale = self._process_point_cloud(tile_info, img_size, transform)

        # Load/Process DSM Data (if modality is active)
        dsm = None
        if "dsm" in self.modalities:
            dsm, dsm_offset, dsm_scale = self._get_dsm_data(tile_info)

        # Apply Augmentations
        if self.augment:
            img, points, dsm, mask = self.augment(img, points, dsm, mask)

        # Prepare Targets
        labels = tile_info["labels"]
        if len(labels) > 1:
            raise NotImplementedError(
                "Multi-label tasks are not implemented in this dataset."
            )
        label = labels[0]
        targets = {}
        if "classification" in self.tasks:
            targets["class"] = label["category_id"]
        if "height" in self.tasks:
            other_attributes = self._get_other_attributes_to_pass(idx)
            targets["height"] = other_attributes[self.height_attr][0]
        if "segmentation" in self.tasks:
            targets["segmentation"] = mask

        # Prepare Metadata
        meta = {"filename": Path(tile_info["path"]).stem}
        meta["resolution"] = resolution
        if "point_cloud" in self.modalities:
            meta["xy_scale"] = float(xy_scale)
        if "dsm" in self.modalities:
            meta["dsm_offset"] = float(dsm_offset)
            meta["dsm_scale"] = float(dsm_scale)

        return meta, img, points, dsm, targets

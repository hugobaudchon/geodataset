import random
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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
    dataset_name: str
        The name identifying the dataset.
    fold: str
        The dataset fold to load ('train', 'valid', 'test').
    root_path: Union[str, List[str], Path, List[Path]]
        The root directory of the dataset
    modalities: List[str]
        List of modalities to include ('image', 'point_cloud', 'dsm').
    tasks: List[str]
        List of tasks to perform ('classification', 'height', 'segmentation').
    voxel_size: Optional[float]
        Voxel size for point cloud downsampling.
    num_points_file: Optional[int]
        Target number of points per file.
    downsample_method_file: Optional[str]
        The method used for file-level downsampling.
    augment: Optional[Any]
        An optional augmentation pipeline for the dataset.
    num_points: Optional[int]
        Fixed number of points to sample during loading.
    few_shot_path: Optional[str]
        Path to few-shot configuration files.
    num_downsample_seeds: int
        Number of seeds to use for downsampling.
    exclude_classes: Optional[List[int]]
        List of COCO category IDs to ignore.
    outlier_removal: Optional[str]
        Method for removing outliers from the point cloud.
    height_attr: Optional[str]
        The COCO attribute name containing height information.
    """

    AVAILABLE_TASKS: Set[str] = {"classification", "height", "segmentation"}

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
        augment: Optional[Any] = None,
        num_points: Optional[int] = None,
        few_shot_path: Optional[str] = None,
        num_downsample_seeds: Optional[int] = 1,
        exclude_classes: Optional[List[int]] = None,
        outlier_removal: Optional[str] = None,
        height_attr: Optional[str] = None,
    ) -> None:
        """
        Initializes the multimodal dataset and validates task compatibility.

        Parameters
        ----------
        (See Class Docstring for parameter details)

        Returns
        -------
        None
        """
        # Validate tasks
        unsupported = set(tasks) - self.AVAILABLE_TASKS
        if unsupported:
            raise ValueError(
                f"Unsupported tasks: {unsupported}. Allowed: {self.AVAILABLE_TASKS}"
            )

        # Handle height attribute passing
        self.height_attr = height_attr
        other_attributes_names_to_pass = []
        if "height" in tasks and height_attr:
            other_attributes_names_to_pass.append(height_attr)

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

        # Pre-process tile paths for selected modalities
        self._initialize_modality_paths(
            voxel_size, num_points_file, downsample_method_file
        )

    def _initialize_modality_paths(
        self,
        voxel_size: Optional[float],
        num_points_file: Optional[int],
        downsample_method_file: Optional[str],
    ) -> None:
        """
        Populates tile metadata with associated file paths for different modalities.

        Parameters
        ----------
        voxel_size : float, optional
            Voxel size for point cloud processing.
        num_points_file : int, optional
            Point count target for the files.
        downsample_method_file : str, optional
            Method used for file-level downsampling.

        Returns
        -------
        None
        """
        for tile in self.tiles.values():
            base_path = Path(tile["path"])

            if "point_cloud" in self.modalities:
                tile["point_cloud_path"] = self._derive_point_cloud_path(
                    str(base_path),
                    voxel_size,
                    num_points_file,
                    downsample_method_file,
                    self.outlier_removal,
                )

            if "dsm" in self.modalities:
                tile["dsm_path"] = self._derive_dsm_path(str(base_path))

    def _derive_dsm_path(self, raster_path: Union[str, Path]) -> Path:
        """
        Resolves the Digital Surface Model (DSM) path based on the dataset-specific naming convention.

        Parameters
        ----------
        raster_path : Union[str, Path]
            The file path to the original RGB raster tile.

        Returns
        -------
        Path
            The resolved path to the corresponding DSM file.

        Raises
        -------
        ValueError
            If the dataset name is unknown or the filename pattern cannot be parsed.
        """
        raster_path = Path(raster_path)
        filename = raster_path.name

        if self.dataset_name == "quebec_plantations":
            return raster_path.with_name(filename.replace("_rgb", "_dsm"))

        elif self.dataset_name == "quebec_trees":
            m = re.search(r"(\d{4})_(\d{2})_(\d{2})_([a-zA-Z]+)_(z\d+)_rgb", filename)
            if not m:
                raise ValueError(
                    f"Could not parse quebec_trees pattern from {filename}"
                )

            # Reconstruct prefix: YYYYMMDD_sitezn_p4rtk_dsm
            date_str, site_str = f"{m[1]}{m[2]}{m[3]}", f"{m[4]}{m[5]}"
            new_prefix = f"{date_str}_{site_str}_p4rtk_dsm"
            old_prefix = f"{m[1]}_{m[2]}_{m[3]}_{m[4]}_{m[5]}_rgb"
            return raster_path.with_name(filename.replace(old_prefix, new_prefix))

        elif self.dataset_name == "bci":
            return raster_path.with_name(filename.replace("_orthomosaic", "_dsm"))

        else:
            raise ValueError(f"Unknown dataset {self.dataset_name}")

    def _derive_point_cloud_path(
        self,
        raster_path: Union[str, Path],
        voxel_size: Optional[float],
        num_points_file: Optional[int],
        downsample_method_file: Optional[str],
        outlier_removal: Optional[str] = None,
    ) -> Path:
        """
        Maps a raster tile path to its corresponding point cloud (.laz) tile path by traversing the directory structure
        and applying voxel/downsampling tokens.

        Parameters
        ----------
        raster_path : Union[str, Path]
            The file path to the RGB raster tile.
        voxel_size : float, optional
            Voxel size for downsampling (mutually exclusive with num_points_file).
        num_points_file : int, optional
            Target point count (mutually exclusive with voxel_size).
        downsample_method_file : str, optional
            The method used (e.g., 'random').
        outlier_removal : str, optional
            The outlier removal strategy used during preprocessing.

        Returns
        -------
        Path
            The resolved path to the .laz point cloud tile.

        Raises
        -------
        ValueError
            If parameters are conflicting or the directory structure is unexpected.
        """
        p = Path(raster_path)
        fold_dir = p.parent.name
        scene_rgb_dir = p.parent.parent.parent
        scene_root = scene_rgb_dir.parent
        scene_rgb_name = scene_rgb_dir.name

        # 1. Resolve Scene PCD Directory Name
        if self.dataset_name == "quebec_plantations":
            scene_pcd_dir_name = scene_rgb_name.replace("_rgb", "_pcd")
        elif self.dataset_name == "quebec_trees":
            m = re.search(
                r"(\d{4})_(\d{2})_(\d{2})_([a-zA-Z]+)_(z\d+)_rgb", scene_rgb_name
            )
            if not m:
                raise ValueError(f"Pattern mismatch in scene dir: {scene_rgb_name}")
            old_prefix = f"{m[1]}_{m[2]}_{m[3]}_{m[4]}_{m[5]}_rgb"
            new_prefix = f"{m[1]}{m[2]}{m[3]}_{m[4]}{m[5]}_pcd"
            scene_pcd_dir_name = scene_rgb_name.replace(old_prefix, new_prefix)
        elif self.dataset_name == "bci":
            scene_pcd_dir_name = scene_rgb_name.replace("_orthomosaic", "_pcd")
        else:
            raise ValueError(f"Unknown dataset {self.dataset_name}")

        # 2. Build Voxel/Downsample Tokens
        voxel_dir, voxel_token = "", ""
        if voxel_size is not None and num_points_file is None:
            v_str = f"{voxel_size:.8f}".rstrip("0").rstrip(".")
            voxel_dir = f"_{v_str}"
            voxel_token = f"vs{v_str.replace('.', 'p')}"
        elif num_points_file is not None and voxel_size is None:
            voxel_dir = f"_{num_points_file}"
            voxel_token = f"np{num_points_file}"
            if downsample_method_file == "random":
                voxel_dir += "_r"
        elif voxel_size is not None and num_points_file is not None:
            raise ValueError("Provide either voxel_size OR num_points_file, not both.")

        # 3. Handle Outlier Removal Suffixes
        if outlier_removal is not None:
            voxel_dir += f"_or_{outlier_removal}"
            processed = [
                "o" + part if any(c in part for c in "kz") else part
                for part in outlier_removal.split("_")
            ]
            voxel_token += "_" + "_".join(processed)

        # 4. Construct Final Path
        pcd_base = scene_root / scene_pcd_dir_name / f"pc_tiles{voxel_dir}" / fold_dir
        stem = p.stem

        replacements = {
            "quebec_plantations": ("_rgb_tile_", "_pg_pc_tile_"),
            "quebec_trees": ("_rgb_tile_", "_pc_tile_"),
            "bci": ("_orthomosaic_tile_", "_cloud_pc_tile_"),
        }
        old, new = replacements[self.dataset_name]
        new_stem = stem.replace(old, new, 1)

        if voxel_token:
            new_stem = re.sub(r"(_sf[^_]+_)", rf"\1{voxel_token}_", new_stem, count=1)

        return pcd_base / f"{new_stem}.laz"

    def _get_image_data(
        self, tile_info: Dict[str, Any]
    ) -> Tuple[np.ndarray, Any, Tuple[float, float]]:
        """
        Reads, normalizes, and extracts geospatial metadata from an RGB raster tile.

        Parameters
        ----------
        tile_info : Dict[str, Any]
            Dictionary containing the 'path' key to the raster file.

        Returns
        -------
        img : np.ndarray
            The image data as a float32 array in (C, H, W) format, normalized to [0, 1].
        transform : rasterio.affine.Affine
            The affine transform mapping pixel coordinates to geographic coordinates.
        resolution : Tuple[float, float]
            The pixel resolution (x_res, y_res) of the tile.

        Raises
        -------
        ValueError
            If the image bit-depth is not 8-bit (uint8).
        """

        with rasterio.open(tile_info["path"]) as tile_file:
            # Handle band extraction
            if tile_file.count >= 3:
                img = tile_file.read([1, 2, 3])
            else:
                img = tile_file.read()
                if img.shape[0] == 1:
                    img = np.repeat(img, 3, axis=0)
            transform = tile_file.transform
            resolution = tile_file.res

        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            raise ValueError(
                f"Expected uint8, found {img.dtype} in {tile_info['path']}"
            )

        return img, transform, resolution

    def _get_point_cloud_path(self, tile_info: Dict[str, Any]) -> Path:
        """
        Determines the final point cloud path by injecting a random seed suffix at a specific directory depth (index 9).

        Parameters
        ----------
        tile_info : Dict[str, Any]
            Dictionary containing metadata for the tile, specifically the
            base 'point_cloud_path'.

        Returns
        -------
        Path
            The resolved Path object pointing to the specific seeded directory.

        Raises
        -------
        IndexError
            If the point cloud path does not have enough directory levels to
            access index 9.
        """
        original_path = Path(tile_info["point_cloud_path"])

        # Skip logic if no multiple seeds are defined
        if not self.num_downsample_seeds or self.num_downsample_seeds <= 1:
            return original_path

        # Randomly select which pre-computed seed folder to use
        seed_idx = random.randrange(self.num_downsample_seeds)

        path_parts = list(original_path.parts)

        # Safety check: Ensure the path is deep enough to have an index 9
        if len(path_parts) <= 9:
            raise IndexError(
                f"Path index 9 out of range for: {original_path}. "
                f"Path only has {len(path_parts)} parts."
            )

        # Inject the seed suffix (e.g., 'pc_tiles_0.01' -> 'pc_tiles_0.01_s2')
        path_target = path_parts[9]
        path_parts[9] = f"{path_target}_s{seed_idx}"

        return Path(*path_parts)

    def _process_point_cloud(
        self, tile_info: Dict[str, Any], img_size: int, transform: Any
    ) -> Tuple[np.ndarray, float]:
        """
        Loads, geo-references, scales, and samples the point cloud to a normalized local coordinate system.

        The transformation maps world coordinates $(x, y)$ to pixel-space $(col, row)$
        using the inverse affine transform, then scales them to a range of $[-1, 1]$.

        Parameters
        ----------
        tile_info : Dict[str, Any]
            Dictionary containing metadata and the path to the point cloud file.
        img_size : int
            The size of the square image tile (in pixels).
        transform : rasterio.affine.Affine
            The affine transform mapping pixel coordinates to world coordinates.

        Returns
        -------
        points : np.ndarray
            A float32 array of shape (N, 6) containing [row_scaled, col_scaled, z_scaled, R, G, B].
        xy_scale : float
            The average pixel size used to normalize the vertical (Z) dimension.

        Raises
        -------
        ValueError
            If pixel dimensions are non-square beyond tolerance or if the tile
            contains fewer points than `self.num_points`.
        """
        inv_transform = ~transform  # Inverse: World (X, Y) -> Pixel (Col, Row)

        # Validate pixel aspect ratio (ensure roughly square pixels)
        pixel_size_x, pixel_size_y = abs(transform.a), abs(transform.e)
        if not np.isclose(pixel_size_x, pixel_size_y, rtol=0.05):
            raise ValueError(f"Non-square pixels: {pixel_size_x} vs {pixel_size_y}")

        xy_scale = 0.5 * (pixel_size_x + pixel_size_y)

        if transform.b != 0 or transform.d != 0:
            warnings.warn(f"Rotation/Shear detected in affine for {tile_info['path']}")

        point_cloud_path = self._get_point_cloud_path(tile_info)

        with laspy.open(point_cloud_path) as point_cloud_file:
            las = point_cloud_file.read()

        # Extract coordinates and project to pixel space
        x, y, z = las.x, las.y, np.asarray(las.z, dtype=np.float32)
        cols, rows = inv_transform * (x, y)

        # Scale coordinates to [-1, 1] relative to the tile center
        half_size = img_size / 2.0
        rows_scaled = (rows - half_size) / half_size
        cols_scaled = (cols - half_size) / half_size

        # Normalize Z by mean and scale it proportionally to X/Y pixel units
        z_mean = np.mean(z) if z.size > 0 else 0
        z_scaled = (z - z_mean) / (xy_scale * half_size)

        # Normalize RGB (LAS standard is 16-bit)
        colors = (
            np.vstack((las.red, las.green, las.blue)).getfield(np.float32) / 65535.0
        )

        points = np.vstack(
            (rows_scaled, cols_scaled, z_scaled, colors), dtype=np.float32
        ).T

        # Filter points within bounds (using a small buffer for edge cases)
        valid_mask = (
            (rows >= -2.5)
            & (rows < img_size + 1.5)
            & (cols >= -2.5)
            & (cols < img_size + 1.5)
        )
        points = points[valid_mask]

        # Point sampling logic
        if self.num_points is not None:
            n_current = points.shape[0]
            if self.num_points < n_current:
                rng = np.random.default_rng()
                indices = rng.choice(n_current, self.num_points, replace=False)
                points = points[indices]
            elif self.num_points > n_current:
                raise ValueError(
                    f"Insufficient points: {n_current} < {self.num_points}"
                )

        return points, xy_scale

    def _get_dsm_data(
        self, tile_info: Dict[str, Any]
    ) -> Tuple[np.ndarray, float, float]:
        """
        Reads, normalizes, and scales the Digital Surface Model (DSM) raster.

        Parameters
        ----------
        tile_info : Dict[str, Any]
            Dictionary containing the 'dsm_path' key.

        Returns
        -------
        dsm_normalized : np.ndarray
            A float32 array normalized to the [0, 1] range.
        dsm_offset : float
            The minimum value of the original DSM (used for inversion/scaling).
        dsm_scale : float
            The range (max - min) of the original DSM.
        """
        with rasterio.open(tile_info["dsm_path"]) as dsm_file:
            dsm = dsm_file.read(1).astype(np.float32)

        # Min-Max Normalization while handling NaNs
        dsm_offset = np.nanmin(dsm)
        dsm_max = np.nanmax(dsm)
        dsm_scale = dsm_max - dsm_offset

        # Prevent division by zero for flat tiles
        if dsm_scale == 0:
            dsm_normalized = np.zeros_like(dsm)
        else:
            dsm_normalized = (dsm - dsm_offset) / dsm_scale

        return dsm_normalized, dsm_offset, dsm_scale

    def __getitem__(self, idx: int) -> Tuple[
        Dict[str, Any],
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Dict[str, Any],
    ]:
        """
        Retrieves a complete multimodal data sample including RGB image, point cloud, DSM, and associated targets for a
        specific index.

        Parameters
        ----------
        idx : int
            The index of the tile to retrieve from the dataset.

        Returns
        -------
        meta : Dict[str, Any]
            Metadata including filename, resolution, and normalization scales.
        img : np.ndarray
            The RGB image tile as a float32 array (C, H, W).
        points : np.ndarray, optional
            The sampled and normalized point cloud (N, 6) or None if not requested.
        dsm : np.ndarray, optional
            The normalized DSM tile (1, H, W) or None if not requested.
        targets : Dict[str, Any]
            Dictionary containing task-specific labels (class, height, segmentation).

        Raises
        -------
        IndexError
            If the index is out of bounds.
        ValueError
            If the tile contains no labels or unexpected data formats.
        NotImplementedError
            If a tile contains multiple labels (multi-label not supported).
        """
        tile_info = self.tiles[idx]

        # 1. Load Primary Image Data
        img, transform, resolution = self._get_image_data(tile_info)
        img_size = img.shape[1]

        # 2. Extract Labels and Safety Check
        labels = tile_info.get("labels", [])
        if not labels:
            raise ValueError(f"No labels found for tile: {tile_info['path']}")
        if len(labels) > 1:
            raise NotImplementedError("Multi-label tasks are not yet supported.")

        label = labels[0]

        # 3. Load Optional Modalities
        points, xy_scale = None, None
        if "point_cloud" in self.modalities:
            points, xy_scale = self._process_point_cloud(tile_info, img_size, transform)

        dsm, dsm_offset, dsm_scale = None, None, None
        if "dsm" in self.modalities:
            dsm, dsm_offset, dsm_scale = self._get_dsm_data(tile_info)

        # 4. Handle Segmentation Mask
        mask = None
        if "segmentation" in self.tasks:
            # decode_coco_segmentation expected to return (H, W)
            mask_raw = decode_coco_segmentation(label, "mask")
            mask = mask_raw[np.newaxis, :, :]  # Expand to (1, H, W)

        # 5. Apply Multimodal Augmentations
        if self.augment:
            img, points, dsm, mask = self.augment(img, points, dsm, mask)

        # 6. Prepare Targets
        targets = {}
        if "classification" in self.tasks:
            targets["class"] = label["category_id"]

        if "height" in self.tasks:
            other_attrs = self._get_other_attributes_to_pass(idx)
            targets["height"] = other_attrs[self.height_attr][0]

        if "segmentation" in self.tasks:
            targets["segmentation"] = mask

        # 7. Consolidate Metadata
        meta = {
            "filename": Path(tile_info["path"]).stem,
            "resolution": resolution,
            "xy_scale": float(xy_scale) if xy_scale is not None else None,
            "dsm_offset": float(dsm_offset) if dsm_offset is not None else None,
            "dsm_scale": float(dsm_scale) if dsm_scale is not None else None,
        }

        return meta, img, points, dsm, targets

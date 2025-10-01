import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import laspy
import numpy as np
import rasterio

from geodataset.dataset.base_dataset import BaseLabeledRasterCocoDataset


class ClassificationLabeledRasterPointCloudCocoDataset(BaseLabeledRasterCocoDataset):
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
        augment: Optional = None,
        num_points: Optional[int] = None,
    ):
        if len(tasks) != 1 or tasks[0] != "classification":
            raise ValueError(
                "This dataset only supports a single task: 'classification'"
            )
        else:
            other_attributes_names_to_pass = None
        super().__init__(
            fold=fold,
            root_path=root_path,
            transform=None,
            other_attributes_names_to_pass=other_attributes_names_to_pass,
        )
        self.dataset_name = dataset_name
        self.modalities = modalities
        self.augment = augment
        self.num_points = num_points

        # Add point cloud tile paths to tiles
        for tile in self.tiles.values():
            labels = tile["labels"]
            labels[0]["category_id"]
            if "point_cloud" in self.modalities:
                tile["point_cloud_path"] = self._derive_point_cloud_path(
                    tile["path"], voxel_size, num_points_file
                )
            if "dsm" in self.modalities:
                tile["dsm_path"] = self._derive_dsm_path(tile["path"])

    def derive_dsm_path(self, raster_path: Union[str, Path]) -> Path:
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
        else:
            raise ValueError(f"Unknown dataset {self.dataset_name}")

    def _derive_point_cloud_path(
        self, raster_path: Union[str, Path], voxel_size: float, num_points_file: int
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
        elif voxel_size is None and num_points_file is None:
            voxel_dir = ""
            voxel_token = ""
        else:
            raise ValueError(
                "Only one of voxel_size or num_points_file should be provided"
            )
        pcd_base = scene_root / scene_pcd_dir_name / f"pc_tiles{voxel_dir}" / fold_dir
        stem = p.stem  # original filename without .tif
        if self.dataset_name == "quebec_plantations":
            new_stem = stem.replace("_rgb_tile_", "_pg_pc_tile_", 1)
        elif self.dataset_name == "quebec_trees":
            new_stem = stem.replace("_rgb_tile_", "_pc_tile_", 1)
        if len(voxel_token) > 0:
            # Insert voxel-size tag after the sf… chunk (e.g., _sf1p0_)
            new_stem = re.sub(r"(_sf[^_]+_)", rf"\1{voxel_token}_", new_stem, count=1)
        new_name = new_stem + ".laz"
        pcd_path = pcd_base / new_name
        return pcd_path

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Retrieves a tile and its class label by index, applying transforms if specified.

        Parameters
        ----------
        idx: int
            The index of the tile to retrieve.

        Returns
        -------
        Tuple[Dict[str, Any], np.ndarray, np.ndarray, Dict[str, Any]]
            meta: dict
                Contains metadata about the tile, such as filename, z_offset, and xy_scale.
            image: np.ndarray
                Float array (C, H, W), RGB channels, values in [0, 1] after 255 scaling.
            point_cloud: np.ndarray
                Array of shape (N, 6) with columns:
                  0: row (image y index) – increases downward in the array. For a north-up geotransform this is geographic south (because transform.e < 0).
                  1: col (image x index) – increases to the right. For a north-up geotransform this is geographic east.
                  2: z_scaled – (original_z - per_tile_min_z) / xy_scale so minimum z becomes 0 and vertical units are expressed in “pixel-size” units (using the average of pixel_size_x and pixel_size_y when they are close; otherwise an error is raised).
                  3: r
                  4: g
                  5: b
                r,g,b are normalized to [0,1] from 16‑bit (divided by 65535).
                Note: If the affine has rotation/shear (b or d != 0) the (row, col) → world direction mapping is rotated accordingly, but (row, col) still index image space.
            targets: dict
                Contains 'labels' (int). Additional attributes may be included if requested.

        Coordinate / scaling notes
        --------------------------
        - Affine forward: x_world = a * col + c ; y_world = e * row + f (typically a > 0, e < 0 for north-up).
        - We invert the transform to map (x_world, y_world) → (col, row) for the LiDAR points, then reorder to (row, col) to match image indexing tile[:, row, col].
        - xy_scale = 0.5 * (|a| + |e|) when |a| and |e| are within 5% relative tolerance; else a ValueError is raised.
        - z is offset by its per-tile minimum so the lowest elevation in the tile becomes 0, then divided by xy_scale to express vertical distances in approximate pixel units.
        """
        tile_info = self.tiles[idx]

        with rasterio.open(tile_info["path"]) as tile_file:
            # Check if we have at least 3 bands (RGB)
            if tile_file.count >= 3:
                # Reading the first three bands
                img = tile_file.read([1, 2, 3])
            else:
                # Handle grayscale images or other band configurations
                img = tile_file.read()
                if img.shape[0] == 1:  # If single band, replicate to 3 channels
                    img = np.repeat(img, 3, axis=0)
            transform = tile_file.transform
        # Normalize the image data (handle 8-bit vs 16-bit)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            raise ValueError(
                f"Unexpected image dtype {img.dtype} in {tile_info['path']}"
            )

        if "point_cloud" in self.modalities:
            inv_transform = ~transform  # Inverse transform: world → pixel
            # Pixel sizes (world units per pixel)
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
            # Open the point cloud file
            point_cloud_path = tile_info["point_cloud_path"]
            with laspy.open(point_cloud_path) as point_cloud_file:
                las = point_cloud_file.read()
            x, y, z = las.x, las.y, las.z
            cols, rows = inv_transform * (x, y)
            # Offset z so minimum is 0, then scale into pixel units
            z_offset = z.min()
            z_zero = z - z_offset
            z_scaled = z_zero / xy_scale
            r, g, b = las.red, las.green, las.blue
            r, g, b = r / 65535.0, g / 65535.0, b / 65535.0  # Normalize RGB values
            points = np.vstack((rows, cols, z_scaled, r, g, b), dtype=np.float32).T
            H, W = img.shape[1:]  # tile is (C, H, W)
            valid_mask = (
                #    (rows >= -0.5) & (rows < H - 0.5) &
                #    (cols >= -0.5) & (cols < W - 0.5)
                # 2 pixels tolerance
                (rows >= -2.5)
                & (rows < H + 1.5)
                & (cols >= -2.5)
                & (cols < W + 1.5)
            )
            points = points[valid_mask]
            if self.num_points is not None:
                N = points.shape[0]
                if self.num_points < N:
                    indices = np.random.choice(
                        points.shape[0], self.num_points, replace=(N >= self.num_points)
                    )
                    points = points[indices]
                elif self.num_points > N:
                    raise ValueError(
                        f"Tile {tile_info['path']} has only {N} points, fewer than requested {self.num_points}"
                    )
        else:
            points = None

        if "dsm" in self.modalities:
            dsm_path = tile_info["dsm_path"]
            dsm = None
            with rasterio.open(dsm_path) as dsm_file:
                dsm = dsm_file.read()
            dsm = dsm.astype(np.float32)
            dsm_offset = np.nanmin(dsm)
            dsm_scale = np.nanmax(dsm) - dsm_offset
            dsm = (dsm - dsm_offset) / dsm_scale
        else:
            dsm = None

        # Apply transformations if specified
        if self.augment:
            img, points, dsm = self.augment(img, points, dsm)

        labels = tile_info["labels"]
        category_id = labels[0]["category_id"]

        targets = {
            "labels": category_id,
        }
        meta = {"filename": Path(tile_info["path"]).stem}
        if "point_cloud" in self.modalities:
            meta["z_offset"] = float(z_offset)
            meta["xy_scale"] = float(xy_scale)
        if "dsm" in self.modalities:
            meta["dsm_offset"] = float(dsm_offset)
            meta["dsm_scale"] = float(dsm_scale)

        return meta, img, points, dsm, targets

import sys
from pathlib import Path
from typing import List, Union

import geopandas as gpd
import laspy
from laspy import CopcReader

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from shapely import Polygon
from tqdm import tqdm

from geodataset.aoi import AOIFromPackageConfig
from geodataset.aoi.aoi_base import AOIBaseFromGeoFile
from geodataset.dataset.coco_generator import PointCloudCOCOGenerator
from geodataset.metadata.tile_metadata import TileMetadata, TileMetadataCollection
from geodataset.utils.file_name_conventions import (
    CocoNameConvention,
    PointCloudTileNameConvention,
)


class PointCloudTilerizer:
    """
    Class to tilerize point cloud data from a point cloud file and associated AOIs.

    Parameters
    ----------
    point_cloud_path : Path
        Path to the point cloud file.
    output_path : Path
        Path to the output folder where the tiles and annotations will be saved.
    tiles_metadata : Union[TileMetadataCollection, None], optional
        Metadata of the tiles, by default None
    aois_config : Union[AOIFromPackageConfig, None], optional
        Configuration for AOIs, by default None
    tile_side_length : float, optional
        Side length of the tile, by default None
    keep_dims : List[str], optional
        List of dimensions to keep, by default None
    downsample_voxel_size : float, optional
        Voxel size for downsampling, by default None
    verbose : bool, optional
        Verbose mode, by default False
    tile_overlap : float, optional
        Overlap between tiles, by default 1.0
    max_tile : int, optional
        Maximum number of tiles, by default 5000
    force : bool, optional
        Force tilerization, by default False
    """


    def __init__(
        self,
        point_cloud_path: Path,
        output_path: Path,
        tiles_metadata: Union[TileMetadataCollection, None] = None,
        aois_config: Union[AOIFromPackageConfig, None] = None,
        tile_side_length: float = None,
        keep_dims: List[str] = None,
        downsample_voxel_size: float = None,
        verbose: bool = False,
        tile_overlap: float = 0.5,
        max_tile: int = 5000,
        force: bool = False,
    ):
        assert not (
            tile_side_length and tiles_metadata
        ), "Only one of tile resolution or tile metadata should be provided"
        if tile_overlap and tiles_metadata:
            print("Tile overlap will be ignored, as tile metadata is also provided")

        self.point_cloud_path = point_cloud_path
        self.tiles_metadata = tiles_metadata
        self.output_path = Path(output_path)
        self.aoi_engine = AOIBaseFromGeoFile(aois_config)
        self.tile_side_length = tile_side_length
        self.tile_overlap = tile_overlap
        self.keep_dims = keep_dims if keep_dims is not None else "ALL"
        self.downsample_voxel_size = downsample_voxel_size
        self.verbose = verbose
        self.force = force

        if self.tiles_metadata is None:
            assert (
                self.tile_overlap is not None
            ), "Tile Overlap is required if tile metadata is not provided"

            self.populate_tiles_metadata()

        # Just a sanity check to make sure that the number of tiles is less than the max_tile
        assert (
            len(self.tiles_metadata) < max_tile
        ), f"Number of max possible tiles {len(self.tiles_metadata)} exceeds the maximum number of tiles {max_tile}"

        self.pc_tiles_folder_path = (
            self.output_path / f"pc_tiles_{self.downsample_voxel_size}"
            if self.downsample_voxel_size
            else self.output_path / "pc_tiles"
        )
        self.annotation_folder_path = self.output_path / "annotations"

        self.create_folder()

    def populate_tiles_metadata(
        self,
        suffix: str = None,
    ):
        name_convention = PointCloudTileNameConvention()

        product_name = self.point_cloud_path.stem.split(".")[0].replace("-", "_")
        
        if suffix:
            product_name = f"{product_name}_{suffix}"


        with CopcReader.open(self.point_cloud_path) as reader:
            min_x, min_y = reader.header.x_min, reader.header.y_min
            max_x, max_y = reader.header.x_max, reader.header.y_max

            tiles_metadata = []
            counter = 0
            for i, x in enumerate(
                np.arange(min_x, max_x, self.tile_side_length * self.tile_overlap)
            ):
                for j, y in enumerate(
                    np.arange(min_y, max_y, self.tile_side_length * self.tile_overlap)
                ):
                    x_bound = [x, x + self.tile_side_length]
                    y_bound = [y, y + self.tile_side_length]

                    tile_name = name_convention.create_name(
                        product_name=product_name, tile_id=f"{counter}", row=i, col=j
                    )

                    tile_md = TileMetadata(
                        id=f"{counter}",
                        x_bound=x_bound,
                        y_bound=y_bound,
                        crs=reader.header.parse_crs(),
                        tile_name=tile_name,
                    )

                    tiles_metadata.append(tile_md)
                    counter += 1

            self.tiles_metadata = TileMetadataCollection(
                tiles_metadata, product_name=product_name
            )

        print(f"Number of tiles generated: {len(self.tiles_metadata)}")

        if self.verbose:
            print(f"Point cloud x range: {min_x} - {max_x}")
            print(f"Point cloud y range: {min_y} - {max_y}")

        if not self.force:
            cont = str(input("Do you want to continue with the tilerization? (y/n)"))
            if cont.lower() != "y":
                print("Aborting tilerization")
                sys.exit()

    def create_folder(self):
        self.pc_tiles_folder_path.mkdir(parents=True, exist_ok=True)
        self.annotation_folder_path.mkdir(parents=True, exist_ok=True)

    def query_tile(self, tile_md, reader):
        mins = np.array([tile_md.min_x, tile_md.min_y])
        maxs = np.array([tile_md.max_x, tile_md.max_y])
        b = laspy.copc.Bounds(mins=mins, maxs=maxs)
        data = reader.query(b)

        return data

    def _tilerize(
        self,
    ):
        new_tile_md_list = []
        with CopcReader.open(self.point_cloud_path) as reader:
            for tile_md in tqdm(self.tiles_metadata):
                data = self.query_tile(tile_md, reader)

                if len(data) == 0:
                    continue

                pcd = self._laspy_to_o3d(data, self.keep_dims.copy())
                if self.downsample_voxel_size:
                    pcd = self._downsample_tile(pcd, self.downsample_voxel_size)
                pcd = self._keep_unique_points(pcd)
                new_tile_md_list.append(tile_md)
                downsampled_tile_path = (
                    self.pc_tiles_folder_path / f"{tile_md.tile_name}"
                )
                o3d.t.io.write_point_cloud(str(downsampled_tile_path), pcd)

        self.tiles_metadata = TileMetadataCollection(new_tile_md_list)

    def tilerize(
        self,
    ):
        self._generate_labels()
        self._tilerize()
        self.plot_aois()

    def _get_aoi_tiles(self):
        aois_gdf = self.aoi_engine.get_aoi_gdf()
        # //NOTE -  Cannot check for color as data is not provided here and only metadata is provided.

        tiles_gdf = self.tiles_metadata.gdf
        aois_gdf.crs = tiles_gdf.crs
        intersections = gpd.overlay(tiles_gdf, aois_gdf, how="intersection")
        intersections["intersection_area"] = intersections.geometry.area
        max_intersection_per_tile = intersections.loc[
            intersections.groupby("tile_id")["intersection_area"].idxmax()
        ]
        aois_tiles = (
            max_intersection_per_tile.groupby("aoi")["tile_id"].apply(list).to_dict()
        )

        return aois_tiles

    def _generate_labels(self):
        aoi_tiles = self._get_aoi_tiles()
        coco_paths = {}

        for aoi in aoi_tiles:
            tiles_metadata = TileMetadataCollection(
                [tile for tile in self.tiles_metadata if tile.id in aoi_tiles[aoi]]
            )

            print(self.tiles_metadata.product_name)
            coco_output_file_path = (
                self.annotation_folder_path
                / CocoNameConvention.create_name(
                    product_name=self.tiles_metadata.product_name,
                    ground_resolution=None,
                    scale_factor=None,
                    fold=aoi,
                )
            )

            polygons = [
                [Polygon()] for _ in tiles_metadata
            ]  # Passing empty polygons as we are not interested in the polygons for now.

            coco_generator = PointCloudCOCOGenerator(
                description="Dataset for the product XYZ",
                tiles_metadata=tiles_metadata,
                polygons=polygons,
                output_path=coco_output_file_path,
                scores=None,
                categories=None,
                other_attributes=None,
                use_rle_for_labels=False,
                n_workers=1,
                coco_categories_list=None,
            )

            coco_generator.generate_coco()
            coco_paths[aoi] = coco_output_file_path

        return coco_paths

    def plot_aois(self) -> None:
        fig, ax = self.tiles_metadata.plot()

        palette = {
            "blue": "#26547C",
            "red": "#EF476F",
            "yellow": "#FFD166",
            "green": "#06D6A0",
        }

        self.aoi_engine.loaded_aois["train"].plot(
            ax=ax, color=palette["blue"], alpha=0.5
        )
        self.aoi_engine.loaded_aois["valid"].plot(
            ax=ax, color=palette["green"], alpha=0.5
        )
        self.aoi_engine.loaded_aois["test"].plot(ax=ax, color=palette["red"], alpha=0.5)

        plt.savefig(self.output_path / f"{self.tiles_metadata.product_name}_aois.png")

    def _laspy_to_o3d(self, pc_file: Path, keep_dims: List[str]):
        dimensions = list(pc_file.point_format.dimension_names)

        if keep_dims == "ALL":
            keep_dims = dimensions

        assert all([dim in keep_dims for dim in ["X", "Y", "Z"]])
        assert all([dim in dimensions for dim in keep_dims])

        pc = np.ascontiguousarray(
            np.vstack([pc_file.x, pc_file.y, pc_file.z]).T.astype(np.float64)
        )

        map_to_tensors = {}
        map_to_tensors["positions"] = pc.astype(np.float64)

        keep_dims.remove("X")
        keep_dims.remove("Y")
        keep_dims.remove("Z")

        if "red" in keep_dims and "green" in keep_dims and "blue" in keep_dims:
            pc_colors = (
                np.ascontiguousarray(
                    np.vstack([pc_file.red, pc_file.green, pc_file.blue]).T.astype(
                        np.float64
                    )
                )
                / 255
            )
            map_to_tensors["colors"] = pc_colors.astype(np.float64)

            keep_dims.remove("red")
            keep_dims.remove("blue")
            keep_dims.remove("green")

        for dim in keep_dims:
            dim_value = np.ascontiguousarray(pc_file[dim]).astype(np.float64)

            assert len(dim_value.shape) == 1
            dim_value = dim_value.reshape(-1, 1)
            map_to_tensors[dim] = dim_value

        pcd = o3d.t.geometry.PointCloud(map_to_tensors)

        return pcd

    def _remove_black_points(self, tile, tile_size):
        is_rgb = self.raster.data.shape[0] == 3
        is_rgba = self.raster.data.shape[0] == 4
        skip_ratio = self.ignore_black_white_alpha_tiles_threshold

        # Checking if the tile has more than a certain ratio of white, black, or alpha pixels.
        if is_rgb:
            if np.sum(tile.data == 0) / (tile_size * tile_size * 3) > skip_ratio:
                return True
            if np.sum(tile.data == 255) / (tile_size * tile_size * 3) > skip_ratio:
                return True
        elif is_rgba:
            if np.sum(tile.data[:-1] == 0) / (tile_size * tile_size * 3) > skip_ratio:
                return True
            if np.sum(tile.data[:-1] == 255) / (tile_size * tile_size * 3) > skip_ratio:
                return True
            if np.sum(tile.data[-1] == 0) / (tile_size * tile_size * 3) > skip_ratio:
                return True

        return False

    def _keep_unique_points(self, pcd):
        map_to_tensors = {}
        positions = pcd.point.positions.numpy()
        unique_pc, ind = np.unique(positions, axis=0, return_index=True)

        map_to_tensors["positions"] = unique_pc

        for attr in pcd.point:
            if attr != "positions":
                map_to_tensors[attr] = getattr(pcd.point, attr)[ind]

        if self.verbose:
            n_removed = positions.shape[0] - unique_pc.shape[0]
            percentage_removed = (n_removed / positions.shape[0]) * 100
            print(f"Removed {n_removed} duplicate points ({percentage_removed:.2f}%)")
        return o3d.t.geometry.PointCloud(map_to_tensors)

    def _downsample_tile(self, pcd, voxel_size) -> None:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        return pcd

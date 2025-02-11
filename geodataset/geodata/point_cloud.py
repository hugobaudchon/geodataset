from pathlib import Path
from typing import Union, Tuple, List

import geopandas as gpd
import numpy as np
import rasterio
from matplotlib import pyplot as plt
from pyproj import CRS as PyProjCRS
from shapely import box

from geodataset.utils import find_tiles_paths, TileNameConvention, strip_all_extensions_and_path
from geodataset.utils.file_name_conventions import PointCloudTileNameConvention, validate_and_convert_product_name
import pandas as pd


class PointCloudTileMetadata:
    """
    Represents metadata for a tile, Generally used for PointCloudTile.
    The tiles are supposed to be rectangular.

    Parameters
    ----------
    tile_id: int
        The unique id of the tile.
    x_bound: Tuple[float, float], optional
        The x-axis bounds of the tile.
    y_bound: Tuple[float, float], optional
        The y-axis bounds of the tile.
    z_bound: Tuple[float, float], optional
        The z-axis bounds of the tile.
    crs: PyProjCRS, optional
        The coordinate reference system of the tile.
    output_filename: str, optional
        The output filename of the tile.
    """

    def __init__(
        self,
        crs: PyProjCRS,
        tile_id: int,
        tile_name: str,
        x_bound: Union[Tuple[float, float], None] = None,
        y_bound: Union[Tuple[float, float], None] = None,
        z_bound: Union[Tuple[float, float], None] = None,

        height: int = None,
        width: int = None,
        aoi: str = None
    ) -> None:
        """
        Initializes a TileMetadata object.
        """
        __slots__ = ( # noqa F841

            "min_x",
            "max_x",
            "min_y",
            "max_y",
            "min_z",
            "max_z",
            "crs",
            "id",
            "filename",
            "geometry",
            "height",
            "width",
        )

        self.min_x, self.max_x = x_bound if x_bound else (None, None)
        self.min_y, self.max_y = y_bound if y_bound else (None, None)
        self.min_z, self.max_z = z_bound if z_bound else (None, None)

        self.crs = crs
        assert PointCloudTileNameConvention._validate_name(
            tile_name
        ), f"Invalid output_filename: {tile_name}"

        self.tile_name = tile_name
        self.tile_id = tile_id

        self.geometry = self._get_bounding_box()

        # Height and width are important to generate coco dataset
        # NOTE: This is important for generating bounding boxes for the tiles

        self.height = height
        self.width = width
        self.aoi = aoi


    def __repr__(self) -> str:
        return self.info()

    def info(self) -> str:
        """
        Returns a string representation of the tile metadata.
        """
        return (
            f"(min_x, max_x): ({self.min_x}, {self.max_x})\n"
            f"(min_y, max_y): ({self.min_y}, {self.max_y})\n"
            f"(min_z, max_z): ({self.min_z}, {self.max_z})\n"
            f"crs: {self.crs.name}\n"
            f"output_filename: {self.tile_name}\n"
        )

    def _bounded(self, bound: Tuple[float, float]) -> bool:
        """
        Returns whether the given bound is bounded.
        """
        return None not in bound

    def is_bounded_x(self) -> bool:
        """
        Returns whether the x-axis is bounded.
        """
        return self._bounded((self.min_x, self.max_x))

    def is_bounded_y(self) -> bool:
        """
        Returns whether the y-axis is bounded.
        """
        return self._bounded((self.min_y, self.max_y))

    def is_bounded_z(self) -> bool:
        """
        Returns whether the z-axis is bounded.
        """
        return self._bounded((self.min_z, self.max_z))

    def _get_bounding_box(self,):
        bounding_box = box(self.min_x, self.min_y, self.max_x, self.max_y )
        return gpd.GeoDataFrame(index=[0], crs= self.crs, geometry=[bounding_box])


class PointCloudTileMetadataCollection:
    """
    Represents metadata for a collection of point cloud tiles.

    Parameters
    ----------
    tile_metadata_list: List[geodataset.geodata.point_cloud.PointCloudTileMetadata]
        A list of tile metadata.
    """

    def __init__(self, tile_metadata_list: List[PointCloudTileMetadata], product_name=Union[str, None]):
        self.product_name = product_name
        self.tile_metadata_list = tile_metadata_list
        self.downsample_voxel_size = None

        self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z = (
            self._calculate_bounds()
        )

        self.unique_x_bounds = (
            sorted(set((t.min_x, t.max_x) for t in tile_metadata_list))
            if self.is_bounded_x()
            else np.inf
        )
        self.unique_y_bounds = (
            sorted(set((t.min_y, t.max_y) for t in tile_metadata_list))
            if self.is_bounded_y()
            else np.inf
        )
        self.unique_z_bounds = (
            sorted(set((t.min_z, t.max_z) for t in tile_metadata_list))
            if self.is_bounded_z()
            else np.inf
        )

        self.unique_xy_bounds = (
            sorted(
                set((t.min_x, t.max_x, t.min_y, t.max_y) for t in tile_metadata_list)
            )
            if self.is_bounded_x() and self.is_bounded_y()
            else np.inf
        )
        self.unique_xz_bounds = (
            sorted(
                set((t.min_x, t.max_x, t.min_z, t.max_z) for t in tile_metadata_list)
            )
            if self.is_bounded_x() and self.is_bounded_z()
            else np.inf
        )
        self.unique_yz_bounds = (
            sorted(
                set((t.min_y, t.max_y, t.min_z, t.max_z) for t in tile_metadata_list)
            )
            if self.is_bounded_y() and self.is_bounded_z()
            else np.inf
        )

        assert all([True if i.crs == self.tile_metadata_list[0].crs else False for i in self.tile_metadata_list]), "All tiles must have the same CRS"
        self.crs = self.tile_metadata_list[0].crs
        assert all([True if i.height == self.tile_metadata_list[0].height else False for i in self.tile_metadata_list]), "All tiles must have the same height"
        self.height = self.tile_metadata_list[0].height
        assert all([True if i.width == self.tile_metadata_list[0].width else False for i in self.tile_metadata_list]), "All tiles must have the same width"
        self.width = self.tile_metadata_list[0].width

        self.gdf = self._create_gdf()
        self.aois = self.gdf.aoi.unique()
        self.tile_id_to_idx = {tile_id: idx for idx, tile_id in enumerate([tile.tile_id for tile in self.tile_metadata_list])}

    @classmethod
    def from_tiles_folder(cls,
                          root_tiles_folder: Union[str, Path],
                          point_cloud_path: Union[str, Path],
                          downsample_voxel_size: float):
        tiles_folder = Path(root_tiles_folder)
        point_cloud_path = Path(point_cloud_path)
        tiles_paths = find_tiles_paths([tiles_folder], extensions=['tif'])
        pc_product_name = validate_and_convert_product_name(strip_all_extensions_and_path(point_cloud_path))
        tiles_metadata = []
        tile_id = 0
        for tile_name in tiles_paths.keys():
            tile_path = tiles_paths[tile_name]
            if ".tif" in str(tile_path):
                (tile_product_name, scale_factor, ground_resolution,
                 col, row, aoi) = TileNameConvention.parse_name(tile_path.name)

                tile_name = PointCloudTileNameConvention.create_name(
                    product_name=pc_product_name, row=row, col=col, aoi=aoi, voxel_size=downsample_voxel_size
                )
                data = rasterio.open(tile_path)
                x_bound = (data.bounds[0], data.bounds[2])
                y_bound = (data.bounds[1], data.bounds[3])
                crs = PyProjCRS.from_user_input(data.crs)
                tile_md = PointCloudTileMetadata(
                    x_bound=x_bound,
                    y_bound=y_bound,
                    crs=crs,
                    tile_name=tile_name,
                    tile_id=tile_id,
                    width=data.width,
                    height=data.height,
                    aoi=aoi
                )

                tiles_metadata.append(tile_md)
                tile_id += 1

        return cls(tiles_metadata, product_name=pc_product_name)

    def _create_gdf(self,):
        return gpd.GeoDataFrame(data={"tile_id":[tile.tile_id for tile in self.tile_metadata_list],
                                      "aoi": [tile.aoi for tile in self.tile_metadata_list]},
                                geometry=[tile.geometry.values[0][0] for tile in self.tile_metadata_list],
                                crs=self.crs)

    def __getitem__(self, idx: int) -> PointCloudTileMetadata:
        return self.tile_metadata_list[idx]

    def __len__(self) -> int:
        return len(self.tile_metadata_list)

    def _find_nonemin(self, a: Union[float, None], b: float) -> float:
        return np.nanmin([a, b]) if a is not None else b

    def _find_nonemax(self, a: Union[float, None], b: float) -> float:
        return np.nanmax([a, b]) if a is not None else b

    def _calculate_bounds(self) -> Tuple[float, float, float, float, float, float]:
        min_x, max_x = np.inf, -np.inf
        min_y, max_y = np.inf, -np.inf
        min_z, max_z = np.inf, -np.inf

        for tile in self.tile_metadata_list:
            min_x = self._find_nonemin(tile.min_x, min_x)
            max_x = self._find_nonemax(tile.max_x, max_x)
            min_y = self._find_nonemin(tile.min_y, min_y)
            max_y = self._find_nonemax(tile.max_y, max_y)
            min_z = self._find_nonemin(tile.min_z, min_z)
            max_z = self._find_nonemax(tile.max_z, max_z)

        return min_x, max_x, min_y, max_y, min_z, max_z

    def is_bounded_x(self) -> bool:
        """
        Returns whether the x-axis is bounded.
        """
        return all(tile.is_bounded_x() for tile in self.tile_metadata_list)

    def is_bounded_y(self) -> bool:
        """
        Returns whether the y-axis is bounded.
        """
        return all(tile.is_bounded_y() for tile in self.tile_metadata_list)

    def is_bounded_z(self) -> bool:
        """
        Returns whether the z-axis is bounded.
        """
        return all(tile.is_bounded_z() for tile in self.tile_metadata_list)

    def plot(self, dim1: str = "x", dim2: str = "y", ) -> plt.Axes:
        """
        Plots the tiles and their boundaries.
        """
        min_dim1, max_dim1 = getattr(self, f"min_{dim1}"), getattr(self, f"max_{dim1}")
        min_dim2, max_dim2 = getattr(self, f"min_{dim2}"), getattr(self, f"max_{dim2}")

        # palette = {
        #     "blue": "#26547C",
        #     "red": "#EF476F",
        #     "yellow": "#FFD166",
        #     "green": "#06D6A0",
        # }
        fig, ax = plt.subplots()
        bound_rect = plt.Rectangle(
            (min_dim1, min_dim2),
            max_dim1 - min_dim1,
            max_dim2 - min_dim2,
            edgecolor="k",
            facecolor="w",
        )
        ax.add_patch(bound_rect)

        pad = 0.05
        ax.set_xlim(
            min_dim1 - pad * (max_dim1 - min_dim1),
            max_dim1 + pad * (max_dim1 - min_dim1),
        )
        ax.set_ylim(
            min_dim2 - pad * (max_dim2 - min_dim2),
            max_dim2 + pad * (max_dim2 - min_dim2),
        )

        for tile in self.tile_metadata_list:

            color = "white"


            tile_min_dim1, tile_max_dim1 = getattr(tile, f"min_{dim1}"), getattr(
                tile, f"max_{dim1}"
            )
            tile_min_dim2, tile_max_dim2 = getattr(tile, f"min_{dim2}"), getattr(
                tile, f"max_{dim2}"
            )

            tile_patch = plt.Rectangle(
                (tile_min_dim1, tile_min_dim2),
                tile_max_dim1 - tile_min_dim1,
                tile_max_dim2 - tile_min_dim2,
                edgecolor="k",
                facecolor=color,
                alpha=0.2,
            )
            ax.add_patch(tile_patch)

        random_tile = np.random.choice(self.tile_metadata_list)
        random_tile_min_dim1, random_tile_max_dim1 = getattr(
            random_tile, f"min_{dim1}"
        ), getattr(random_tile, f"max_{dim1}")
        random_tile_min_dim2, random_tile_max_dim2 = getattr(
            random_tile, f"min_{dim2}"
        ), getattr(random_tile, f"max_{dim2}")

        rand_patch = plt.Rectangle(
            (random_tile_min_dim1, random_tile_min_dim2),
            random_tile_max_dim1 - random_tile_min_dim1,
            random_tile_max_dim2 - random_tile_min_dim2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rand_patch)

        ax.set_xlabel(dim1)
        ax.set_ylabel(dim2)
        ax.set_title("Tiles with train, test, val split (random tile highlighted)")

        return fig, ax

    def get_tile_by_id(self, tile_id: str) -> PointCloudTileMetadata:
        """
        Returns the tile with the given id.
        """

        idx = self.tile_id_to_idx[tile_id]

        return self.tile_metadata_list[idx]

    def info(self) -> str:
        """
        Returns a string representation of the tile metadata.
        """
        return (
            f"min_x: {self.min_x}\n"
            f"max_x: {self.max_x}\n"
            f"min_y: {self.min_y}\n"
            f"max_y: {self.max_y}\n"
            f"min_z: {self.min_z}\n"
            f"max_z: {self.max_z}\n"
            f"n_tiles: {len(self.tile_metadata_list)}\n"
            f"num_unique_x_bounds: {len(self.unique_x_bounds)}\n"
            f"num_unique_y_bounds: {len(self.unique_y_bounds)}\n"
            f"num_unique_z_bounds: {len(self.unique_z_bounds)}\n"
            f"num_unique_xy_bounds: {len(self.unique_xy_bounds)}\n"
            f"num_unique_xz_bounds: {len(self.unique_xz_bounds)}\n"
            f"num_unique_yz_bounds: {len(self.unique_yz_bounds)}\n"
        )

    def save(self, output_folder, downsample_voxel_size):
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        tile_name_list = []
        min_x_list = []
        max_x_list = []
        min_y_list = []
        max_y_list = []
        crs_list = []

        for tile in self.tile_metadata_list:
            tile_name_list.append(tile.tile_name)
            min_x_list.append(tile.min_x)
            max_x_list.append(tile.max_x)
            min_y_list.append(tile.min_y)
            max_y_list.append(tile.max_y)
            crs_list.append(tile.crs)
        
        df = pd.DataFrame()
        df["tile_name"] = tile_name_list
        df["min_x"] = min_x_list
        df["max_x"] = max_x_list
        df["min_y"] = min_y_list
        df["max_y"] = max_y_list
        df["crs"] = crs_list
        
        file_path = output_folder / f"pc_tiles_{downsample_voxel_size}" 
        file_path.mkdir(parents=True, exist_ok=True)
        file_name = file_path / f"{self.product_name}_tile_metadata.csv"
        
        df.to_csv(file_name, index=False)

        print(f"Tile metadata written to {file_path}")

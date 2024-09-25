from pyproj import CRS as PyProjCRS
from typing import List, Tuple, Union
from geodataset.utils.file_name_conventions import PointCloudTileNameConvention
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box
import geopandas as gpd

class TileMetadata:
    """
    Represents metadata for a tile, Generally used for PointCloudTile.
    The tiles are supposed to be rectangular.

    Parameters
    ----------
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
        id: Union[int, str, None],
        x_bound: Union[Tuple[float, float], None] = None,
        y_bound: Union[Tuple[float, float], None] = None,
        z_bound: Union[Tuple[float, float], None] = None,
        crs: PyProjCRS = None,
        output_filename: Union[str, None] = None,
        
        height: int = None,
        width: int = None,

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
            "output_filename",
            "geometry",
            "height",
            "width",
        )

        self.min_x, self.max_x = x_bound if x_bound else (None, None)
        self.min_y, self.max_y = y_bound if y_bound else (None, None)
        self.min_z, self.max_z = z_bound if z_bound else (None, None)

        self.crs = crs
        #TODO: Add validation for output_filename
        # assert PointCloudTileNameConvention._validate_name(
        #     output_filename
        # ), f"Invalid output_filename: {output_filename}"

        self.output_filename = output_filename
        self.id = id

        self.geometry = self._get_bounding_box()
        
        # Height and width are important to generate coco dataset
        # NOTE: This is important for generating bounding boxes for the tiles
         
        self.height = height
        self.width = width

        
    
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
            f"output_filename: {self.output_filename}\n"
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



class TileMetadataCollection:
    """
    Represents metadata for a collection of point cloud tiles.

    Parameters
    ----------
    tile_metadata_list: List[TileMetadata]
        A list of tile metadata.
    """

    def __init__(self, tile_metadata_list: List[TileMetadata], product_name=Union[str, None]):
        self.product_name = product_name
        self.tile_metadata_list = tile_metadata_list
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
        self.gdf = self._create_gdf()
        self.tile_id_to_idx = {tile_id: idx for idx, tile_id in enumerate([tile.id for tile in self.tile_metadata_list])}

    def _create_gdf(self,):
        return gpd.GeoDataFrame(data={"tile_id":[tile.id for tile in self.tile_metadata_list]}, geometry=[tile.geometry.values[0][0] for tile in self.tile_metadata_list], crs=self.crs)
    
    def __getitem__(self, idx: int) -> TileMetadata:
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

        palette = {
            "blue": "#26547C",
            "red": "#EF476F",
            "yellow": "#FFD166",
            "green": "#06D6A0",
        }
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
    
    def get_tile_by_id(self, id: int) -> Tuple[float, float, float, float]:
        """
        Returns the tile with the given id.
        """

        idx = self.tile_id_to_idx[id]

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

        
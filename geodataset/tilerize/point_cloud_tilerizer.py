from typing import List, Tuple, Union
import numpy as np
import laspy
from pyproj import CRS as PyProjCRS
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from geodataset.aoi import AOIConfig


class TileMetadata:
    """
    Represents metadata for an individual tile in a point cloud dataset.
    """
    
    def __init__(self, x_bound: Union[Tuple[float, float], None] = None, y_bound: Union[Tuple[float, float], None] = None, z_bound: Union[Tuple[float, float], None] = None, crs: PyProjCRS = None, tile_id: Union[str, None] = None) -> None:
        """
        Initializes a TileMetadata object.
        """
        __slots__ = "min_x", "max_x", "min_y", "max_y", "min_z", "max_z", "crs", "tile_id"

        self.min_x, self.max_x = x_bound if x_bound else (None, None)
        self.min_y, self.max_y = y_bound if y_bound else (None, None)
        self.min_z, self.max_z = z_bound if z_bound else (None, None)

        self.crs = crs
        self.tile_id = tile_id

    def __repr__(self) -> str:
        return self.info()

    def info(self) -> str:
        """
        Returns a string representation of the tile metadata.
        """
        return f"(min_x, max_x): ({self.min_x}, {self.max_x})\n" \
               f"(min_y, max_y): ({self.min_y}, {self.max_y})\n" \
               f"(min_z, max_z): ({self.min_z}, {self.max_z})\n" \
               f"crs: {self.crs.name}\n" \
               f"tile_id: {self.tile_id}\n"

    def _bounded(self, bound: Tuple[float, float]) -> bool:
        return None not in bound

    def is_bounded_x(self) -> bool:
        return self._bounded((self.min_x, self.max_x))
    
    def is_bounded_y(self) -> bool:
        return self._bounded((self.min_y, self.max_y))
    
    def is_bounded_z(self) -> bool:
        return self._bounded((self.min_z, self.max_z))


class TileCollectionMetadata:
    """
    Represents metadata for a collection of point cloud tiles.
    """
    
    def __init__(self, tile_metadata_list: List[TileMetadata]):
        self.tile_metadata_list = tile_metadata_list
        self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z = self._calculate_bounds()

        self.unique_x_bounds = sorted(set((t.min_x, t.max_x) for t in tile_metadata_list)) if self.is_bounded_x() else np.inf
        self.unique_y_bounds = sorted(set((t.min_y, t.max_y) for t in tile_metadata_list)) if self.is_bounded_y() else np.inf
        self.unique_z_bounds = sorted(set((t.min_z, t.max_z) for t in tile_metadata_list)) if self.is_bounded_z() else np.inf

        self.unique_xy_bounds = sorted(set((t.min_x, t.max_x, t.min_y, t.max_y) for t in tile_metadata_list)) if self.is_bounded_x() and self.is_bounded_y() else np.inf
        self.unique_xz_bounds = sorted(set((t.min_x, t.max_x, t.min_z, t.max_z) for t in tile_metadata_list)) if self.is_bounded_x() and self.is_bounded_z() else np.inf
        self.unique_yz_bounds = sorted(set((t.min_y, t.max_y, t.min_z, t.max_z) for t in tile_metadata_list)) if self.is_bounded_y() and self.is_bounded_z() else np.inf

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
        return all(tile.is_bounded_x() for tile in self.tile_metadata_list)

    def is_bounded_y(self) -> bool:
        return all(tile.is_bounded_y() for tile in self.tile_metadata_list)

    def is_bounded_z(self) -> bool:
        return all(tile.is_bounded_z() for tile in self.tile_metadata_list)

    def plot(self, dim1: str = "x", dim2: str = "y") -> plt.Axes:
        """
        Plots the tiles and their boundaries.
        """
        min_dim1, max_dim1 = getattr(self, f"min_{dim1}"), getattr(self, f"max_{dim1}")
        min_dim2, max_dim2 = getattr(self, f"min_{dim2}"), getattr(self, f"max_{dim2}")

        fig, ax = plt.subplots()
        bound_rect = plt.Rectangle((min_dim1, min_dim2), max_dim1 - min_dim1, max_dim2 - min_dim2, edgecolor="k", facecolor="w")
        ax.add_patch(bound_rect)

        pad = 0.05
        ax.set_xlim(min_dim1 - pad * (max_dim1 - min_dim1), max_dim1 + pad * (max_dim1 - min_dim1))
        ax.set_ylim(min_dim2 - pad * (max_dim2 - min_dim2), max_dim2 + pad * (max_dim2 - min_dim2))

        for tile in self.tile_metadata_list:
            tile_min_dim1, tile_max_dim1 = getattr(tile, f"min_{dim1}"), getattr(tile, f"max_{dim1}")
            tile_min_dim2, tile_max_dim2 = getattr(tile, f"min_{dim2}"), getattr(tile, f"max_{dim2}")

            tile_patch = plt.Rectangle((tile_min_dim1, tile_min_dim2), tile_max_dim1 - tile_min_dim1, tile_max_dim2 - tile_min_dim2, edgecolor="k", facecolor="k", alpha=0.1)
            ax.add_patch(tile_patch)

        random_tile = np.random.choice(self.tile_metadata_list)
        random_tile_min_dim1, random_tile_max_dim1 = getattr(random_tile, f"min_{dim1}"), getattr(random_tile, f"max_{dim1}")
        random_tile_min_dim2, random_tile_max_dim2 = getattr(random_tile, f"min_{dim2}"), getattr(random_tile, f"max_{dim2}")

        rand_patch = plt.Rectangle((random_tile_min_dim1, random_tile_min_dim2), random_tile_max_dim1 - random_tile_min_dim1, random_tile_max_dim2 - random_tile_min_dim2, edgecolor="r", facecolor="none")
        ax.add_patch(rand_patch)
        
        ax.set_xlabel(dim1)
        ax.set_ylabel(dim2)
        ax.set_title("All tiles with a randomly highlighted tile")
        
        return ax

    def info(self) -> str:
        """
        Returns a string representation of the tile metadata.
        """
        return f"min_x: {self.min_x}\n" \
               f"max_x: {self.max_x}\n" \
               f"min_y: {self.min_y}\n" \
               f"max_y: {self.max_y}\n" \
               f"min_z: {self.min_z}\n" \
               f"max_z: {self.max_z}\n" \
               f"n_tiles: {len(self.tile_metadata_list)}\n" \
               f"num_unique_x_bounds: {len(self.unique_x_bounds)}\n" \
               f"num_unique_y_bounds: {len(self.unique_y_bounds)}\n" \
               f"num_unique_z_bounds: {len(self.unique_z_bounds)}\n" \
               f"num_unique_xy_bounds: {len(self.unique_xy_bounds)}\n" \
               f"num_unique_xz_bounds: {len(self.unique_xz_bounds)}\n" \
               f"num_unique_yz_bounds: {len(self.unique_yz_bounds)}\n"


class PointCloudTiles:
    """
    Represents a collection of point cloud tiles data.
    """
    
    def __init__(self, n_tiles: int, header: laspy.LasHeader):
        self.n_tiles = n_tiles
        self.data = [None] * n_tiles
        self.header = header

    def append(self, index: int, data: laspy.ScaleAwarePointRecord) -> None:
        """
        Appends the given data to the specified index in the point cloud tilerizer.
        """
        if self.data[index] is not None:
            self.data[index] = laspy.ScaleAwarePointRecord(
                np.concatenate([self.data[index].array, data.array]), 
                point_format=self.header.point_format, 
                scales=self.header.scales, 
                offsets=self.header.offsets
            )
        else:    
            self.data[index] = data

    def __getitem__(self, index: int) -> laspy.ScaleAwarePointRecord:
        return self.data[index]

    def clear_data(self) -> None:
        """
        Clears the data stored in the point cloud tilerizer.
        """
        self.data = [None] * self.n_tiles


class PointCloudTiler:
    """
    A class for tilerizing point cloud data.
    """
    
    def __init__(self, point_cloud_path: Path, output_folder_path: Path, tiles_metadata: List[TileCollectionMetadata], aois_config: AOIConfig = None):
        self.point_cloud_path = point_cloud_path
        self.tiles_metadata = tiles_metadata
        self.output_folder_path = Path(output_folder_path)

    def lazy_tilerize(self, chunk_size: int = 500_000) -> None:
        """
        Tilerizes the point cloud data lazily.
        """
        self.xy_to_tile_index = self._get_xy_to_tile_index()

        for tile_md in self.tiles_metadata:
            file_path = self.output_folder_path / f"{tile_md.tile_id}.las"
            if file_path.is_file():
                raise FileExistsError(f"{tile_md.tile_id} exists at {str(self.output_folder_path)}. Update tile_id in metadata or remove existing files.") 

        with laspy.open(self.point_cloud_path) as f:
            self.tile_data = PointCloudTiles(n_tiles=len(self.tiles_metadata), header=f.header)
            with tqdm(total=-(-f.header.point_records_count // chunk_size)) as pbar:  # ceil divide -(a // -b)
                for chunked_points in f.chunk_iterator(chunk_size):
                    self._bin_chunk_points(chunked_points)
                    self._lazy_write()
                    pbar.update(1)

    def _lazy_write(self) -> None:
        """
        Lazily writes the tile data to LAS files.
        """
        for i, point_cloud_tile_data in enumerate(self.tile_data):
            if point_cloud_tile_data is not None:
                file_path = self.output_folder_path / f"{self.tiles_metadata[i].tile_id}.las"
                
                if file_path.is_file():
                    with laspy.open(file_path, mode="a") as writer:
                        writer.append_points(point_cloud_tile_data)
                else:
                    new_header = laspy.LasHeader(version=self.tile_data.header.version, point_format=self.tile_data.header.point_format)
                    new_header.offsets = self.tile_data.header.offsets
                    new_header.scales = self.tile_data.header.scales

                    with laspy.open(file_path, mode="w", header=new_header) as writer:
                        writer.write_points(point_cloud_tile_data)

        self.tile_data.clear_data()

    def _bin_chunk_points(self, chunked_data: laspy.ScaleAwarePointRecord) -> None:
        """
        Bins the chunked points into the appropriate tiles.
        """
        for i, x_b in enumerate(self.tiles_metadata.unique_x_bounds):
            index = (x_b[0] < chunked_data.x) & (chunked_data.x <= x_b[1])
            subset_px = chunked_data[index]
            if subset_px:
                for j, y_b in enumerate(self.tiles_metadata.unique_y_bounds):
                    if (x_b + y_b) in self.tiles_metadata.unique_xy_bounds:
                        index = (y_b[0] < subset_px.y) & (subset_px.y <= y_b[1])
                        subset_pxy = subset_px[index]
                        if (i, j) in self.xy_to_tile_index:
                            file_index = self.xy_to_tile_index[(i, j)]
                            if subset_pxy:
                                self.tile_data.append(file_index, subset_pxy)
    
    def _get_xy_to_tile_index(self) -> dict:
        """
        Returns a dictionary mapping (x, y) coordinates to tile indexes.
        """
        xy_to_tile = {}
        for i, x_bound in enumerate(self.tiles_metadata.unique_x_bounds):
            for j, y_bound in enumerate(self.tiles_metadata.unique_y_bounds):
                for k, pt_md in enumerate(self.tiles_metadata):
                    if (pt_md.min_x == x_bound[0]) and (pt_md.max_x == x_bound[1]) and (pt_md.min_y == y_bound[0]) and (pt_md.max_y == y_bound[1]):
                        xy_to_tile[(i, j)] = k

        return xy_to_tile

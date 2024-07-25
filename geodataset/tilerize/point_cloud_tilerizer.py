
from pathlib import Path
from typing import List, Tuple, Union

import laspy
import matplotlib.pyplot as plt
import numpy as np
from pyproj.crs.crs import CRS as PyProjCRS
from tqdm import tqdm

from geodataset.aoi import AOIConfig


class TileMetadata:
    """
    Represents metadata for a tile.
    """

    def _info(self,):
        raise NotImplementedError

    def __repr__(self, ):
        return self._info()
    
class PointCloudIndividualTileMetadata(TileMetadata):
    """
    Represents metadata for an individual tile in a point cloud dataset.

    Attributes:
        min_x (float): The minimum x-coordinate bound of the tile.
        max_x (float): The maximum x-coordinate bound of the tile.
        min_y (float): The minimum y-coordinate bound of the tile.
        max_y (float): The maximum y-coordinate bound of the tile.
        min_z (float): The minimum z-coordinate bound of the tile.
        max_z (float): The maximum z-coordinate bound of the tile.
        crs (PyProjCRS): The coordinate reference system of the tile.
        tile_name (str): The name of the tile.
    """

    def __init__(self, x_bound: Union[Tuple, None] = None, y_bound: Union[Tuple, None] = None, z_bound: Union[Tuple, None] = None, crs: PyProjCRS = None, tile_name: Union[str, None] = None) -> None:
        """
        Initializes a PointCloudIndividualTileMetadata object.

        Args:
            x_bound (tuple or None): The x-coordinate bounds of the tile. Defaults to None.
            y_bound (tuple or None): The y-coordinate bounds of the tile. Defaults to None.
            z_bound (tuple or None): The z-coordinate bounds of the tile. Defaults to None.
            crs (PyProjCRS): The coordinate reference system of the tile. Defaults to None.
            tile_name (str or None): The name of the tile. Defaults to None.
        """
        __slots__ = "min_x", "max_x", "min_y", "max_y", "min_z", "max_z", "crs", "name"

        self.min_x, self.max_x = x_bound if type(x_bound) == tuple else (None, None)
        self.min_y, self.max_y = y_bound if type(y_bound) == tuple else (None, None)
        self.min_z, self.max_z = z_bound if type(z_bound) == tuple else (None, None)

        self.crs = crs
        self.tile_name = tile_name  # //NOTE - tile name will be the name of the files by default

    def _info(self) -> str:
        """
        Returns a string representation of the tile metadata.

        Returns:
            str: A string representation of the tile metadata.
        """
        return f"(min_x,max_x): ({self.min_x}, {self.max_x})\n" \
               + f"(min_y,max_y): ({self.min_y}, {self.max_y})\n" \
               + f"(min_z,max_z): ({self.min_z}, {self.max_z})\n" \
               + f"crs: {self.crs.name}\n" \
               + f"tile_name: {self.tile_name}\n"

class PointCloudTilesMetadata(TileMetadata):
    """
    Represents metadata for a collection of point cloud tiles.

    Attributes:
        tile_metadata_list (List[PointCloudIndividualTileMetadata]): A list of tile metadata objects.
        unique_x_bounds (List[Tuple[float, float]]): A sorted list of unique x-axis bounds.
        unique_y_bounds (List[Tuple[float, float]]): A sorted list of unique y-axis bounds.
        unique_xy_bounds (List[Tuple[float, float, float, float]]): A sorted list of unique x and y-axis bounds.
        min_x (float): The minimum x-coordinate value.
        max_x (float): The maximum x-coordinate value.
        min_y (float): The minimum y-coordinate value.
        max_y (float): The maximum y-coordinate value.
        min_z (float): The minimum z-coordinate value.
        max_z (float): The maximum z-coordinate value.
    """

    def __init__(self, tile_metadata_list:List[PointCloudIndividualTileMetadata]):
        """
        Initializes a PointCloudTilesMetadata object.

        Args:
            tile_metadata_list (List[PointCloudIndividualTileMetadata]): A list of tile metadata objects.
        """
        self.tile_metadata_list = tile_metadata_list

        self.unique_x_bounds = sorted(list(set([(t.min_x, t.max_x) for t in tile_metadata_list])))
        self.unique_y_bounds = sorted(list(set([(t.min_y, t.max_y) for t in tile_metadata_list])))
        self.unique_xy_bounds = sorted(list(set([(t.min_x, t.max_x, t.min_y, t.max_y) for t in tile_metadata_list])))

        self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z = self._add_bounds()

    def __getitem__(self, idx):
        return self.tile_metadata_list[idx]

    def _find_nonemin(self, a, b):
        """
        Finds the minimum value between two numbers, excluding any `None` values in the first number.

        Args:
            a (float or None): The first number.
            b (float): The second number.

        Returns:
            float: The minimum value between `a` and `b`, excluding any `None` values in the first number.
        """
        if a is not None:
            return np.nanmin([a, b])
        else:
            return b

    def _find_nonemax(self, a, b):
        """
        Finds the maximum value between two numbers, excluding any `None` values in the first number.

        Parameters:
        a (float or None): The first number.
        b (float): The second number.

        Returns:
        float: The maximum value between `a` and `b`, excluding any `None` values in the first number.
        """
        if a is not None:
            return np.nanmax([a, b])
        else:
            return b
    
    def _add_bounds(self):
        """
        Calculates the minimum and maximum coordinate values among all the tiles.

        Returns:
            Tuple[float, float, float, float, float, float]: The minimum and maximum x, y, and z-coordinate values.
        """
        min_x = np.inf
        max_x = -np.inf

        min_y = np.inf
        max_y = -np.inf

        min_z = np.inf
        max_z = -np.inf

        for tile in self.tile_metadata_list:
            min_x = self._find_nonemin(tile.min_x, min_x)
            max_x = self._find_nonemax(tile.max_x, max_x)

            min_y = self._find_nonemin(tile.min_y, min_y)
            max_y = self._find_nonemax(tile.max_y, max_y)
            
            min_z = self._find_nonemin(tile.min_z, min_z)
            max_z = self._find_nonemax(tile.max_z, max_z)

        return min_x, max_x, min_y, max_y, min_z, max_z

    def __len__(self):
        return len(self.tile_metadata_list)

    def plot(self) -> plt.Axes:
        """
        Plots the tiles and their boundaries.

        Returns:
            matplotlib.axes.Axes: The plot axes.
        """
        fig, ax = plt.subplots(1,1)
        left, bottom = self.min_x, self.min_y
        width = self.max_x-self.min_x
        height = self.max_y-self.min_y
        bound_rect = plt.Rectangle((left, bottom), width=width, height = height, edgecolor="k", facecolor="w")
        ax.add_patch(bound_rect)

        pad = 0.05

        ax.set_xlim(self.min_x-pad*(width), self.max_x+pad*(width))
        ax.set_ylim(self.min_y-pad*(height), self.max_y+pad*(height))

        for i, tile in enumerate(self.tile_metadata_list):
            width = tile.max_x-tile.min_x
            height = tile.max_y-tile.min_y

            tile_patch = plt.Rectangle((tile.min_x, tile.min_y),width=width, height =height , edgecolor="k", facecolor="k", alpha=0.1)

            ax.add_patch(tile_patch) 
            
        
        randind = np.random.randint(len(self.tile_metadata_list))
        
        tile = self.tile_metadata_list[randind]
        
        rand_patch = plt.Rectangle((tile.min_x, tile.min_y),width=(tile.max_x-tile.min_x), height =(tile.max_y-tile.min_y) , edgecolor="r", facecolor="none")
        ax.add_patch(rand_patch)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("All tiles with a randomly highlighted tile")

        return ax

class PointCloudTilesData:
    """
    Represents a collection of point cloud tiles data.

    Attributes:
        n_tiles (int): The number of tiles in the collection.
        data (list): The list of point cloud data tiles.
        header: The header information of the point cloud data.
    """

    def __init__(self, n_tiles, header: laspy.LasHeader):
            """
            Initializes a PointCloudTilerizer object.

            Args:
                n_tiles (int): The number of tiles.
                header (laspy.LasHeader): The header information.

            Attributes:
                n_tiles (int): The number of tiles.
                data (list): A list to store the data for each tile.
                header (laspy.LasHeader): The header information.
            """
            self.n_tiles = n_tiles
            self.data = [None] * self.n_tiles
            self.header = header
        
    def append(self, index, data):
        """
        Appends the given data to the specified index in the point cloud tilerizer.

        Args:
            index (int): The index where the data should be appended.
            data (laspy.ScaleAwarePointRecord): The data to be appended.

        Returns:
            None
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

    def __getitem__(self, index):
        return self.data[index]

    def clear_data(self):
            """
            Clears the data stored in the point cloud tilerizer.

            This method sets all elements in the `data` list to `None`.

            Parameters:
                None

            Returns:
                None
            """
            self.data = [None] * self.n_tiles
        

class PointCloudTilerizer:
    """
A class for tilerizing point cloud data.

Attributes:
    point_cloud_path (Path): The path to the point cloud file.
    output_folder_path (Path): The path to the output folder where the tiled point clouds will be saved.
    tiles_metadata (List[PointCloudTilesMetadata]): A list of PointCloudTilesMetadata objects containing
        information about the tiles.
    aois_config (AOIConfig, optional): The AOI (Area of Interest) configuration. Defaults to None.
"""
    def __init__(self,
                point_cloud_path: Path,
                output_folder_path: Path,
                tiles_metadata: List[PointCloudTilesMetadata],
                aois_config: AOIConfig = None):
        """
        Initialize the PointCloudTilerizer object.

        Args:
            point_cloud_path (Path): The path to the point cloud file.
            output_folder_path (Path): The path to the output folder where the tiles will be saved.
            tiles_metadata (List[PointCloudTilesMetadata]): A list of metadata objects for each tile.
            aois_config (AOIConfig, optional): The configuration object for the AOIs (Areas of Interest). Defaults to None.
        """
        
        self.point_cloud_path = point_cloud_path
        self.tiles_metadata = tiles_metadata
        self.output_folder_path = Path(output_folder_path)
        
    def lazy_tilerize(self, chunk_size=500_000):
        """
        Tilerizes the point cloud data lazily.

        Args:
            chunk_size (int, optional): The size of each chunk of points to process. Defaults to 500,000.

        Raises:
            AssertionError: If a tile already exists in the output folder.

        """
        self.xy_to_tile_index = self._get_xy_to_tile_index()

        for tile_md in self.tiles_metadata:
            if (self.output_folder_path / ((tile_md.tile_name) + ".las")).is_file():
                raise AssertionError (f"{tile_md.tile_name} exists at {str(self.output_folder_path)}\n" \
                        +  "Updata tile_name in metadata or remove existing files") 

        with laspy.open(self.point_cloud_path) as f:
            self.tile_data = PointCloudTilesData(n_tiles = len(self.tiles_metadata) , header=f.header)
            with tqdm(total= - (f.header.point_records_count//-chunk_size)) as pbar: #NOTE - ceil divide -(a//-b)
                for i, chunked_points in enumerate(f.chunk_iterator(chunk_size)):
                    self.bin_chunk_points(chunked_points)
                    self.lazy_write()
                    pbar.update(1)

    def lazy_write(self):
        """
        Lazily writes the tile data to LAS files.

        Args:
            header (laspy.LasHeader): The header information for the LAS files.

        """
        for i, point_cloud_tile_data in enumerate(self.tile_data):
            if point_cloud_tile_data is not None:
                file_path = self.output_folder_path / (self.tiles_metadata[i].tile_name + ".las")
                
                if file_path.is_file():
                    with laspy.open(file_path, mode="a") as writer:
                        writer.append_points(point_cloud_tile_data)
                else:
                    new_header = laspy.LasHeader(version=self.tile_data.header.version,
                            point_format=self.tile_data.header.point_format)
                    
                    new_header.offsets = self.tile_data.header.offsets
                    new_header.scales = self.tile_data.header.scales

                    with laspy.open(file_path, mode="w", header=new_header) as writer:
                        writer.write_points(point_cloud_tile_data)

        self.tile_data.clear_data()

    def bin_chunk_points(self, chunked_data):
        """
        Bins the chunked points into the appropriate tiles.

        Args:
            chunked_data (laspy.util.ChunkIterator): The chunked point cloud data.
            f (laspy.file.File): The laspy file object.

        """
        for i, x_b in enumerate(self.tiles_metadata.unique_x_bounds):
            index = (x_b[0] < np.array(chunked_data.x)) & (np.array(chunked_data.x) <= x_b[1])
            subset_px = chunked_data[index]
            if len(subset_px)>0:
                for j, y_b in enumerate(self.tiles_metadata.unique_y_bounds):
                    if (x_b+y_b) in self.tiles_metadata.unique_xy_bounds:
                        index = (y_b[0] < np.array(subset_px.y)) & (np.array(subset_px.y) <= y_b[1])
                        subset_pxy = subset_px[index]
                        if (i, j) in self.xy_to_tile_index.keys():
                            file_index = self.xy_to_tile_index[(i, j)]
                            if len(subset_pxy) > 0:
                                self.tile_data.append(file_index, subset_pxy)
    
    def _get_xy_to_tile_index(self):
        """
        Returns a dictionary mapping (x, y) coordinates to tile indexes.

        Returns:
            dict: A dictionary mapping (x, y) coordinates to tile indexes.

        """
        xy_to_tile = {}
        for i, x_bound in enumerate(self.tiles_metadata.unique_x_bounds):
            for j, y_bound in enumerate(self.tiles_metadata.unique_y_bounds):
                for k, pt_md in enumerate(self.tiles_metadata):
                    # //NOTE - Only save the indexes which has data                 
                    if (pt_md.min_x == x_bound[0]) and (pt_md.max_x == x_bound[1]) and (pt_md.min_y == y_bound[0])and (pt_md.max_y == y_bound[1]):
                        xy_to_tile[(i, j)]=k

        return xy_to_tile


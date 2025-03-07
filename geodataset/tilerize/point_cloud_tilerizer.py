import sys
from pathlib import Path
from typing import List, Union

import laspy
import warnings
from laspy import CopcReader

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from shapely import box
from shapely.geometry import Point
from tqdm import tqdm
import rasterio

from geodataset.aoi import AOIFromPackageConfig
from geodataset.aoi.aoi_base import AOIBaseFromGeoFileInCRS
from geodataset.utils import strip_all_extensions_and_path
from geodataset.geodata.point_cloud import PointCloudTileMetadata, PointCloudTileMetadataCollection
from geodataset.utils.file_name_conventions import (
    PointCloudTileNameConvention, validate_and_convert_product_name,
)


class PointCloudTilerizer:
    """
    Class to tilerize point cloud data aligned with raster tiles.
    
    This class creates point cloud tiles that perfectly align with a raster tiling grid.
    It supports the same parameters as RasterTilerizer for consistency.

    Parameters
    ----------
    point_cloud_path : Union[str, Path]
        Path to the point cloud file.
    output_path : Union[str, Path]
        Path to the output folder where the tiles will be saved.
    tile_size : int
        Size of the tile in pixels (when rendered as a raster).
    tile_overlap : float
        Overlap between tiles (0 <= overlap < 1).
    ground_resolution : float, optional
        Ground resolution in meters per pixel.
        Only one of ground_resolution and scale_factor can be provided.
    scale_factor : float, optional
        Scale factor for the data. 
        Only one of ground_resolution and scale_factor can be provided.
    aois_config : Union[AOIFromPackageConfig, None], optional
        Configuration for AOIs.
    keep_dims : List[str], optional
        List of dimensions to keep in the point cloud.
    downsample_voxel_size : float, optional
        Voxel size for downsampling the point cloud.
    reference_raster_path: Union[str, Path, None], optional
        Path to a reference raster for perfect alignment. If provided, 
        the point cloud tiles will exactly match the raster tiles.
    verbose : bool, optional
        Enable verbose output.
    max_tile : int, optional
        Maximum number of tiles to generate.
    force : bool, optional
        Force tilerization even if it would generate too many tiles.
    """

    def __init__(
        self,
        point_cloud_path: Union[str, Path],
        output_path: Union[str, Path],
        tile_size: int,
        tile_overlap: float,
        ground_resolution: float = None,
        scale_factor: float = None,
        aois_config: Union[AOIFromPackageConfig, None] = None,
        keep_dims: List[str] = None,
        downsample_voxel_size: float = None,
        reference_raster_path: Union[str, Path, None] = None,
        verbose: bool = False,
        max_tile: int = 5000,
        force: bool = False,
    ):
        assert not (ground_resolution and scale_factor), (
            "Only one of ground_resolution or scale_factor should be provided"
        )
        
        self.point_cloud_path = Path(point_cloud_path)
        self.product_name = validate_and_convert_product_name(
            strip_all_extensions_and_path(self.point_cloud_path)
        )
        self.output_path = Path(output_path)
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        assert self.tile_overlap < 1.0, "Tile overlap should be less than 1.0"
        
        self.ground_resolution = ground_resolution
        self.scale_factor = scale_factor
        self.aois_config = aois_config
        
        self.keep_dims = keep_dims if keep_dims is not None else "ALL"
        self.downsample_voxel_size = downsample_voxel_size
        self.reference_raster_path = reference_raster_path
        self.verbose = verbose
        self.max_tile = max_tile
        self.force = force
        self.tile_coordinate_step = int((1 - self.tile_overlap) * self.tile_size)

        # initialize the AOI engine if config is provided
        self.aoi_engine = AOIBaseFromGeoFileInCRS(aois_config) if aois_config else None
        
        # Calculate the real world distance equivalent to tile_size pixels
        if self.reference_raster_path:
            self._align_with_reference_raster()
        else:
            self._calculate_tile_side_length()

        self.pc_tiles_folder_path = (
            self.output_path / f"pc_tiles_{self.downsample_voxel_size}"
            if self.downsample_voxel_size
            else self.output_path / "pc_tiles"
        )

        # Generate tiles metadata
        self.tiles_metadata = self._generate_tiles_metadata()

        # Just a sanity check for the maximum number of tiles
        if len(self.tiles_metadata) > max_tile and not force:
            raise ValueError(
                f"Number of tiles {len(self.tiles_metadata)} exceeds the maximum {max_tile}. "
                "Use force=True to override."
            )
            
        # Assign AOIs to tiles if an AOI configuration is provided
        if self.aoi_engine:
            self._assign_aois_to_tiles()
            
        self.create_folder()
        
    def _align_with_reference_raster(self):
        """Align point cloud tiles with an existing raster"""
        with rasterio.open(self.reference_raster_path) as src:
            transform = src.transform
            crs = src.crs
            self.raster_height = src.height
            self.raster_width = src.width
            
            # Get bounds in world coordinates
            bounds = src.bounds
            self.min_x, self.min_y = bounds.left, bounds.bottom
            self.max_x, self.max_y = bounds.right, bounds.top
            
            # Calculate meters per pixel from the transform
            self.x_resolution = transform.a
            self.y_resolution = abs(transform.e)
            
            # Calculate tile side length in world coordinates
            self.tile_side_length_x = self.tile_size * self.x_resolution
            self.tile_side_length_y = self.tile_size * self.y_resolution
            
            if self.verbose:
                print(f"Reference raster bounds: {bounds}")
                print(f"Reference resolution: {self.x_resolution}m/px, {self.y_resolution}m/px")
                print(f"Tile side length: {self.tile_side_length_x}m x {self.tile_side_length_y}m")
                print(f"Using reference raster CRS: {crs}")
                
            # Set the point cloud reader CRS to match the raster if needed
            self.point_cloud_crs = crs
                
    def _calculate_tile_side_length(self):
        """Calculate tile side length based on ground resolution or scale factor"""
        # Read point cloud bounds
        with CopcReader.open(self.point_cloud_path) as reader:
            self.min_x, self.min_y = reader.header.x_min, reader.header.y_min
            self.max_x, self.max_y = reader.header.x_max, reader.header.y_max
            self.point_cloud_crs = reader.header.parse_crs()
            
            if self.ground_resolution:
                # Ground resolution is in meters/pixel
                self.x_resolution = self.y_resolution = self.ground_resolution
                self.tile_side_length_x = self.tile_side_length_y = self.tile_size * self.ground_resolution
            elif self.scale_factor:
                # Need to estimate a reasonable resolution from point cloud density
                # This is approximate and might need adjustment
                point_density = reader.header.point_count / ((self.max_x - self.min_x) * (self.max_y - self.min_y))
                base_resolution = 1.0 / math.sqrt(point_density)  # Estimated meters per point
                self.x_resolution = self.y_resolution = base_resolution * self.scale_factor
                self.tile_side_length_x = self.tile_side_length_y = self.tile_size * self.x_resolution
            else:
                # Default to 1 meter per pixel if neither is provided
                self.x_resolution = self.y_resolution = 1.0
                self.tile_side_length_x = self.tile_side_length_y = self.tile_size
                
            if self.verbose:
                print(f"Point cloud bounds: X={self.min_x}-{self.max_x}, Y={self.min_y}-{self.max_y}")
                print(f"Resolution: {self.x_resolution}m/px")
                print(f"Tile side length: {self.tile_side_length_x}m")
                print(f"Point cloud CRS: {self.point_cloud_crs}")


    def _generate_tiles_metadata(self):
        """Generate tile metadata for all tiles covering the point cloud"""
        name_convention = PointCloudTileNameConvention()
        tiles_metadata = []
        tile_id = 0
        
        # Generate tiles using the tile_overlap parameter
        step_x = self.tile_side_length_x * (1 - self.tile_overlap) 
        step_y = self.tile_side_length_y * (1 - self.tile_overlap)
        
        # Calculate the grid of tiles
        x_positions = list(np.arange(self.min_x, self.max_x, step_x))
        y_positions = list(np.arange(self.min_y, self.max_y, step_y))
        
        for i, y in enumerate(y_positions):
            for j, x in enumerate(x_positions):
                # Define tile bounds
                x_bound = [x, x + self.tile_side_length_x]
                y_bound = [y, y + self.tile_side_length_y]
                
                tile_name = name_convention.create_name(
                    product_name=self.product_name,
                    row=i,
                    col=j,
                    voxel_size=self.downsample_voxel_size,
                    ground_resolution=self.ground_resolution,
                    scale_factor=self.scale_factor
                )
                
                tile_md = PointCloudTileMetadata(
                    x_bound=x_bound,
                    y_bound=y_bound,
                    crs=self.point_cloud_crs,
                    tile_name=tile_name,
                    tile_id=tile_id,
                    row=i,
                    col=j,
                    height=self.tile_size,
                    width=self.tile_size,
                    ground_resolution=self.ground_resolution,
                    scale_factor=self.scale_factor
                )
                
                tiles_metadata.append(tile_md)
                tile_id += 1
        
        return PointCloudTileMetadataCollection(
            tiles_metadata, product_name=self.product_name
        )
        
    def query_tile(self, tile_md, reader):
        """
        Extract points within a tile boundary from the point cloud.
        
        Parameters
        ----------
        tile_md : PointCloudTileMetadata
            Metadata for the tile
        reader : laspy.CopcReader
            Open COPC reader for the point cloud
            
        Returns
        -------
        laspy.LasData
            The points within the tile boundary
        """
        reader_crs = reader.header.parse_crs()

        if tile_md.crs.to_epsg() != reader_crs.to_epsg():
            # Transform bounds if CRS doesn't match
            min_max_gdf = gpd.GeoDataFrame(
                geometry=[box(tile_md.min_x, tile_md.min_y, tile_md.max_x, tile_md.max_y)],
                crs=tile_md.crs
            )
            min_max_gdf = min_max_gdf.to_crs(reader_crs)
            bounds = min_max_gdf.geometry[0].bounds
            mins = np.array([bounds[0], bounds[1]])
            maxs = np.array([bounds[2], bounds[3]])
        else:
            mins = np.array([tile_md.min_x, tile_md.min_y])
            maxs = np.array([tile_md.max_x, tile_md.max_y])

        b = laspy.copc.Bounds(mins=mins, maxs=maxs)
        data = reader.query(b)
        return data

    def tilerize(self):
        """
        Generate point cloud tiles for the entire dataset.
        
        Returns
        -------
        PointCloudTileMetadataCollection
            Collection of tiles metadata
        """
        # Tilerize the point cloud
        self._tilerize()
        
        # Plot AOIs if available
        if hasattr(self, 'aoi_engine') and self.aoi_engine:
            self.plot_aois()
            
        return self.tiles_metadata
    
    def _tilerize(self):
        """
        Internal implementation of tilerization.
        """
        new_tile_md_list = []
        with CopcReader.open(self.point_cloud_path) as reader:
            for tile_md in tqdm(self.tiles_metadata, desc="Processing tiles"):
                # Query points within this tile
                data = self.query_tile(tile_md, reader)
                
                if len(data) == 0:
                    if self.verbose:
                        print(f"Skipping empty tile {tile_md.tile_name}")
                    continue
                    
                # Convert to Open3D format
                pcd = self._laspy_to_o3d(
                    data, 
                    self.keep_dims.copy() if type(self.keep_dims) is not str else self.keep_dims
                )
                
                # Apply downsampling if requested
                if self.downsample_voxel_size:
                    pcd = self._downsample_tile(pcd, self.downsample_voxel_size)
                    
                # Remove duplicate points
                pcd = self._keep_unique_points(pcd)
                
                # Remove points outside the tile's AOI if assigned
                if hasattr(self, 'aoi_engine') and tile_md.aoi:
                    pcd = self._remove_points_outside_aoi(pcd, tile_md, reader.header.parse_crs())
                    
                # Save the tile only if it has points
                if pcd.point.positions.shape[0] > 0:
                    new_tile_md_list.append(tile_md)
                    
                    # Save the tile to disk
                    output_dir = self.pc_tiles_folder_path / f"{tile_md.aoi if tile_md.aoi else 'noaoi'}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_file = output_dir / tile_md.tile_name
                    
                    o3d.t.io.write_point_cloud(str(output_file), pcd)

        print(f"Finished tilerizing. Generated {len(new_tile_md_list)} tiles with points.")
        self.tiles_metadata = PointCloudTileMetadataCollection(
            new_tile_md_list, product_name=self.tiles_metadata.product_name
        )
            
    def _laspy_to_o3d(self, pc_file, keep_dims):
        """
        Convert LAS point cloud to Open3D format.
        
        Parameters
        ----------
        pc_file : laspy.LasData
            Point cloud data from LAS/LAZ file
        keep_dims : list or str
            Dimensions to keep
            
        Returns
        -------
        o3d.t.geometry.PointCloud
            Open3D point cloud
        """
        float_type = np.float32
        dimensions = list(pc_file.point_format.dimension_names)

        if keep_dims == "ALL":
            keep_dims = dimensions.copy()

        # Ensure required dimensions are present
        assert all([dim in keep_dims for dim in ["X", "Y", "Z"]])
        assert all([dim in dimensions for dim in keep_dims])

        # Get XYZ coordinates
        pc = np.ascontiguousarray(
            np.vstack([pc_file.x, pc_file.y, pc_file.z]).T.astype(float_type)
        )

        # Build tensor map
        map_to_tensors = {"positions": pc}

        # Remove XYZ from dimensions to keep
        processing_dims = [d for d in keep_dims if d not in ["X", "Y", "Z"]]

        # Process colors
        if all(color in keep_dims for color in ["red", "green", "blue"]):
            colors = np.ascontiguousarray(
                np.vstack([pc_file.red, pc_file.green, pc_file.blue]).T.astype(float_type)
            )
            pc_colors = np.round(colors/255)
            map_to_tensors["colors"] = pc_colors.astype(np.uint8)
            
            # Remove color dimensions from further processing
            processing_dims = [d for d in processing_dims if d not in ["red", "green", "blue"]]

        # Add remaining dimensions
        for dim in processing_dims:
            dim_value = np.ascontiguousarray(pc_file[dim]).astype(float_type)
            dim_value = dim_value.reshape(-1, 1)
            map_to_tensors[dim.lower()] = dim_value

        return o3d.t.geometry.PointCloud(map_to_tensors)
        
    def _keep_unique_points(self, pcd):
        """
        Remove duplicate points from point cloud
        
        Parameters
        ----------
        pcd : o3d.t.geometry.PointCloud
            Input point cloud
            
        Returns
        -------
        o3d.t.geometry.PointCloud
            Point cloud with duplicates removed
        """
        map_to_tensors = {}
        positions = pcd.point.positions.numpy()
        
        # Find unique points by position
        unique_pc, ind = np.unique(positions, axis=0, return_index=True)
        
        map_to_tensors["positions"] = unique_pc

        # Keep the attributes for each unique point
        for attr in pcd.point:
            if attr != "positions":
                map_to_tensors[attr] = getattr(pcd.point, attr)[ind]

        if self.verbose:
            n_removed = positions.shape[0] - unique_pc.shape[0]
            percentage_removed = (n_removed / positions.shape[0]) * 100 if positions.shape[0] > 0 else 0
            print(f"Removed {n_removed} duplicate points ({percentage_removed:.2f}%)")
            
        return o3d.t.geometry.PointCloud(map_to_tensors)

    def _downsample_tile(self, pcd, voxel_size):
        """
        Downsample a point cloud using voxel grid
        
        Parameters
        ----------
        pcd : o3d.t.geometry.PointCloud
            Input point cloud
        voxel_size : float
            Voxel size for downsampling
            
        Returns
        -------
        o3d.t.geometry.PointCloud
            Downsampled point cloud
        """
        if pcd.point.positions.shape[0] == 0:
            return pcd
            
        downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
        
        if self.verbose:
            n_removed = pcd.point.positions.shape[0] - downsampled.point.positions.shape[0]
            percentage_removed = (n_removed / pcd.point.positions.shape[0]) * 100
            print(f"Downsampling removed {n_removed} points ({percentage_removed:.2f}%)")
            
        return downsampled

    def _remove_points_outside_aoi(self, pcd, tile_md, reader_crs):
        """
        Remove points that fall outside the AOI for this tile
        
        Parameters
        ----------
        pcd : o3d.t.geometry.PointCloud
            Input point cloud
        tile_md : PointCloudTileMetadata
            Metadata for the tile
        reader_crs : PyProjCRS
            CRS of the point cloud reader
            
        Returns
        -------
        o3d.t.geometry.PointCloud
            Point cloud with only points inside the AOI
        """
        if not hasattr(self, 'aoi_engine') or tile_md.aoi is None:
            return pcd
            
        aoi_gdf = self.aoi_engine.loaded_aois.get(tile_md.aoi)
        if aoi_gdf is None:
            if self.verbose:
                print(f"No AOI found for {tile_md.aoi}")
            return pcd
            
        # Convert AOI to point cloud CRS if needed
        aoi_gdf = aoi_gdf.to_crs(reader_crs)
        
        # Check if we have points in the point cloud
        if pcd.point.positions.shape[0] == 0:
            return pcd
            
        positions = pcd.point.positions.numpy()
        
        # Quick check if all points are in the AOI bounds
        aoi_bounds = aoi_gdf.total_bounds
        min_x, min_y, max_x, max_y = aoi_bounds
        
        # If all points are within the AOI bounds, return the original point cloud
        points_min_x = np.min(positions[:, 0])
        points_max_x = np.max(positions[:, 0])
        points_min_y = np.min(positions[:, 1])
        points_max_y = np.max(positions[:, 1])
        
        if (points_min_x >= min_x and points_max_x <= max_x and 
            points_min_y >= min_y and points_max_y <= max_y):
            return pcd
            
        # If not, do a proper spatial join
        point_gdf = gpd.GeoDataFrame(
            geometry=[Point(x, y, z) for x, y, z in positions],
            crs=reader_crs
        )
        
        # Find points in the AOI
        points_in_aoi = gpd.sjoin(point_gdf, aoi_gdf, predicate="within")
        
        if points_in_aoi.empty:
            # No points in AOI, return empty point cloud
            return o3d.t.geometry.PointCloud({"positions": np.zeros((0, 3))})
            
        # Get indices of points in the AOI
        indices_in_aoi = points_in_aoi.index.values
        
        # Extract positions for points in AOI
        positions_in_aoi = positions[indices_in_aoi]
        
        # Create new tensor map with only points in AOI
        map_to_tensors = {"positions": positions_in_aoi}
        
        # Copy other attributes for points in AOI
        for attr in pcd.point:
            if attr != "positions":
                map_to_tensors[attr] = getattr(pcd.point, attr)[indices_in_aoi]
                
        if self.verbose:
            n_removed = positions.shape[0] - positions_in_aoi.shape[0]
            percentage_removed = (n_removed / positions.shape[0]) * 100
            print(f"Removed {n_removed} points outside AOI {tile_md.aoi} ({percentage_removed:.2f}%)")
        
        return o3d.t.geometry.PointCloud(map_to_tensors)


    def create_folder(self):
        """Create output folders for tiles"""
        self.pc_tiles_folder_path.mkdir(parents=True, exist_ok=True)
        
        # Create folders for each AOI
        if hasattr(self, 'aoi_engine') and self.aoi_engine:
            aoi_set = {tile.aoi for tile in self.tiles_metadata if tile.aoi}
            for aoi in aoi_set:
                (self.pc_tiles_folder_path / aoi).mkdir(parents=True, exist_ok=True)
        
        # Always create noaoi folder for tiles without AOI assignment
        (self.pc_tiles_folder_path / "noaoi").mkdir(parents=True, exist_ok=True)
            
    def _check_aois_in_both_tiles_and_config(self):
        """Check which AOIs are both in tiles metadata and AOI config"""
        if not hasattr(self, 'aoi_engine') or not hasattr(self, 'tiles_metadata'):
            return []
            
        tile_aois = set(getattr(self.tiles_metadata, 'aois', []))
        config_aois = set(self.aoi_engine.loaded_aois.keys())
        
        aois_in_both = tile_aois & config_aois
        
        if tile_aois != config_aois:
            warnings.warn(
                f"AOIs in the AOI engine and the tiles metadata do not match. "
                f"Got {tile_aois} from tiles_metadata and "
                f"{config_aois} from aoi_config. "
                f"Will only save tiles for AOIs present in both: {aois_in_both}"
            )
            
        return aois_in_both

    def _assign_aois_to_tiles(self):
        """Assign AOIs to tiles based on spatial intersection"""
        if not hasattr(self, 'aoi_engine') or not self.aoi_engine:
            return
        
        print("Assigning AOIs to point cloud tiles...")
        
        # Create a GeoDataFrame from the tiles
        tiles_gdf = gpd.GeoDataFrame(
            {
                'tile_id': [tile.tile_id for tile in self.tiles_metadata],
                'geometry': [box(tile.min_x, tile.min_y, tile.max_x, tile.max_y) for tile in self.tiles_metadata]
            },
            crs=self.point_cloud_crs
        )
        
        # Get the AOIs GeoDataFrame and ensure same CRS
        aois_gdf = self.aoi_engine.get_aoi_gdf()
        if aois_gdf.crs != tiles_gdf.crs:
            aois_gdf = aois_gdf.to_crs(tiles_gdf.crs)
        
        # Find intersections between tiles and AOIs
        intersections = gpd.overlay(tiles_gdf, aois_gdf, how="intersection")
        
        # Calculate intersection areas
        intersections['intersection_area'] = intersections.geometry.area
        
        # Find which AOI has the maximum intersection with each tile
        max_intersections = intersections.loc[intersections.groupby('tile_id')['intersection_area'].idxmax()]
        
        # Create a mapping from tile_id to AOI
        tile_id_to_aoi = dict(zip(max_intersections.tile_id, max_intersections.aoi))
        
        # Update the AOI attribute for each tile
        for tile in self.tiles_metadata:
            if tile.tile_id in tile_id_to_aoi:
                tile.aoi = tile_id_to_aoi[tile.tile_id]
                if self.verbose:
                    print(f"Assigned tile {tile.tile_name} to AOI {tile.aoi}")
        
        # Get count of tiles per AOI
        aoi_counts = {}
        for tile in self.tiles_metadata:
            aoi = tile.aoi if tile.aoi else "noaoi"
            aoi_counts[aoi] = aoi_counts.get(aoi, 0) + 1
        
        for aoi, count in aoi_counts.items():
            print(f"AOI '{aoi}': {count} tiles")

    def plot_aois(self):
        """Plot the AOIs and tiles"""
        fig, ax = self.tiles_metadata.plot()
        
        # Add AOIs to the plot if available
        if hasattr(self, 'aoi_engine'):
            palette = {
                "blue": "#26547C",
                "red": "#EF476F",
                "yellow": "#FFD166",
                "green": "#06D6A0",
            }
            
            for aoi_name, color in zip(self.aoi_engine.loaded_aois.keys(), palette.values()):
                if aoi_name in self.aoi_engine.loaded_aois:
                    self.aoi_engine.loaded_aois[aoi_name].plot(
                        ax=ax, color=color, alpha=0.5
                    )
        
        plt.savefig(self.output_path / f"{self.tiles_metadata.product_name}_aois.png")
        return fig, ax

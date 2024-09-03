from pathlib import Path
from typing import List, Tuple, Union

import laspy
import matplotlib.pyplot as plt
import numpy as np
from pyproj import CRS as PyProjCRS
from tqdm import tqdm

from geodataset.aoi import AOIFromPackageConfig

import geopandas as gpd

from shapely import MultiPolygon, Polygon
import json

from geodataset.metadata.tile_metadata import TileMetadataCollection
import laspy
import numpy as np
import pandas as pd
from geodataset.utils.file_name_conventions import CocoNameConvention
from geodataset.dataset.coco_generator import PointCloudCOCOGenerator


class PointCloudTiles:
    """
    A class for storing point cloud data in tiles.

    Parameters
    ----------
    n_tiles: int
        The number of tiles.
    header: laspy.LasHeader
        The header of the point cloud data.
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
                offsets=self.header.offsets,
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
        

class PointCloudTilerizer:
    """
    A class for tilerizing point cloud data.

    Parameters
    ----------
    point_cloud_path: Path
        The path to the point cloud data.
    output_folder_path: Path
        The path to the output folder.
    tiles_metadata: List[TileCollectionMetadata]
        A list of tile collection metadata.
    """

    def __init__(
        self,
        point_cloud_path: Path,
        output_path: Path,
        tiles_metadata: Union[TileMetadataCollection, None]=None,
        aois_config: Union[AOIFromPackageConfig , None] = None,
    ):
        
        self.point_cloud_path = point_cloud_path
        self.tiles_metadata = tiles_metadata
        self.output_path = Path(output_path)
        self.aoi_engine =  AOIBaseFromGeoFile(aois_config)

        
        self.pc_tiles_folder_path = self.output_path / "point_clouds"
        self.annotation_folder_path = self.output_path / "annotations"

        self.create_folder()


    def create_folder(self):

        self.pc_tiles_folder_path.mkdir(parents=True, exist_ok=True)
        self.annotation_folder_path.mkdir(parents=True, exist_ok=True)

    def _lazy_tilerize(self, chunk_size: Union[int, None] = 500_000) -> None:
        """
        Tilerizes the point cloud data lazily.
        """
        self.xy_to_tile_index = self._get_xy_to_tile_index()

        for tile_md in self.tiles_metadata:
            file_path = self.pc_tiles_folder_path /f"{tile_md.output_filename}.las"
            if file_path.is_file():
                raise FileExistsError(
                    f"{file_path} exists at {str(self.pc_tiles_folder_path)}." +
                    "Update output_filename in metadata or remove existing files."
                )

        with laspy.open(self.point_cloud_path) as f:
            self.tile_data = PointCloudTiles(
                n_tiles=len(self.tiles_metadata), header=f.header
            )
            if chunk_size is None:
                chunk_size = f.header.point_records_count
            with tqdm(
                total=-(-f.header.point_records_count // chunk_size)
            ) as pbar:  # ceil divide -(a // -b)
                for chunked_points in f.chunk_iterator(chunk_size):
                    self._bin_chunk_points(chunked_points)
                    self._lazy_write()
                    pbar.update(1)

        self.plot_aoi()

    def lazy_tilerize(self, chunk_size: Union[int, None] = 500_000) -> None:
        
        self.generate_coco_labels()

        self._lazy_tilerize(chunk_size=chunk_size)
        self.generate_coco_labels()

    def _lazy_write(self) -> None:
        """
        Lazily writes the tile data to LAS files.
        """
        for i, point_cloud_tile_data in enumerate(self.tile_data):
            if point_cloud_tile_data is not None:
                file_path = (
                    self.pc_tiles_folder_path / 
                    f"{self.tiles_metadata[i].output_filename}"
                )
                if file_path.is_file():
                    with laspy.open(file_path, mode="a") as writer:
                        writer.append_points(point_cloud_tile_data)
                else:
                    new_header = laspy.LasHeader(
                        version=self.tile_data.header.version,
                        point_format=self.tile_data.header.point_format,
                    )
                    new_header.offsets = self.tile_data.header.offsets
                    new_header.scales = self.tile_data.header.scales

                    with laspy.open(file_path, mode="w", header=new_header) as writer:
                        writer.write_points(point_cloud_tile_data)

        self.tile_data.clear_data()

    def tilerize(
        self,
    ) -> None:
        """
        Tilerizes the point cloud data.
        """
        self.lazy_tilerize(chunk_size=None)
        # Add AOI to the tile


    def _get_aoi_tiles(self):
        aois_gdf = self.aoi_engine.get_aoi_gdf()
        # //NOTE -  Cannot check for color as data is not provided here and only metadata is provided.

        tiles_gdf = self.tiles_metadata.gdf
        
        intersections = gpd.overlay(tiles_gdf, aois_gdf, how='intersection')
        intersections['intersection_area'] = intersections.geometry.area
        max_intersection_per_tile = intersections.loc[intersections.groupby('tile_id')['intersection_area'].idxmax()]
        aois_tiles = max_intersection_per_tile.groupby('aoi')['tile_id'].apply(list).to_dict()

        return aois_tiles

    def generate_coco_labels(self):
        
        aoi_tiles = self._get_aoi_tiles()
        coco_paths = {}

        for aoi in aoi_tiles:

            tiles_metadata = TileMetadataCollection([tile for tile in self.tiles_metadata if tile.id in aoi_tiles[aoi]])

            coco_output_file_path = self.annotation_folder_path / CocoNameConvention.create_name(
                    product_name="PointCloud",
                    ground_resolution=None,
                    scale_factor=None,
                    fold=aoi
                )

            polygons = [[Polygon()] for _ in tiles_metadata]

            coco_generator = PointCloudCOCOGenerator(
                description=f"Dataset for the product XYZ",
                tiles_metadata=tiles_metadata,
                polygons=polygons,
                output_path=coco_output_file_path,
                scores=None,
                categories=None,
                other_attributes=None,
                use_rle_for_labels=False,
                n_workers=1,
                coco_categories_list=None
            )

            coco_generator.generate_coco()
            coco_paths[aoi] = coco_output_file_path

        return coco_paths
    
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
                    if (
                        (pt_md.min_x == x_bound[0])
                        and (pt_md.max_x == x_bound[1])
                        and (pt_md.min_y == y_bound[0])
                        and (pt_md.max_y == y_bound[1])
                    ):
                        xy_to_tile[(i, j)] = k

        return xy_to_tile


#### AOIConfig

class AOIBaseFromGeoFile:
    def __init__(self, aois_config: AOIFromPackageConfig):
        
        self.aois_config = aois_config
        self.loaded_aois = self._load_aois()
    
    def _load_aois(self):
        """
        Load the AOI from the provided path, converting it to a MultiPolygon if necessary.
        """

        loaded_aois = {}
        for aoi_name in self.aois_config.aois:
            # Load the AOI using geopandas
            aoi_gdf = gpd.read_file(self.aois_config.aois[aoi_name])

            # Ensure the geometry is a MultiPolygon
            aoi_gdf['geometry'] = aoi_gdf['geometry'].astype(object).apply(
                lambda geom: MultiPolygon([geom]) if isinstance(geom, Polygon) else geom
            )

            loaded_aois[aoi_name] = aoi_gdf

        return loaded_aois
    
    def get_aoi_gdf(self):
        aois_frames = []

        for aoi, gdf in self.loaded_aois.items():
            gdf = gdf.copy()
            gdf["aoi"] = aoi
            aois_frames.append(gdf)

        aois_gdf = gpd.GeoDataFrame(pd.concat(aois_frames, ignore_index=True)).reset_index()

        return aois_gdf
    
    def plot_aoi(self) -> None:
        
        fig, ax = self.tiles_metadata.plot()

        self.write_aoi()
        palette = {
            "blue": "#26547C",
            "red": "#EF476F",
            "yellow": "#FFD166",
            "green": "#06D6A0",
        }

        self.aoi_engine.loaded_aois["train"].plot(ax=ax, color=palette["blue"], alpha=0.5)
        self.aoi_engine.loaded_aois["valid"].plot(ax=ax, color=palette["green"], alpha=0.5)
        self.aoi_engine.loaded_aois["test"].plot(ax=ax, color=palette["red"], alpha=0.5)

        plt.savefig(self.output_path / "split.png")
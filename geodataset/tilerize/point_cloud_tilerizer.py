from pathlib import Path
from typing import List, Tuple, Union

import laspy
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from geodataset.aoi import AOIFromPackageConfig

import geopandas as gpd

from shapely import MultiPolygon, Polygon
from laspy import CopcReader

from geodataset.metadata.tile_metadata import TileMetadataCollection, TileMetadata
import laspy
import numpy as np
import pandas as pd
from geodataset.utils.file_name_conventions import CocoNameConvention
from geodataset.dataset.coco_generator import PointCloudCOCOGenerator
import open3d as o3d
import warnings 
from geodataset.utils.file_name_conventions import PointCloudTileNameConvention

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
        tile_resolution: float = None,
        tile_overlap: float = None,
        max_tile: int = 5000,
        keep_dims: List[str] = None,
        downsample_voxel_size: float = None,
        verbose: bool = False,
        
):
        
        self.point_cloud_path = point_cloud_path
        self.tiles_metadata = tiles_metadata
        self.output_path = Path(output_path)
        self.aoi_engine =  AOIBaseFromGeoFile(aois_config)
        self.tile_resolution = tile_resolution
        self.tile_overlap = tile_overlap
        self.keep_dims = keep_dims if keep_dims is not None else "ALL"
        self.downsample_voxel_size = downsample_voxel_size
        self.verbose = verbose


        assert tile_resolution or self.tiles_metadata

        if self.tiles_metadata:
            if self.tile_resolution:
                warnings.warm("Tile Metadata is provided, so the tile resolution will be ignored")
            if self.tile_overlap!=None:
                warnings.warn("Tile Metadata is provided, so the tile overlap will be ignored")

        if self.tiles_metadata is None:
            assert self.tile_overlap is not None, "Tile Overlap is required if tile metadata is not provided"

            self.populate_tiles_metadata()

        assert len(self.tiles_metadata) < max_tile, f"Number of max possible tiles {len(self.tiles_metadata)} exceeds the maximum number of tiles {max_tile}"

        self.pc_tiles_folder_path = self.output_path / f"pc_tiles_{self.downsample_voxel_size}" if self.downsample_voxel_size else self.output_path / "pc_tiles"
        self.annotation_folder_path = self.output_path / "annotations"



        self.create_folder()

    def populate_tiles_metadata(self,):

        name_convention = PointCloudTileNameConvention()
        
        product_name = self.point_cloud_path.stem.split(".")[0]
        with CopcReader.open(self.point_cloud_path) as reader:
            min_x, min_y = reader.header.x_min, reader.header.y_min
            max_x, max_y = reader.header.x_max, reader.header.y_max

            tiles_metadata = []
            for i, x in enumerate(np.arange(min_x, max_x, self.tile_resolution*self.tile_overlap)):
                for j, y in enumerate(np.arange(min_y, max_y, self.tile_resolution*self.tile_overlap)):
                    x_bound = [x, x + self.tile_resolution]
                    y_bound = [y, y + self.tile_resolution]

                    tile_name = name_convention.create_name(product_name=product_name, tile_id=f"{i}_{j}")
                    
                    tile_md = TileMetadata(id = f"{i}_{j}", x_bound=x_bound, y_bound=y_bound, crs = reader.header.parse_crs(), output_filename=tile_name)

                    tiles_metadata.append(tile_md)

            self.tiles_metadata = TileMetadataCollection(tiles_metadata)


    def create_folder(self):

        self.pc_tiles_folder_path.mkdir(parents=True, exist_ok=True)
        self.annotation_folder_path.mkdir(parents=True, exist_ok=True)
        
    def query_tile(self, tile_md, reader):
        mins = np.array([tile_md.min_x, tile_md.min_y])
        maxs = np.array([tile_md.max_x, tile_md.max_y])
        b = laspy.copc.Bounds(mins=mins, maxs=maxs)    
        data = reader.query(b)
        
        return data
    
    def _tilerize(self,):
        
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
                downsampled_tile_path = self.pc_tiles_folder_path / f"{tile_md.output_filename.replace('.las', '.ply')}"
                o3d.t.io.write_point_cloud(str(downsampled_tile_path), pcd)

        self.tiles_metadata = TileMetadataCollection(new_tile_md_list)

    def tilerize(self,):
        self._generate_coco_labels()
        self._tilerize()
        self.plot_aois() # This should come after tilerize as the tiles_metadata will be updated after tilerize


    def _get_aoi_tiles(self):
        aois_gdf = self.aoi_engine.get_aoi_gdf()
        # //NOTE -  Cannot check for color as data is not provided here and only metadata is provided.

        tiles_gdf = self.tiles_metadata.gdf
        aois_gdf.crs = tiles_gdf.crs       
        intersections = gpd.overlay(tiles_gdf, aois_gdf, how='intersection')
        intersections['intersection_area'] = intersections.geometry.area
        max_intersection_per_tile = intersections.loc[intersections.groupby('tile_id')['intersection_area'].idxmax()]
        aois_tiles = max_intersection_per_tile.groupby('aoi')['tile_id'].apply(list).to_dict()

        return aois_tiles

    def _generate_coco_labels(self):
        
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

    def plot_aois(self) -> None:
        
        fig, ax = self.tiles_metadata.plot()

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
    

    def _laspy_to_o3d(self, pc_file: Path, keep_dims:List[str]):

        dimensions = list(pc_file.point_format.dimension_names)
        
        if keep_dims == "ALL":
            keep_dims = dimensions

        
        
        assert all([dim in keep_dims for dim in ["X", "Y", "Z"]])
        assert all([dim in dimensions for dim in keep_dims])

        pc = np.ascontiguousarray(np.vstack([pc_file.x, pc_file.y, pc_file.z]).T.astype(np.float64))


        map_to_tensors = {}
        map_to_tensors["positions"] = pc.astype(np.float64)


        keep_dims.remove("X")
        keep_dims.remove("Y")
        keep_dims.remove("Z")

        if "red" in keep_dims and "green" in keep_dims and "blue" in keep_dims:

            pc_colors = np.ascontiguousarray(np.vstack([pc_file.red, pc_file.green, pc_file.blue]).T.astype(np.float64))/255
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

    def _keep_unique_points(self, pcd):
        
        map_to_tensors = {}
        positions =  pcd.point.positions.numpy()
        unique_pc, ind = np.unique(positions, axis=0, return_index=True)

        map_to_tensors["positions"] = unique_pc
        
        for attr in pcd.point:
            if attr != "positions":
                map_to_tensors[attr] = getattr(pcd.point, attr)[ind]

        if self.verbose:
            n_removed = positions.shape[0] - unique_pc.shape[0]
            percentage_removed = (n_removed/positions.shape[0])*100
            print(f"Removed {n_removed} duplicate points ({percentage_removed:.2f}%)")
        return o3d.t.geometry.PointCloud(map_to_tensors)


    def _downsample_tile(self, pcd, voxel_size) -> None:
        
        pcd =  pcd.voxel_down_sample(voxel_size=voxel_size)

        return pcd


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
    
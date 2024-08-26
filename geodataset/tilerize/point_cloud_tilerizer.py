from pathlib import Path
from typing import List, Tuple, Union

import laspy
import matplotlib.pyplot as plt
import numpy as np
from pyproj import CRS as PyProjCRS
from tqdm import tqdm

from geodataset.utils.file_name_conventions import PointCloudTileNameConvention
from geodataset.aoi import AOIFromPackageConfig

from abc import ABC, abstractmethod
import geopandas as gpd

from shapely import MultiPolygon, Polygon
from shapely.geometry import box
import json

class TileMetadata:
    """
    Represents metadata for a tile, Generally used for PointCloudTile.

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
        x_bound: Union[Tuple[float, float], None] = None,
        y_bound: Union[Tuple[float, float], None] = None,
        z_bound: Union[Tuple[float, float], None] = None,
        crs: PyProjCRS = None,
        output_filename: Union[str, None] = None,
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
            "output_filename",
            "bbox"
        )

        self.min_x, self.max_x = x_bound if x_bound else (None, None)
        self.min_y, self.max_y = y_bound if y_bound else (None, None)
        self.min_z, self.max_z = z_bound if z_bound else (None, None)

        self.crs = crs
        assert PointCloudTileNameConvention._validate_name(
            output_filename
        ), f"Invalid output_filename: {output_filename}"

        self.output_filename = output_filename
        self.bbox = self._get_bounding_box()

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


class TileCollectionMetadata:
    """
    Represents metadata for a collection of point cloud tiles.

    Parameters
    ----------
    tile_metadata_list: List[TileMetadata]
        A list of tile metadata.
    """

    def __init__(self, tile_metadata_list: List[TileMetadata]):
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
        ax.set_title("All tiles with a randomly highlighted tile")

        return fig, ax

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
        output_folder_path: Path,
        tiles_metadata: List[TileCollectionMetadata],
        aois_config: Union[AOIFromPackageConfig , None] = None,
    ):
        self.point_cloud_path = point_cloud_path
        self.tiles_metadata = tiles_metadata
        self.output_folder_path = Path(output_folder_path)
        self.aois_config = aois_config
        self.aoi_engine =  AOIBaseFromGeoFile(self.aois_config)

        # Create point_clouds folder
        (self.output_folder_path / "point_clouds").mkdir(parents=True, exist_ok=True)

        # Create annotation folder
        (self.output_folder_path / "annotations").mkdir(parents=True, exist_ok=True)


    def lazy_tilerize(self, chunk_size: Union[int, None] = 500_000) -> None:
        """
        Tilerizes the point cloud data lazily.
        """
        # self.xy_to_tile_index = self._get_xy_to_tile_index()

        # for tile_md in self.tiles_metadata:
        #     file_path = self.output_folder_path / "point_cloud" /f"{tile_md.output_filename}.las"
        #     if file_path.is_file():
        #         raise FileExistsError(
        #             f"{file_path} exists at {str(self.output_folder_path)}." +
        #             "Update output_filename in metadata or remove existing files."
        #         )

        # with laspy.open(self.point_cloud_path) as f:
        #     self.tile_data = PointCloudTiles(
        #         n_tiles=len(self.tiles_metadata), header=f.header
        #     )
        #     if chunk_size is None:
        #         chunk_size = f.header.point_records_count
        #     with tqdm(
        #         total=-(-f.header.point_records_count // chunk_size)
        #     ) as pbar:  # ceil divide -(a // -b)
        #         for chunked_points in f.chunk_iterator(chunk_size):
        #             self._bin_chunk_points(chunked_points)
        #             self._lazy_write()
        #             pbar.update(1)

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

        plt.savefig(self.output_folder_path / "split.png")

    def _lazy_write(self) -> None:
        """
        Lazily writes the tile data to LAS files.
        """
        for i, point_cloud_tile_data in enumerate(self.tile_data):
            if point_cloud_tile_data is not None:
                file_path = (
                    self.output_folder_path
                    / "point_clouds" / 
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


    def write_aoi(self, ):
        

        train_coco = COCO_MMS(self.output_folder_path / "annotations"/ "train.json")
        valid_coco = COCO_MMS(self.output_folder_path / "annotations"/ "valid.json")
        test_coco = COCO_MMS(self.output_folder_path / "annotations"/ "test.json")

        for i, tile_md in enumerate(self.tiles_metadata):
            bbox = tile_md.bbox
            train_polygon = gpd.overlay(bbox, self.aoi_engine.loaded_aois["train"])
            valid_polygon = gpd.overlay(bbox, self.aoi_engine.loaded_aois["valid"])
            test_polygon = gpd.overlay(bbox, self.aoi_engine.loaded_aois["test"])

            file_name = tile_md.output_filename

            if len(train_polygon):
                mask = self._mask(bbox, train_polygon)
                if mask is None:
                    print(i, file_name)
                train_coco.add_point_cloud(point_cloud_id=i, file_name=file_name, mask=mask)
            if len(valid_polygon):
                mask = self._mask(bbox, valid_polygon)
                valid_coco.add_point_cloud(point_cloud_id=i, file_name=file_name, mask=mask)
            if len(test_polygon):
                mask = self._mask(bbox,test_polygon)
                test_coco.add_point_cloud(point_cloud_id=i, file_name=file_name, mask=mask)

        train_coco.write_json()
        valid_coco.write_json()
        test_coco.write_json()

    def _mask(self, bbox, aoi):
        return None if (aoi.area == bbox.area).all() else json.loads(aoi.to_json())
    
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
        self.crs = None #TODO: Check for CRS  

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

            # Making sure the geometries have the same CRS as the raster
            # aoi_gdf = self.associated_raster.adjust_geometries_to_raster_crs_if_necessary(gdf=aoi_gdf)

            # Scaling the geometries to pixel coordinates aligned with the Raster
            # aoi_gdf = self.associated_raster.adjust_geometries_to_raster_pixel_coordinates(gdf=aoi_gdf)

            # Store the loaded data
            loaded_aois[aoi_name] = aoi_gdf

        return loaded_aois


class COCO_MMS:
    """
    Multi-modal coco dataset generator with split information 
    """
    def __init__(self, output_path):
        self.output_path = output_path
        self.json_dict = dict(info={}, licenses=[], images =[], point_cloud =[], annotation=[], categories = [])

    def update_info(self, year=None, version=None, descriptor=None, contributor=None, url=None, date_created=None):

        self.json_dict.info = dict(year=year, version=version, descriptor=descriptor, contributor=contributor, url=url, date_created=date_created)

    def add_images(self, image_id, file_name, width, height, date_captured, license, coco_url, flickr_url, mask=None):
        self.json_dict.images.append(dict(id=image_id, file_name=file_name, width=width, height=height, date_captured=date_captured, license=license, coco_url=coco_url, flickr_url=flickr_url, mask=mask))

    def add_point_cloud(self, point_cloud_id, file_name, mask=None):
        self.json_dict["point_cloud"].append(dict(id=point_cloud_id, file_name=file_name, mask=mask))

    def add_annotation(self, annotation_id, image_id, category_id, segmentation, area, bbox, iscrowd):
        self.json_dict.annotation.append(dict(id=annotation_id, image_id=image_id, category_id=category_id, segmentation=segmentation, area=area, bbox=bbox, iscrowd=iscrowd))

    def add_categories(self, category_id, name, supercategory):
        self.json_dict.categories.append(dict(id=category_id, name=name, supercategory=supercategory))

    def write_json(self):
        with open(self.output_path, 'w') as json_file:
            json.dump(self.json_dict, json_file, indent=4)



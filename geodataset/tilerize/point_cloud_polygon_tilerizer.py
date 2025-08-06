from pathlib import Path
from typing import List, cast, Optional

import geopandas as gpd
import pandas as pd
from shapely import box
from tqdm import tqdm
from laspy import CopcReader, PointFormat, ExtraBytesParams
import laspy
import rasterio
from rasterio.windows import Window
import numpy as np
from shapely.affinity import translate
import cv2
import re
from pyproj import CRS

from geodataset.aoi import AOIFromPackageConfig, AOIGeneratorConfig, AOIConfig
from geodataset.aoi.aoi_base import DEFAULT_AOI_NAME
from geodataset.aoi.aoi_from_package import AOIFromPackageForPolygons
from geodataset.geodata import Raster
from geodataset.geodata.point_cloud import PointCloudTileMetadata, PointCloudTileMetadataCollection
from geodataset.labels.base_labels import PolygonLabels
from geodataset.utils import CocoNameConvention, COCOGenerator, strip_all_extensions_and_path
from geodataset.utils.file_name_conventions import AoiGeoPackageConvention, PointCloudTileNameConvention, validate_and_convert_product_name
from geodataset.labels import RasterPolygonLabels


class PointCloudPolygonTilerizer:
    """
    This class is designed to create individual point cloud tiles for each polygon in a GeoPackages/GeoJson/GeoDataFrame associated with a raster.

    Parameters
    ----------
    point_cloud_path : str or pathlib.Path
        Path to the point cloud (.laz, .las...).
    labels_path : str or pathlib.Path or None
        Path to the labels. Supported formats are: .gpkg, .geojson, .shp, .xml, .csv.
    output_path : str or pathlib.Path
        Path to parent folder where to save the point cloud tiles and associated labels.
    tile_size : int
        All polygon tiles will have the same size (tile_size, tile_size).
    labels_gdf: geopandas.GeoDataFrame, optional
        A GeoDataFrame containing the labels. If provided, labels_path must be None.
    global_aoi : str or pathlib.Path or geopandas.GeoDataFrame, optional
        Path to the global AOI file, or directly a GeoDataFrame.
        If provided, only the tiles intersecting this AOI will be kept, even if some tiles are inside one of the aois
        in aois_config (if AOIFromPackageConfig).

        This parameter can be really useful to create a kfold dataset in association with an AOIGeneratorConfig config like this:

        aois_config = AOIGeneratorConfig(aois={
                'zone1': {'percentage': 0.2, 'position': 1, 'actual_name': f'train{kfold_id}'},
                'zone2': {'percentage': 0.2, 'position': 2, 'actual_name': f'train{kfold_id}'},
                'zone3': {'percentage': 0.2, 'position': 3, 'actual_name': f'valid{kfold_id}'},
                'zone4': {'percentage': 0.2, 'position': 4, 'actual_name': f'train{kfold_id}'},
                'zone5': {'percentage': 0.2, 'position': 5, 'actual_name': f'train{kfold_id}'}
            },
            aoi_type='band'
        )
    aois_config : :class:`~geodataset.aoi.AOIGeneratorConfig` or :class:`~geodataset.aoi.AOIFromPackageConfig` or None
        An instance of AOIConfig to use, or None if all tiles should be kept in a DEFAULT_AOI_NAME AOI.
    ground_resolution : float, optional
        The ground resolution in meter per pixel desired when loading the raster.
        Only one of ground_resolution and scale_factor can be set at the same time.
    scale_factor : float, optional
        Scale factor for rescaling the data (change pixel resolution).
        Only one of ground_resolution and scale_factor can be set at the same time.
    output_name_suffix : str, optional
        Suffix to add to the output file names.
    geopackage_layer_name : str, optional
        The name of the layer in the geopackage file to use as labels. Only used if the labels_path is a .gpkg, .geojson
        or .shp file. Only useful when the labels geopackage file contains multiple layers.
    main_label_category_column_name : str, optional
        The name of the column in the labels file that contains the main category of the labels.
    other_labels_attributes_column_names : list of str, optional
        The names of the columns in the labels file that contains other attributes of the labels, which should be kept
        as a dictionary in the COCO annotations data.
    temp_dir : str or pathlib.Path
        Temporary directory to store the resampled Raster, if it is too big to fit in memory.
    """

    unique_polygon_id_column_name = 'polygon_id_geodataset'

    def __init__(self,
                 point_cloud_path: str or Path,
                 labels_path: str or Path or None,
                 output_path: str or Path,
                 tile_size: int,
                 reference_raster_path: str or Path,
                 labels_gdf: gpd.GeoDataFrame = None,
                 global_aoi: str or Path or gpd.GeoDataFrame = None,
                 aois_config: Optional[AOIConfig] = None,
                 ground_resolution: float = None,
                 scale_factor: float = None,
                 downsample_voxel_size: float = None,
                 output_name_suffix: str = None,
                 geopackage_layer_name: str = None,
                 main_label_category_column_name: str = None,
                 other_labels_attributes_column_names: List[str] = None,
                 temp_dir: str or Path = './tmp'):

        self.point_cloud_path = Path(point_cloud_path)
        self.product_name = validate_and_convert_product_name(
            strip_all_extensions_and_path(self.point_cloud_path)
        )
        self.labels_path = Path(labels_path) if labels_path is not None else None
        self.tile_size = tile_size
        self.global_aoi = global_aoi
        self.aois_config = aois_config
        self.scale_factor = scale_factor
        self.ground_resolution = ground_resolution
        self.downsample_voxel_size = downsample_voxel_size
        self.reference_raster_path = reference_raster_path
        self.output_name_suffix = output_name_suffix
        self.temp_dir = Path(temp_dir)

        self._check_parameters()
        self._check_aois_config()

        self.raster = self._load_raster()
        self.labels = self._load_labels(
            labels_gdf=labels_gdf,
            geopackage_layer_name=geopackage_layer_name,
            main_label_category_column_name=main_label_category_column_name,
            other_labels_attributes_column_names=other_labels_attributes_column_names
        )

        self.labels.geometries_gdf[self.unique_polygon_id_column_name] = range(1, len(self.labels.geometries_gdf) + 1)

        self._align_with_reference_raster()

        self.output_path = Path(output_path) / f"{Path(output_path).name}_pcd"

        self.pc_tiles_folder_path = (
            self.output_path / f"pc_tiles_{self.downsample_voxel_size}"
            if self.downsample_voxel_size
            else self.output_path / "pc_tiles"
        )

        # Generate tiles metadata
        self.tiles_metadata = self._generate_tiles_metadata()

    def _check_parameters(self):
        assert self.point_cloud_path.exists(), \
            f"Point cloud file not found at {self.point_cloud_path}."
        assert isinstance(self.tile_size, int) and self.tile_size > 0, \
            "The tile size must be and integer greater than 0."
        assert not (self.ground_resolution and self.scale_factor), \
            "Both a ground_resolution and a scale_factor were provided. Please only specify one."

    def _check_aois_config(self):
        if self.aois_config is None:
            print("PointCloudPolygonTilerizer: No AOIs configuration provided."
                  f"Defaulting to a single '{DEFAULT_AOI_NAME}' AOI for all polygons.")
            self.aois_config = AOIGeneratorConfig(aois={DEFAULT_AOI_NAME: {'percentage': 1.0, 'position': 1}}, aoi_type='band')
        elif isinstance(self.aois_config, AOIGeneratorConfig) and not self.aois_config.aois: # Empty AOIGeneratorConfig
            print("PointCloudPolygonTilerizer: Empty AOIGeneratorConfig.aois."
                  f"Defaulting to a single '{DEFAULT_AOI_NAME}' AOI.")
            self.aois_config = AOIGeneratorConfig(aois={DEFAULT_AOI_NAME: {'percentage': 1.0, 'position': 1}}, aoi_type='band')
        else:
            self.aois_config = self.aois_config

    def _align_with_reference_raster(self):
        """Align point cloud tiles with an existing raster"""
        with rasterio.open(self.reference_raster_path) as src:
            self.transform = src.transform
            self.point_cloud_crs = src.crs

    def _load_labels(self,
                     labels_gdf: gpd.GeoDataFrame,
                     geopackage_layer_name: str or None,
                     main_label_category_column_name: str or None,
                     other_labels_attributes_column_names: List[str] or None):

        if self.labels_path and labels_gdf:
            raise ValueError("You can't provide both a labels_path and a labels_gdf.")
        elif self.labels_path:
            labels = RasterPolygonLabels(
                path=self.labels_path,
                associated_raster=self.raster,
                geopackage_layer_name=geopackage_layer_name,
                main_label_category_column_name=main_label_category_column_name,
                other_labels_attributes_column_names=other_labels_attributes_column_names
            )
        elif labels_gdf is not None:
            labels = RasterPolygonLabels(
                path=None,
                labels_gdf=labels_gdf,
                associated_raster=self.raster,
                main_label_category_column_name=main_label_category_column_name,
                other_labels_attributes_column_names=other_labels_attributes_column_names
            )
        else:
            raise ValueError("RasterPolygonTilerizer: You must"
                             "provide either a labels_path or a labels_gdf.")
        return labels

    def _generate_tiles_metadata(self):
        """Generate tile metadata for all tiles covering the point cloud"""
        name_convention = PointCloudTileNameConvention()
        tiles_metadata = []

        aois_polygons, _ = self._generate_aois_polygons()

        print(f"Number of items in aois_polygons: {len(aois_polygons)}")
        
        for aoi, polygons in aois_polygons.items():
            print(f"Processing AOI: {aoi} with {len(polygons)} polygons.")
            for _, polygon_row in tqdm(polygons.iterrows(),
                                       f"Generating polygon tiles for AOI {aoi}...",
                                       total=len(polygons)):

                polygon = polygon_row['geometry']
                polygon_id = polygon_row[self.unique_polygon_id_column_name]

                x_centroid, y_centroid = polygon.centroid.coords[0]
                x_centroid, y_centroid = int(x_centroid), int(y_centroid)
                binary_mask = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)
                mask_box = box(x_centroid - 0.5 * self.tile_size,
                       y_centroid - 0.5 * self.tile_size,
                       x_centroid + 0.5 * self.tile_size,
                       y_centroid + 0.5 * self.tile_size)
                # Making sure the polygon is valid
                polygon = polygon.buffer(0)
                polygon_intersection = mask_box.intersection(polygon)
                translated_polygon_intersection = translate(
                    polygon_intersection,
                    xoff=-mask_box.bounds[0],
                    yoff=-mask_box.bounds[1]
                )
                # Making sure the polygon is a Polygon and not a GeometryCollection (GeometryCollection can happen on Raster edges due to intersection)
                # Use an empty Polygon as the default value
                if translated_polygon_intersection.geom_type != 'Polygon':
                    # Get the Polygon with the largest area
                    translated_polygon_intersection = max(translated_polygon_intersection.geoms, key=lambda x: x.area, default=Polygon())
                # Ensure the result has an exterior before accessing its coordinates
                if not translated_polygon_intersection.is_empty:
                    contours = np.array(translated_polygon_intersection.exterior.coords).reshape((-1, 1, 2)).astype(np.int32)
                else:
                    # Handle the case when the intersection is empty (e.g., set contours to an empty array)
                    contours = np.array([])

                # Check if contours is not empty before calling cv2.fillPoly
                if contours.size > 0:
                    cv2.fillPoly(binary_mask, [contours], 1)
                else:
                    # Handle the case where there are no contours (e.g., skip filling the polygon)
                    pass

                # Getting the pixels from the raster
                mask_bounds = mask_box.bounds
                mask_bounds = [int(x) for x in mask_bounds]

                # Handling bounds outside the data
                start_row = mask_bounds[1]
                end_row = mask_bounds[3]
                start_col = mask_bounds[0]
                end_col = mask_bounds[2]

                if len(self.raster.data.shape) > 2:
                    import ipdb; ipdb.set_trace()
                # Padding to tile_size if necessary
                pre_row_pad = max(0, -mask_bounds[1])
                post_row_pad = max(0, mask_bounds[3] - self.raster.data.shape[0])
                pre_col_pad = max(0, -mask_bounds[0])
                post_col_pad = max(0, mask_bounds[2] - self.raster.data.shape[1])

                if self.tile_size - (end_row - start_row + pre_row_pad + post_row_pad) == 1:
                    post_row_pad += 1
                if self.tile_size - (end_col - start_col + pre_col_pad + post_col_pad) == 1:
                    post_col_pad += 1

                # Padding the mask to the final tile size
                start_row = max(0, start_row - pre_row_pad)
                end_row = min(self.raster.data.shape[0], end_row + post_row_pad)
                start_col = max(0, start_col - pre_col_pad)
                end_col = min(self.raster.data.shape[1], end_col + post_col_pad)

                # Creating the PolygonTile, with the appropriate metadata
                window = Window(
                    col_off=start_col,
                    row_off=start_row,
                    width=end_col - start_col,
                    height=end_row - start_row
                )


                left, bottom, right, top = rasterio.windows.bounds(window, transform=self.transform)

                x_bound = [left, right]
                y_bound = [bottom, top]
                
                #start_row = y_centroid - int(0.5 * self.tile_size)
                #start_col = x_centroid - int(0.5 * self.tile_size)

                #x, y = self.transform * (start_col, start_row)

                # Define tile bounds
                #x_bound = [x, x + self.tile_side_length_x]
                #y_bound = [y, y + self.tile_side_length_y]

                #x_left, y_top = rasterio.transform.xy(
                #    self.transform, start_row, start_col, offset='ul'
                #)
                #x_right, y_bottom = rasterio.transform.xy(
                #    self.transform, end_row, end_col, offset='lr'
                #)

                #x_bound = [x_left, x_right]
                #y_bound = [y_bottom, y_top]

                tile_name = name_convention.create_name(
                    product_name=self.product_name,
                    row=start_row,
                    col=start_col,
                    voxel_size=self.downsample_voxel_size,
                    ground_resolution=self.ground_resolution,
                    scale_factor=self.scale_factor,
                    aoi=aoi,
                    width=self.tile_size,
                    height=self.tile_size,
                    tile_id=polygon_id,
                )

                tile_md = PointCloudTileMetadata(
                    tile_id=polygon_id,
                    x_bound=x_bound,
                    y_bound=y_bound,
                    crs=self.point_cloud_crs,
                    tile_name=tile_name,
                    height=self.tile_size,
                    width=self.tile_size,
                    aoi=aoi,
                    ground_resolution=self.ground_resolution,
                    scale_factor=self.scale_factor
                )
                
                tiles_metadata.append(tile_md)
        
        print(f"Generated {len(tiles_metadata)} tiles for point cloud {self.point_cloud_path.name}.")

        del self.raster  # Clear the raster reference to free memory
        
        return PointCloudTileMetadataCollection(
            tiles_metadata, product_name=self.product_name
        )

    def _generate_aois_polygons(self):
        polygons_for_aoi_processing = self.labels.geometries_gdf.copy()

        if isinstance(self.aois_config, AOIGeneratorConfig):
            is_simple_generate_all = (
                len(self.aois_config.aois) == 1 and
                next(iter(self.aois_config.aois.keys())).lower() in [DEFAULT_AOI_NAME.lower(), "infer"]
            )
            if is_simple_generate_all:
                aoi_name = next(iter(self.aois_config.aois.keys())) # Get the actual name ('train', 'infer', etc.)
                aois_polygons = {aoi_name: polygons_for_aoi_processing}
                raster_extent_poly = box(0, 0, self.raster.metadata['width'], self.raster.metadata['height'])
                aois_gdf = gpd.GeoDataFrame(
                    {'geometry': [raster_extent_poly], 'aoi': [aoi_name]},
                    crs=None     # the AOI is in pixel coordinates, so no CRS is needed
                )
            else:
                # This case remains for truly complex AOIGeneratorConfigs not meant for polygon tiling
                raise NotImplementedError(
                    "Complex AOIGeneratorConfig (defining multiple spatial splits not named 'all'/'infer') "
                    "is not directly supported for RasterPolygonTilerizer when the goal is to tile existing polygons. "
                    "Use AOIFromPackageConfig to assign polygons to predefined vector AOIs, "
                    "or ensure AOIGeneratorConfig defines a single AOI named 'all'/'infer'/something of your choosing."
                )
        elif isinstance(self.aois_config, AOIFromPackageConfig):
            aoi_engine = AOIFromPackageForPolygons(
                labels=self.labels,
                global_aoi=self.global_aoi,
                aois_config=self.aois_config,
                associated_raster=self.raster,
            )
            aois_polygons, aois_gdf = aoi_engine.get_aoi_polygons()
        else:
            # This should not be reached if __init__ correctly sets self.aois_config
            raise Exception(f'Internal error: aois_config type unsupported or None: {type(self.aois_config)}')
        return aois_polygons, aois_gdf

    def _load_raster(self):
        raster = Raster(path=self.reference_raster_path,
                        output_name_suffix=self.output_name_suffix,
                        ground_resolution=self.ground_resolution,
                        scale_factor=self.scale_factor,
                        temp_dir=self.temp_dir)
        return raster

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
        mins = np.array([tile_md.min_x, tile_md.min_y])
        maxs = np.array([tile_md.max_x, tile_md.max_y])
        b = laspy.copc.Bounds(mins=mins, maxs=maxs)
        data = reader.query(b)
        return data
    
    def generate_coco_dataset(self):
        new_tile_md_list = []
        with CopcReader.open(self.point_cloud_path) as reader:
            for tile_md in tqdm(self.tiles_metadata, desc="Processing tiles"):
                # Query points within this tile
                las = self.query_tile(tile_md, reader)
                if len(las) == 0:
                    raise ValueError(
                        f"No points found in tile {tile_md.tile_name}. "
                        "This might indicate an issue with the AOI filtering or point cloud data."
                    )
                # Apply downsampling
                las_downsampled = self._voxel_downsample_centroid(las, self.downsample_voxel_size, reader.header)
                del las
                # Save the tile only if it has points
                if len(las_downsampled) > 0:
                    new_tile_md_list.append(tile_md)
                    # Save the tile to disk
                    output_dir = self.pc_tiles_folder_path / f"{tile_md.aoi if tile_md.aoi else 'noaoi'}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_file = output_dir / tile_md.tile_name
                    las_downsampled.write(f"{output_file}")
                    del las_downsampled
                else:
                    raise ValueError(
                        f"Tile {tile_md.tile_name} has no points after processing. "
                        "This might indicate an issue with the AOI filtering or point cloud data."
                    )
        print(f"Finished tilerizing. Generated {len(new_tile_md_list)} tiles with points.")

    def _voxel_downsample_centroid(self, las: laspy.LasData, voxel_size: float, header) -> laspy.LasData:
        coords = np.vstack((las.x, las.y, las.z)).T  # shape: (N, 3)
        if coords.shape[0] == 0:
            raise ValueError("No points in the tile to downsample.")
        # Compute voxel indices
        voxel_indices = np.floor(coords / voxel_size).astype(np.int64)
        # Get unique voxel IDs and mapping from points to voxel ID
        unique_voxels, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
        num_voxels = unique_voxels.shape[0]
        original_point_count = coords.shape[0]
        # Print compression ratio
        compression_ratio = original_point_count / num_voxels if num_voxels > 0 else 0
        print(f"Point cloud compressed from {original_point_count} to {num_voxels} points "
            f"(compression ratio: {compression_ratio:.2f}x)")
        # Prepare containers
        centroids = np.zeros((num_voxels, 3), dtype=np.float64)
        # Compute centroids using NumPy bincount
        for i in range(3):
            centroids[:, i] = np.bincount(inverse_indices, weights=coords[:, i]) / np.bincount(inverse_indices)
        # Determine continuous fields to average
        valid_fields = set(las.point_format.dimension_names)
        continuous_fields = [
            "red", "green", "blue",
            "normal x", "normal y", "normal z",
        ]
        continuous_fields = [f for f in continuous_fields if f in valid_fields]
        # Prepare dictionary for continuous fields
        continuous_arrays = {}
        for field in continuous_fields:
            values = np.asarray(las[field])
            dtype = values.dtype
            sums = np.bincount(inverse_indices, weights=values.astype(np.float64))
            counts = np.bincount(inverse_indices)
            averaged = (sums / counts).astype(dtype)
            continuous_arrays[field] = averaged
        # Start from minimal base format (just X, Y, Z + RGB)
        point_format = PointFormat(2)  # Format 2 includes X,Y,Z + RGB
        # Create new header from scratch
        new_header = laspy.LasHeader(version=header.version, point_format=point_format)
        # Add extra dimensions for any non-standard fields
        extra_dims = []
        for field in continuous_fields:
            if field not in ('red', 'green', 'blue'):
                dtype = las[field].dtype
                extra_dims.append(ExtraBytesParams(name=field, type=dtype))
        if extra_dims:
            new_header.add_extra_dims(extra_dims)
        # Create minimal LasData
        new_las = laspy.LasData(new_header)
            
        crs = header.parse_crs()
        crs = crs.source_crs
        epsg_code = crs.to_epsg()
        crs = CRS.from_epsg(epsg_code)
        new_las.header.add_crs(crs)
        
        # Assign coordinates (centroids)
        new_las.x = centroids[:, 0]
        new_las.y = centroids[:, 1]
        new_las.z = centroids[:, 2]
        # Assign other averaged fields
        for f, arr in continuous_arrays.items():
            new_las[f] = arr
        return new_las
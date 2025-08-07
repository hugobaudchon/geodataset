from pathlib import Path
from typing import List, Optional, Union

import geopandas as gpd
import laspy
import numpy as np
import rasterio
from laspy import CopcReader, ExtraBytesParams, PointFormat
from pyproj import CRS
from rasterio.windows import Window
from shapely import box
from tqdm import tqdm

from geodataset.aoi import AOIConfig, AOIFromPackageConfig, AOIGeneratorConfig
from geodataset.aoi.aoi_base import DEFAULT_AOI_NAME
from geodataset.aoi.aoi_from_package import AOIFromPackageForPolygons
from geodataset.geodata import Raster
from geodataset.geodata.point_cloud import PointCloudTileMetadata, PointCloudTileMetadataCollection
from geodataset.labels import RasterPolygonLabels
from geodataset.utils import strip_all_extensions_and_path
from geodataset.utils.file_name_conventions import PointCloudTileNameConvention, validate_and_convert_product_name


class PointCloudPolygonTilerizer:
    """
    Generates individual point cloud tiles based on polygons in a given GeoPackage, GeoJSON, or GeoDataFrame.

    Parameters
    ----------
    point_cloud_path : str or Path
        Path to the input point cloud file (.laz, .las, etc.).
    labels_path : str or Path or None
        Path to the labels file (.gpkg, .geojson, .shp, .xml, .csv).
    output_path : str or Path
        Path to the parent directory where output will be saved.
    tile_size : int
        Output tiles will be square with shape (tile_size, tile_size).
    reference_raster_path : str or Path
        Path to the raster used for alignment.
    labels_gdf : geopandas.GeoDataFrame, optional
        GeoDataFrame containing the label geometries. Use instead of `labels_path`.
    global_aoi : str or Path or geopandas.GeoDataFrame, optional
        Global area of interest (AOI). Can be a path or a GeoDataFrame.
    aois_config : AOIGeneratorConfig or AOIFromPackageConfig or None
        Configuration for how to divide polygons into AOIs.
    ground_resolution : float, optional
        Desired resolution in meters per pixel (mutually exclusive with scale_factor).
    scale_factor : float, optional
        Rescale factor for raster data (mutually exclusive with ground_resolution).
    downsample_voxel_size : float, optional
        Voxel size for downsampling the point cloud.
    output_name_suffix : str, optional
        Optional suffix for output file names.
    geopackage_layer_name : str, optional
        Layer name for label extraction from .gpkg or .geojson files.
    main_label_category_column_name : str, optional
        Main category column name in label data.
    other_labels_attributes_column_names : list of str, optional
        Other label attributes to retain in COCO annotations.
    temp_dir : str or Path, optional
        Path to a temporary directory for storing large resampled rasters.
    """

    def __init__(self,
                 point_cloud_path: Union[str, Path],
                 labels_path: Optional[Union[str, Path]],
                 output_path: Union[str, Path],
                 tile_size: int,
                 reference_raster_path: Union[str, Path],
                 labels_gdf: Optional[gpd.GeoDataFrame] = None,
                 global_aoi: Optional[Union[str, Path, gpd.GeoDataFrame]] = None,
                 aois_config: Optional[AOIConfig] = None,
                 ground_resolution: Optional[float] = None,
                 scale_factor: Optional[float] = None,
                 downsample_voxel_size: Optional[float] = None,
                 output_name_suffix: Optional[str] = None,
                 geopackage_layer_name: Optional[str] = None,
                 main_label_category_column_name: Optional[str] = None,
                 other_labels_attributes_column_names: Optional[List[str]] = None,
                 temp_dir: Union[str, Path] = './tmp'):
        self.unique_polygon_id_column_name = 'polygon_id_geodataset'
        # File paths
        self.point_cloud_path = Path(point_cloud_path)
        self.reference_raster_path = Path(reference_raster_path)
        self.labels_path = Path(labels_path) if labels_path else None
        self.output_path = Path(output_path) / f"{Path(output_path).name}_pcd"
        self.temp_dir = Path(temp_dir)
        # Configs & Params
        self.tile_size = tile_size
        self.global_aoi = global_aoi
        self.aois_config = aois_config
        self.ground_resolution = ground_resolution
        self.scale_factor = scale_factor
        self.downsample_voxel_size = downsample_voxel_size
        self.output_name_suffix = output_name_suffix
        # Product name from filename
        self.product_name = validate_and_convert_product_name(
            strip_all_extensions_and_path(self.point_cloud_path)
        )
        # Parameter checks
        self._check_parameters()
        self._check_aois_config()
        # Load raster and labels
        self.raster = self._load_raster()
        self.labels = self._load_labels(
            labels_gdf=labels_gdf,
            geopackage_layer_name=geopackage_layer_name,
            main_label_category_column_name=main_label_category_column_name,
            other_labels_attributes_column_names=other_labels_attributes_column_names
        )
        # Assign unique polygon IDs
        self.labels.geometries_gdf[self.unique_polygon_id_column_name] = range(1, len(self.labels.geometries_gdf) + 1)
        # Align labels with raster
        self._align_with_reference_raster()
        # Output folder for tiles
        self.pc_tiles_folder_path = (
            self.output_path / f"pc_tiles_{self.downsample_voxel_size}"
            if self.downsample_voxel_size
            else self.output_path / "pc_tiles"
        )
        # Generate tiles metadata
        self.tiles_metadata = self._generate_tiles_metadata()
        
    def generate_coco_dataset(self) -> None:
        """
        Generate individual point cloud tiles from the full point cloud,
        apply voxel downsampling, and write them to disk in LAS format.
        """
        valid_tiles: List[PointCloudTileMetadata] = []
        with CopcReader.open(self.point_cloud_path) as reader:
            for tile_md in tqdm(self.tiles_metadata, desc="Processing tiles"):
                # Query points in the tile bounds
                las = self._query_tile(tile_md, reader)
                if len(las.x) == 0:
                    raise ValueError(
                        f"[{tile_md.tile_name}] No points found in tile. "
                        "Check AOI filtering or point cloud coverage."
                    )
                # Apply voxel downsampling
                las_downsampled = self._voxel_downsample_centroid(
                    las, self.downsample_voxel_size, reader.header
                )
                if len(las_downsampled.x) == 0:
                    raise ValueError(
                        f"[{tile_md.tile_name}] No points remain after downsampling. "
                        "Consider adjusting voxel size or verifying input data."
                    )
                valid_tiles.append(tile_md)
                # Write downsampled LAS to disk
                aoi_name = tile_md.aoi or "noaoi"
                output_dir = self.pc_tiles_folder_path / aoi_name
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / tile_md.tile_name
                las_downsampled.write(str(output_file))
        print(f"[Done] Generated {len(valid_tiles)} tiles with valid point data.")

    def _check_parameters(self) -> None:
        assert self.point_cloud_path.exists(), \
            f"Point cloud file not found at {self.point_cloud_path}."
        assert isinstance(self.tile_size, int) and self.tile_size > 0, \
            "The tile size must be and integer greater than 0."
        assert not (self.ground_resolution and self.scale_factor), \
            "Both a ground_resolution and a scale_factor were provided. Please only specify one."

    def _check_aois_config(self) -> None:
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

    def _load_raster(self) -> Raster:
        """Load the reference raster with configured parameters."""
        return Raster(
            path=self.reference_raster_path,
            output_name_suffix=self.output_name_suffix,
            ground_resolution=self.ground_resolution,
            scale_factor=self.scale_factor,
            temp_dir=self.temp_dir,
        )

    def _load_labels(self,
                 labels_gdf: Optional[gpd.GeoDataFrame],
                 geopackage_layer_name: Optional[str],
                 main_label_category_column_name: Optional[str],
                 other_labels_attributes_column_names: Optional[List[str]]) -> RasterPolygonLabels:
        """
        Loads label polygons either from a file or a provided GeoDataFrame.

        Raises
        ------
        ValueError:
            If both `labels_path` and `labels_gdf` are provided or neither is provided.
        """

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

    def _align_with_reference_raster(self) -> None:
        """
        Aligns the point cloud tiles with the spatial reference and transform of the given raster.
        """
        with rasterio.open(self.reference_raster_path) as src:
            self.transform = src.transform
            self.point_cloud_crs = src.crs

    def _generate_tiles_metadata(self) -> PointCloudTileMetadataCollection:
        """Generate metadata for all point cloud tiles aligned with polygons."""
        name_convention = PointCloudTileNameConvention()
        tiles_metadata: List[PointCloudTileMetadata] = []
        aois_polygons, _ = self._generate_aois_polygons()
        for aoi, polygons in aois_polygons.items():
            for _, row in tqdm(polygons.iterrows(),
                                       desc=f"Generating tiles for AOI '{aoi}'",
                                       total=len(polygons)):
                tile_md = self._process_polygon_tile(row, aoi, name_convention)
                if tile_md is not None:
                    tiles_metadata.append(tile_md)
        print(f"Generated {len(tiles_metadata)} tiles for point cloud {self.point_cloud_path.name}.")
        del self.raster  # Free memory
        return PointCloudTileMetadataCollection(tiles_metadata, product_name=self.product_name)

    def _process_polygon_tile(self,
                          polygon_row: gpd.GeoSeries,
                          aoi: str,
                          name_convention: PointCloudTileNameConvention) -> Optional[PointCloudTileMetadata]:
        """Process a single polygon tile and return its metadata, or None if invalid."""
        polygon = polygon_row['geometry'].buffer(0)
        polygon_id = polygon_row[self.unique_polygon_id_column_name]
        x_centroid, y_centroid = map(int, polygon.centroid.coords[0])
        mask_box = box(
            x_centroid - 0.5 * self.tile_size,
            y_centroid - 0.5 * self.tile_size,
            x_centroid + 0.5 * self.tile_size,
            y_centroid + 0.5 * self.tile_size
        )
        intersection = mask_box.intersection(polygon)
        if intersection.is_empty:
            return None
         # Compute raster bounds and padding
        bounds = [int(coord) for coord in mask_box.bounds]
        start_col, start_row, end_col, end_row = bounds[0], bounds[1], bounds[2], bounds[3]
        pre_row_pad, post_row_pad = max(0, -start_row), max(0, end_row - self.raster.data.shape[0])
        pre_col_pad, post_col_pad = max(0, -start_col), max(0, end_col - self.raster.data.shape[1])
        # Adjust for off-by-one edge cases
        if (end_row - start_row + pre_row_pad + post_row_pad) == self.tile_size - 1:
            post_row_pad += 1
        if (end_col - start_col + pre_col_pad + post_col_pad) == self.tile_size - 1:
            post_col_pad += 1
        # Clip to raster shape
        start_row = max(0, start_row - pre_row_pad)
        end_row = min(self.raster.data.shape[0], end_row + post_row_pad)
        start_col = max(0, start_col - pre_col_pad)
        end_col = min(self.raster.data.shape[1], end_col + post_col_pad)
        window = Window(
            col_off=start_col,
            row_off=start_row,
            width=end_col - start_col,
            height=end_row - start_row
        )
        left, bottom, right, top = rasterio.windows.bounds(window, transform=self.transform)
        x_bound, y_bound = [left, right], [bottom, top]
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
        return PointCloudTileMetadata(
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

    def _generate_aois_polygons(self) -> tuple[dict[str, gpd.GeoDataFrame], gpd.GeoDataFrame]:
        """Generate AOI polygons"""
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

    def _query_tile(
        self,
        tile_md: PointCloudTileMetadata,
        reader: laspy.CopcReader
    ) -> laspy.LasData:
        """
        Extract points within a tile boundary from the point cloud.

        Parameters
        ----------
        tile_md : PointCloudTileMetadata
            Metadata describing the tile boundaries.
        reader : laspy.CopcReader
            An open COPC reader for the point cloud.

        Returns
        -------
        laspy.LasData
            Points within the tile boundary.
        """
        bounds = laspy.copc.Bounds(
            mins=np.array([tile_md.min_x, tile_md.min_y]),
            maxs=np.array([tile_md.max_x, tile_md.max_y])
        )
        return reader.query(bounds)

    def _voxel_downsample_centroid(
        self, 
        las: laspy.LasData, 
        voxel_size: float, 
        header: laspy.LasHeader
    ) -> laspy.LasData:
        """
        Downsamples a point cloud using voxel grid centroid sampling.

        For each voxel of size `voxel_size`, all points within the voxel
        are averaged to compute a single representative point. Continuous
        fields (e.g., RGB or normals) are also averaged if available.

        Parameters
        ----------
        las : laspy.LasData
            The input LAS point cloud.
        voxel_size : float
            The size of the voxel grid used for downsampling.
        header : laspy.LasHeader
            The header from the original LAS file, used to construct the output.

        Returns
        -------
        laspy.LasData
            The downsampled point cloud as a new LasData object.
        """
        coords = np.vstack((las.x, las.y, las.z)).T  # (N, 3)
        if coords.shape[0] == 0:
            raise ValueError("No points in the tile to downsample.")
        # Compute voxel indices
        voxel_indices = np.floor(coords / voxel_size).astype(np.int64)
        unique_voxels, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
        num_voxels = unique_voxels.shape[0]
        original_point_count = coords.shape[0]
        compression_ratio = original_point_count / num_voxels if num_voxels > 0 else 0
        print(f"Point cloud compressed from {original_point_count} to {num_voxels} points "
            f"(compression ratio: {compression_ratio:.2f}x)")
        # Compute voxel centroids
        centroids = np.vstack([
            np.bincount(inverse_indices, weights=coords[:, i]) / np.bincount(inverse_indices)
            for i in range(3)
        ]).T  # shape: (num_voxels, 3)
        # Continuous fields to average
        valid_fields = set(las.point_format.dimension_names)
        continuous_fields = [
            f for f in ["red", "green", "blue", "normal x", "normal y", "normal z"]
            if f in valid_fields
        ]
        def _average_by_voxel(values: np.ndarray) -> np.ndarray:
            float_values = np.asarray(values).astype(np.float64)
            return (np.bincount(inverse_indices, weights=float_values) /
                    np.bincount(inverse_indices)).astype(values.dtype)
        # Compute averaged attributes
        averaged_fields = {
            field: _average_by_voxel(las[field])
            for field in continuous_fields
        }
        # Define new point format (Format 2: X, Y, Z, RGB)
        point_format = PointFormat(2)
        new_header = laspy.LasHeader(version=header.version, point_format=point_format)
        # Add extra dimensions for non-standard fields
        extra_dims = [
            ExtraBytesParams(name=field, type=las[field].dtype)
            for field in continuous_fields
            if field not in ('red', 'green', 'blue')
        ]
        if extra_dims:
            new_header.add_extra_dims(extra_dims)
        # Create new LasData and assign EPSG
        new_las = laspy.LasData(new_header)
        crs = header.parse_crs().source_crs
        new_las.header.add_crs(CRS.from_epsg(crs.to_epsg()))
        # Assign centroid coordinates
        new_las.x, new_las.y, new_las.z = centroids[:, 0], centroids[:, 1], centroids[:, 2]
        # Assign averaged fields
        for field, values in averaged_fields.items():
            new_las[field] = values
        return new_las
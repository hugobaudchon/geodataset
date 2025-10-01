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
from geodataset.geodata.point_cloud import (PointCloudTileMetadata,
                                            PointCloudTileMetadataCollection)
from geodataset.labels import RasterPolygonLabels
from geodataset.utils import strip_all_extensions_and_path
from geodataset.utils.file_name_conventions import (
    PointCloudTileNameConvention, validate_and_convert_product_name)


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
    downsample_num_points : int, optional
        Number of points to retain after downsampling.
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

    def __init__(
        self,
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
        downsample_num_points: Optional[int] = None,
        downsample_method: Optional[str] = None,
        output_name_suffix: Optional[str] = None,
        geopackage_layer_name: Optional[str] = None,
        main_label_category_column_name: Optional[str] = None,
        other_labels_attributes_column_names: Optional[List[str]] = None,
        temp_dir: Union[str, Path] = "./tmp",
    ):
        self.unique_polygon_id_column_name = "polygon_id_geodataset"
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
        self.downsample_num_points = downsample_num_points
        self.downsample_method = downsample_method
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
            other_labels_attributes_column_names=other_labels_attributes_column_names,
        )
        # Assign unique polygon IDs
        self.labels.geometries_gdf[self.unique_polygon_id_column_name] = range(
            1, len(self.labels.geometries_gdf) + 1
        )
        # Align labels with raster
        self._align_with_reference_raster()
        # Output folder for tiles
        if self.downsample_voxel_size is not None and self.downsample_voxel_size > 0:
            assert (
                self.downsample_num_points is None
            ), "Both downsample_voxel_size and downsample_num_points were provided. Please only specify one."
            self.pc_tiles_folder_path = (
                self.output_path / f"pc_tiles_{self.downsample_voxel_size}"
            )
        elif self.downsample_num_points is not None and self.downsample_num_points > 0:
            assert (
                self.downsample_voxel_size is None
            ), "Both downsample_voxel_size and downsample_num_points were provided. Please only specify one."
            if self.downsample_method == "fps":
                self.pc_tiles_folder_path = (
                    self.output_path / f"pc_tiles_{self.downsample_num_points}"
                )
            elif self.downsample_method == "random":
                self.pc_tiles_folder_path = (
                    self.output_path / f"pc_tiles_{self.downsample_num_points}_random"
                )
            else:
                raise ValueError(
                    f"Invalid downsample_method: {self.downsample_method}. Supported methods are 'fps' and 'random'."
                )
        else:
            self.pc_tiles_folder_path = self.output_path / "pc_tiles"
        # Generate tiles metadata
        self.tiles_metadata = self._generate_tiles_metadata()

    def generate_coco_dataset(self) -> None:
        """Generate individual point cloud tiles from the full point cloud, apply voxel downsampling, and write them to
        disk in LAS format."""
        valid_tiles: List[PointCloudTileMetadata] = []
        with CopcReader.open(self.point_cloud_path) as reader:
            parsed_crs = reader.header.parse_crs()
            self.epsg_code = self._extract_epsg(parsed_crs)
            if self.epsg_code is None:
                raise ValueError("Could not determine EPSG code from CRS")
            raster_epsg = self.raster_crs.to_epsg()
            if self.epsg_code != raster_epsg:
                raise ValueError(
                    f"Point cloud CRS EPSG code {self.epsg_code} does not match raster CRS EPSG code "
                    f"{raster_epsg}. Ensure both datasets are in the same CRS."
                )
            for tile_md in tqdm(self.tiles_metadata, desc="Processing tiles"):
                # Query points in the tile bounds
                las = self._query_tile(tile_md, reader)
                if len(las.x) == 0:
                    raise ValueError(
                        f"[{tile_md.tile_name}] No points found in tile. "
                        "Check AOI filtering or point cloud coverage."
                    )
                # Apply voxel downsampling
                if (
                    self.downsample_voxel_size is not None
                    and self.downsample_voxel_size > 0
                ):
                    las_downsampled = self._voxel_downsample_centroid(
                        las, self.downsample_voxel_size, reader.header
                    )
                elif (
                    self.downsample_num_points is not None
                    and self.downsample_num_points > 0
                ):
                    if self.downsample_method == "random":
                        las_downsampled = self._random_downsample(
                            las, self.downsample_num_points, reader.header
                        )
                    elif self.downsample_method == "fps":
                        las_downsampled = self._fps_downsample(
                            las, self.downsample_num_points, reader.header
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
        assert (
            self.point_cloud_path.exists()
        ), f"Point cloud file not found at {self.point_cloud_path}."
        assert (
            isinstance(self.tile_size, int) and self.tile_size > 0
        ), "The tile size must be and integer greater than 0."
        assert not (
            self.ground_resolution and self.scale_factor
        ), "Both a ground_resolution and a scale_factor were provided. Please only specify one."

    def _check_aois_config(self) -> None:
        if self.aois_config is None:
            print(
                "PointCloudPolygonTilerizer: No AOIs configuration provided."
                f"Defaulting to a single '{DEFAULT_AOI_NAME}' AOI for all polygons."
            )
            self.aois_config = AOIGeneratorConfig(
                aois={DEFAULT_AOI_NAME: {"percentage": 1.0, "position": 1}},
                aoi_type="band",
            )
        # Empty AOIGeneratorConfig
        elif (
            isinstance(self.aois_config, AOIGeneratorConfig)
            and not self.aois_config.aois
        ):
            print(
                "PointCloudPolygonTilerizer: Empty AOIGeneratorConfig.aois."
                f"Defaulting to a single '{DEFAULT_AOI_NAME}' AOI."
            )
            self.aois_config = AOIGeneratorConfig(
                aois={DEFAULT_AOI_NAME: {"percentage": 1.0, "position": 1}},
                aoi_type="band",
            )
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

    def _load_labels(
        self,
        labels_gdf: Optional[gpd.GeoDataFrame],
        geopackage_layer_name: Optional[str],
        main_label_category_column_name: Optional[str],
        other_labels_attributes_column_names: Optional[List[str]],
    ) -> RasterPolygonLabels:
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
                other_labels_attributes_column_names=other_labels_attributes_column_names,
            )
        elif labels_gdf is not None:
            labels = RasterPolygonLabels(
                path=None,
                labels_gdf=labels_gdf,
                associated_raster=self.raster,
                main_label_category_column_name=main_label_category_column_name,
                other_labels_attributes_column_names=other_labels_attributes_column_names,
            )
        else:
            raise ValueError(
                "RasterPolygonTilerizer: You must"
                "provide either a labels_path or a labels_gdf."
            )
        return labels

    def _align_with_reference_raster(self) -> None:
        """Aligns the point cloud tiles with the spatial reference and transform of the given raster."""
        with rasterio.open(self.reference_raster_path) as src:
            self.transform = src.transform
            self.raster_crs = src.crs

    def _generate_tiles_metadata(self) -> PointCloudTileMetadataCollection:
        """Generate metadata for all point cloud tiles aligned with polygons."""
        name_convention = PointCloudTileNameConvention()
        tiles_metadata: List[PointCloudTileMetadata] = []
        aois_polygons, _ = self._generate_aois_polygons()
        for aoi, polygons in aois_polygons.items():
            for _, row in tqdm(
                polygons.iterrows(),
                desc=f"Generating tiles for AOI '{aoi}'",
                total=len(polygons),
            ):
                tile_md = self._process_polygon_tile(row, aoi, name_convention)
                if tile_md is not None:
                    tiles_metadata.append(tile_md)
        print(
            f"Generated {len(tiles_metadata)} tiles for point cloud {self.point_cloud_path.name}."
        )
        del self.raster  # Free memory
        return PointCloudTileMetadataCollection(
            tiles_metadata, product_name=self.product_name
        )

    def _process_polygon_tile(
        self,
        polygon_row: gpd.GeoSeries,
        aoi: str,
        name_convention: PointCloudTileNameConvention,
    ) -> Optional[PointCloudTileMetadata]:
        """Process a single polygon tile and return its metadata, or None if invalid."""
        polygon = polygon_row["geometry"].buffer(0)
        polygon_id = polygon_row[self.unique_polygon_id_column_name]
        x_centroid, y_centroid = map(int, polygon.centroid.coords[0])
        mask_box = box(
            x_centroid - 0.5 * self.tile_size,
            y_centroid - 0.5 * self.tile_size,
            x_centroid + 0.5 * self.tile_size,
            y_centroid + 0.5 * self.tile_size,
        )
        intersection = mask_box.intersection(polygon)
        if intersection.is_empty:
            return None
        # Compute raster bounds and padding
        bounds = [int(coord) for coord in mask_box.bounds]
        start_col, start_row, end_col, end_row = (
            bounds[0],
            bounds[1],
            bounds[2],
            bounds[3],
        )
        pre_row_pad, post_row_pad = max(0, -start_row), max(
            0, end_row - self.raster.data.shape[0]
        )
        pre_col_pad, post_col_pad = max(0, -start_col), max(
            0, end_col - self.raster.data.shape[1]
        )
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
            height=end_row - start_row,
        )
        """
        Window = Window( col_off=mask_box.bounds[0], row_off=mask_box.bounds[1], width=self.tile_size,

        height=self.tile_size )
        """

        left, bottom, right, top = rasterio.windows.bounds(
            window, transform=self.transform
        )
        x_bound, y_bound = [left, right], [bottom, top]
        tile_name = name_convention.create_name(
            product_name=self.product_name,
            row=start_row,
            col=start_col,
            voxel_size=self.downsample_voxel_size,
            num_points=self.downsample_num_points,
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
            crs=self.raster_crs,
            tile_name=tile_name,
            height=self.tile_size,
            width=self.tile_size,
            aoi=aoi,
            ground_resolution=self.ground_resolution,
            scale_factor=self.scale_factor,
        )

    def _generate_aois_polygons(
        self,
    ) -> tuple[dict[str, gpd.GeoDataFrame], gpd.GeoDataFrame]:
        """Generate AOI polygons."""
        polygons_for_aoi_processing = self.labels.geometries_gdf.copy()

        if isinstance(self.aois_config, AOIGeneratorConfig):
            is_simple_generate_all = len(self.aois_config.aois) == 1 and next(
                iter(self.aois_config.aois.keys())
            ).lower() in [DEFAULT_AOI_NAME.lower(), "infer"]
            if is_simple_generate_all:
                # Get the actual name ('train', 'infer', etc.)
                aoi_name = next(iter(self.aois_config.aois.keys()))
                aois_polygons = {aoi_name: polygons_for_aoi_processing}
                raster_extent_poly = box(
                    0, 0, self.raster.metadata["width"], self.raster.metadata["height"]
                )
                aois_gdf = gpd.GeoDataFrame(
                    {"geometry": [raster_extent_poly], "aoi": [aoi_name]},
                    crs=None,  # the AOI is in pixel coordinates, so no CRS is needed
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
            raise Exception(
                f"Internal error: aois_config type unsupported or None: {type(self.aois_config)}"
            )
        return aois_polygons, aois_gdf

    def _extract_epsg(self, parsed_crs) -> Optional[int]:
        """
        Return EPSG code from a laspy parsed CRS (checks source_crs, sub_crs_list, then fallback).

        None if unresolved.
        """
        """
        source_crs = getattr(parsed_crs, "source_crs", None)
        if source_crs is not None:
            epsg = source_crs.to_epsg()
            if epsg is not None:
                return epsg
        for crs in getattr(parsed_crs, "sub_crs_list", []):
            base_crs = getattr(crs, "source_crs", crs)
            epsg = base_crs.to_epsg()
            if epsg is not None:
                return epsg
        try:
            return parsed_crs.to_epsg()
        except Exception:
            return None
        """
        try:
            epsg = parsed_crs.to_epsg()
            if epsg is not None:
                return epsg
        except Exception:
            pass
        source_crs = getattr(parsed_crs, "source_crs", None)
        if source_crs is not None:
            epsg = source_crs.to_epsg()
            if epsg is not None:
                return epsg
        for crs in getattr(parsed_crs, "sub_crs_list", []):
            base_crs = getattr(crs, "source_crs", crs)
            epsg = base_crs.to_epsg()
            if epsg is not None:
                return epsg
        raise ValueError("Could not extract EPSG code from parsed CRS.")

    def _query_tile(
        self, tile_md: PointCloudTileMetadata, reader: laspy.CopcReader
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
            maxs=np.array([tile_md.max_x, tile_md.max_y]),
        )
        return reader.query(bounds)

    def _voxel_downsample_centroid(
        self, las: laspy.LasData, voxel_size: float, header: laspy.LasHeader
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
        unique_voxels, inverse_indices = np.unique(
            voxel_indices, axis=0, return_inverse=True
        )
        num_voxels = unique_voxels.shape[0]
        original_point_count = coords.shape[0]
        original_point_count / num_voxels if num_voxels > 0 else 0
        # print(f"Point cloud compressed from {original_point_count} to {num_voxels} points "
        #    f"(compression ratio: {compression_ratio:.2f}x)")
        # Compute voxel centroids
        centroids = np.vstack(
            [
                np.bincount(inverse_indices, weights=coords[:, i])
                / np.bincount(inverse_indices)
                for i in range(3)
            ]
        ).T  # shape: (num_voxels, 3)
        # Continuous fields to average
        valid_fields = set(las.point_format.dimension_names)
        continuous_fields = [
            f
            for f in ["red", "green", "blue", "normal x", "normal y", "normal z"]
            if f in valid_fields
        ]

        def _average_by_voxel(values: np.ndarray) -> np.ndarray:
            float_values = np.asarray(values).astype(np.float64)
            return (
                np.bincount(inverse_indices, weights=float_values)
                / np.bincount(inverse_indices)
            ).astype(values.dtype)

        # Compute averaged attributes
        averaged_fields = {
            field: _average_by_voxel(las[field]) for field in continuous_fields
        }
        # Define new point format (Format 2: X, Y, Z, RGB)
        point_format = PointFormat(2)
        new_header = laspy.LasHeader(version=header.version, point_format=point_format)
        # Add extra dimensions for non-standard fields
        extra_dims = [
            ExtraBytesParams(name=field, type=las[field].dtype)
            for field in continuous_fields
            if field not in ("red", "green", "blue")
        ]
        if extra_dims:
            new_header.add_extra_dims(extra_dims)
        # Create new LasData and assign EPSG
        new_las = laspy.LasData(new_header)
        new_las.header.add_crs(CRS.from_epsg(self.epsg_code))
        # Assign centroid coordinates
        new_las.x, new_las.y, new_las.z = (
            centroids[:, 0],
            centroids[:, 1],
            centroids[:, 2],
        )
        # Assign averaged fields
        for field, values in averaged_fields.items():
            new_las[field] = values
        return new_las

    def _random_downsample(
        self, las: laspy.LasData, num_samples: int, header: laspy.LasHeader
    ) -> laspy.LasData:
        """
        Downsamples a point cloud using Random Sampling.

        Parameters
        ----------
        las : laspy.LasData
            The input LAS point cloud.
        num_samples : int
            The number of points to sample.
        header : laspy.LasHeader
            Header from the original LAS file.

        Returns
        -------
        laspy.LasData
            The downsampled point cloud as a new LasData object.
        """
        coords = np.vstack((las.x, las.y, las.z)).T  # (N, 3)
        if coords.shape[0] == 0:
            raise ValueError("No points in the tile to downsample.")

        N = coords.shape[0]
        if num_samples > N:
            print(
                f"[INFO] Requested {num_samples} samples, but point cloud only has {N} points. "
                f"Duplicating points to reach {num_samples}."
            )
            # Include each point once
            sampled_idx = list(np.arange(N))
            extra = num_samples - N
            # Add random duplicates
            extra_idx = np.random.choice(N, size=extra, replace=True)
            sampled_idx.extend(extra_idx)
            # Shuffle so duplicates aren’t clustered
            sampled_idx = np.array(sampled_idx)
            np.random.shuffle(sampled_idx)
        else:
            # Sample without replacement
            sampled_idx = np.random.choice(N, size=num_samples, replace=False)

        # Select sampled points
        sampled_coords = coords[sampled_idx]

        # Continuous fields to keep
        valid_fields = set(las.point_format.dimension_names)
        continuous_fields = [
            f
            for f in ["red", "green", "blue", "normal x", "normal y", "normal z"]
            if f in valid_fields
        ]

        # Define new point format (Format 2: X, Y, Z, RGB)
        point_format = PointFormat(2)
        new_header = laspy.LasHeader(version=header.version, point_format=point_format)

        # Add extra dimensions for non-standard fields
        extra_dims = [
            ExtraBytesParams(name=field, type=las[field].dtype)
            for field in continuous_fields
            if field not in ("red", "green", "blue")
        ]
        if extra_dims:
            new_header.add_extra_dims(extra_dims)

        # Create new LasData and assign EPSG
        new_las = laspy.LasData(new_header)
        new_las.header.add_crs(CRS.from_epsg(self.epsg_code))

        # Assign coordinates
        new_las.x, new_las.y, new_las.z = (
            sampled_coords[:, 0],
            sampled_coords[:, 1],
            sampled_coords[:, 2],
        )

        # Assign continuous fields
        for field in continuous_fields:
            new_las[field] = las[field][sampled_idx]

        return new_las

    def _fps_downsample(
        self, las: laspy.LasData, num_samples: int, header: laspy.LasHeader
    ) -> laspy.LasData:
        """
        Downsamples a point cloud using Farthest Point Sampling (FPS).

        Parameters
        ----------
        las : laspy.LasData
            The input LAS point cloud.
        num_samples : int
            The number of points to sample.
        header : laspy.LasHeader
            Header from the original LAS file.

        Returns
        -------
        laspy.LasData
            The downsampled point cloud as a new LasData object.
        """
        coords = np.vstack((las.x, las.y, las.z)).T  # (N, 3)
        if coords.shape[0] == 0:
            raise ValueError("No points in the tile to downsample.")

        N = coords.shape[0]
        if num_samples > N:
            print(
                f"[INFO] Requested {num_samples} samples, but point cloud only has {N} points. "
                f"Duplicating points to reach {num_samples}."
            )
            # First, include each point once
            sampled_idx = list(np.arange(N))
            extra = num_samples - N
            if extra > N:
                raise ValueError(
                    f"Cannot duplicate points to reach {num_samples} samples "
                    f"when point cloud only has {N} points."
                )
            if extra > 0:
                # Choose extra points without replacement, so each gets duplicated at most once
                extra_idx = np.random.choice(N, size=extra, replace=False)
                sampled_idx.extend(extra_idx)
            # Shuffle so duplicates are not clustered
            sampled_idx = np.array(sampled_idx)
            np.random.shuffle(sampled_idx)
        else:
            # Initialize FPS
            sampled_idx = np.zeros(num_samples, dtype=np.int64)
            distances = np.full(N, np.inf)

            # Pick first point randomly
            sampled_idx[0] = np.random.randint(0, N)

            for i in range(1, num_samples):
                # Compute distance from last added point to all points
                diff = coords - coords[sampled_idx[i - 1]]
                dist_sq = np.sum(diff**2, axis=1)
                distances = np.minimum(distances, dist_sq)
                sampled_idx[i] = np.argmax(distances)

        # Select sampled points
        fps_coords = coords[sampled_idx]

        # Continuous fields to keep
        valid_fields = set(las.point_format.dimension_names)
        continuous_fields = [
            f
            for f in ["red", "green", "blue", "normal x", "normal y", "normal z"]
            if f in valid_fields
        ]

        # Define new point format (Format 2: X, Y, Z, RGB)
        point_format = PointFormat(2)
        new_header = laspy.LasHeader(version=header.version, point_format=point_format)

        # Add extra dimensions for non-standard fields
        extra_dims = [
            ExtraBytesParams(name=field, type=las[field].dtype)
            for field in continuous_fields
            if field not in ("red", "green", "blue")
        ]
        if extra_dims:
            new_header.add_extra_dims(extra_dims)

        # Create new LasData and assign EPSG
        new_las = laspy.LasData(new_header)
        new_las.header.add_crs(CRS.from_epsg(self.epsg_code))

        # Assign coordinates
        new_las.x, new_las.y, new_las.z = (
            fps_coords[:, 0],
            fps_coords[:, 1],
            fps_coords[:, 2],
        )

        # Assign continuous fields
        for field in continuous_fields:
            new_las[field] = las[field][sampled_idx]

        return new_las

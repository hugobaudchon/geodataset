from pathlib import Path
from typing import List, cast, Optional

import geopandas as gpd
import pandas as pd
from shapely import box
from tqdm import tqdm
from laspy import CopcReader
import laspy
import rasterio
import numpy as np
import open3d as o3d

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
        If use_variable_tile_size is set to True, then this parameter defines the maximum size of the tiles in pixels (tile_size, tile_size).
        If use_variable_tile_size is set to False, all polygon tiles will have the same size (tile_size, tile_size).
    use_variable_tile_size: bool
        Whether to use variable tile size. If True, the tile size will match the size of the polygon,
         with a buffer defined by variable_tile_size_pixel_buffer.
    variable_tile_size_pixel_buffer: int or None
        If use_variable_tile_size is True, this parameter defines the pixel buffer to add around the polygon when creating the tile.
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
    use_rle_for_labels : bool, optional
        Whether to use RLE encoding for the labels. If False, the labels will be saved as polygons.
    min_intersection_ratio : float, optional
        When finding the associated polygon labels to a tile, this ratio will specify the minimal required intersection
        ratio (intersecting_polygon_area / polygon_area) between a candidate polygon and the tile in order to keep this
        polygon as a label for that tile.
    geopackage_layer_name : str, optional
        The name of the layer in the geopackage file to use as labels. Only used if the labels_path is a .gpkg, .geojson
        or .shp file. Only useful when the labels geopackage file contains multiple layers.
    main_label_category_column_name : str, optional
        The name of the column in the labels file that contains the main category of the labels.
    other_labels_attributes_column_names : list of str, optional
        The names of the columns in the labels file that contains other attributes of the labels, which should be kept
        as a dictionary in the COCO annotations data.
    coco_n_workers : int, optional
        Number of workers to use when generating the COCO dataset.
        Useful when use_rle_for_labels=True as it is quite slow.
    coco_categories_list : list of dict, optional
        A list of category dictionaries in COCO format.

        If provided, category ids for the annotations in the final COCO file
        will be determined by matching the category name (defined by 'main_label_category_column_name' parameter) of
        each polygon with the categories names in coco_categories_list.

        If a polygon has a category that is not in this list, its category_id will be set to None in its COCO annotation.

        If 'main_label_category_column_name' is not provided, but 'coco_categories_list' is a single
        coco category dictionary, then it will be used for all annotations automatically.

        If 'coco_categories_list' is None, the categories ids will be automatically generated from the
        unique categories found in the 'main_label_category_column_name' column.

        .. raw:: html

            <br>

        To assign a category_id to a polygon, the code will check the 'name' and 'other_names' fields of the categories.

        .. raw:: html

            <br>

        **IMPORTANT**: It is strongly advised to provide this list if you want to have consistent category ids across
        multiple COCO datasets.

        .. raw:: html

            <br>

        Exemple of 2 categories, one being the parent of the other::

            [{
                "id": 1,
                "name": "Pinaceae",
                "other_names": [],
                "supercategory": null
            },
            {
                "id": 2,
                "name": "Picea",
                "other_names": ["PIGL", "PIMA", "PIRU"],
                "supercategory": 1
            }]
    keep_dims : List[str], optional
        List of dimensions to keep.
    tile_batch_size : int, optional
        The number of polygon tiles to process in a single batch when saving them to disk.
    temp_dir : str or pathlib.Path
        Temporary directory to store the resampled Raster, if it is too big to fit in memory.
    """

    unique_polygon_id_column_name = 'polygon_id_geodataset'

    def __init__(self,
                 point_cloud_path: str or Path,
                 labels_path: str or Path or None,
                 output_path: str or Path,
                 tile_size: int,
                 use_variable_tile_size: bool,
                 variable_tile_size_pixel_buffer: int or None,
                 labels_gdf: gpd.GeoDataFrame = None,
                 global_aoi: str or Path or gpd.GeoDataFrame = None,
                 aois_config: Optional[AOIConfig] = None,
                 ground_resolution: float = None,
                 scale_factor: float = None,
                 downsample_voxel_size: float = None,
                 reference_raster_path: str or Path = None,
                 output_name_suffix: str = None,
                 use_rle_for_labels: bool = True,
                 min_intersection_ratio: float = 0.5,
                 geopackage_layer_name: str = None,
                 main_label_category_column_name: str = None,
                 other_labels_attributes_column_names: List[str] = None,
                 coco_n_workers: int = 5,
                 coco_categories_list: list[dict] = None,
                 keep_dims: List[str] = None,
                 tile_batch_size: int = 1000,
                 temp_dir: str or Path = './tmp'):

        self.point_cloud_path = Path(point_cloud_path)
        self.product_name = validate_and_convert_product_name(
            strip_all_extensions_and_path(self.point_cloud_path)
        )
        self.labels_path = Path(labels_path) if labels_path is not None else None
        self.tile_size = tile_size
        self.use_variable_tile_size = use_variable_tile_size
        self.variable_tile_size_pixel_buffer = variable_tile_size_pixel_buffer
        self.global_aoi = global_aoi
        self.aois_config = aois_config
        self.scale_factor = scale_factor
        self.ground_resolution = ground_resolution
        self.keep_dims = keep_dims if keep_dims is not None else "ALL"
        self.downsample_voxel_size = downsample_voxel_size
        self.reference_raster_path = reference_raster_path
        self.output_name_suffix = output_name_suffix
        self.use_rle_for_labels = use_rle_for_labels
        self.min_intersection_ratio = min_intersection_ratio
        self.coco_n_workers = coco_n_workers
        self.coco_categories_list = coco_categories_list
        self.tile_batch_size = tile_batch_size
        self.temp_dir = Path(temp_dir)

        self._check_parameters()
        self._check_aois_config()

        self.category_to_id_map = self.create_category_to_id_map()
        """self.labels = self._load_labels(
            geopackage_layer=geopackage_layer_name,
            main_label_category_column_name=main_label_category_column_name,
            other_labels_attributes_column_names=other_labels_attributes_column_names,
            keep_categories=list(self.category_to_id_map.keys()),
        )"""
        self.raster = self._load_raster()
        self.labels = self._load_labels(
            labels_gdf=labels_gdf,
            geopackage_layer_name=geopackage_layer_name,
            main_label_category_column_name=main_label_category_column_name,
            other_labels_attributes_column_names=other_labels_attributes_column_names
        )

        self.labels.geometries_gdf[self.unique_polygon_id_column_name] = range(1, len(self.labels.geometries_gdf) + 1)

        # Calculate the real world distance equivalent to tile_size pixels
        if self.reference_raster_path:
            self._align_with_reference_raster()
        else:
            self._calculate_tile_side_length()

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
        if self.use_variable_tile_size:
            assert self.variable_tile_size_pixel_buffer, "A variable_tile_size_pixel_buffer must be provided if use_variable_tile_size is True."

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

    def create_category_to_id_map(self):
        """Create a mapping from category names to IDs"""
        category_id_map = {}
        if self.coco_categories_list:
            for category_dict in self.coco_categories_list:
                category_id_map[category_dict["name"]] = category_dict["id"]
                for name in category_dict.get("other_names", []):
                    category_id_map[name] = category_dict["id"]

        return category_id_map


    """ def _load_labels(
        self,
        geopackage_layer=None,
        main_label_category_column_name="labels",
        other_labels_attributes_column_names=None,
        keep_categories=None,
    ) -> PolygonLabels:
        labels = PolygonLabels(
            path=self.labels_path,
            geopackage_layer_name=geopackage_layer,
            main_label_category_column_name=main_label_category_column_name,
            other_labels_attributes_column_names=other_labels_attributes_column_names,
            keep_categories=keep_categories,
        )
        return labels """

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
        tile_id = 0

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

                start_row = y_centroid - int(0.5 * self.tile_size)
                start_col = x_centroid - int(0.5 * self.tile_size)

                x = self.min_x + self.x_resolution * start_col
                y = self.min_y + self.y_resolution * start_row

                # Define tile bounds
                x_bound = [x, x + self.tile_side_length_x]
                y_bound = [y, y + self.tile_side_length_y]

                tile_name = name_convention.create_name(
                    product_name=self.product_name,
                    row=start_row,
                    col=start_col,
                    voxel_size=self.downsample_voxel_size,
                    ground_resolution=self.ground_resolution,
                    scale_factor=self.scale_factor,
                    aoi=aoi,
                )

                """
                print(f"x_bound: {x_bound}, y_bound: {y_bound}")
                print(f"start_row: {start_row}, start_col: {start_col}")
                print(f"product_name: {self.product_name}, ")
                print(f"Creating tile {tile_name} at bounds {x_bound}, {y_bound} for polygon ID {polygon_id}.")
                raise ValueError("Tile creation failed.")
                """

                tile_md = PointCloudTileMetadata(
                    x_bound=x_bound,
                    y_bound=y_bound,
                    crs=self.point_cloud_crs,
                    tile_name=tile_name,
                    tile_id=tile_id,
                    height=self.tile_size,
                    width=self.tile_size,
                    ground_resolution=self.ground_resolution,
                    scale_factor=self.scale_factor
                )
                
                tiles_metadata.append(tile_md)
                tile_id += 1
        
        print(f"Generated {len(tiles_metadata)} tiles for point cloud {self.point_cloud_path.name}.")
        
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
        del self.raster  # Clear the raster reference to free memory
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
        Generate point cloud polygon tiles for the entire dataset.
        
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
        
        return o3d.t.geometry.PointCloud(map_to_tensors)
    
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

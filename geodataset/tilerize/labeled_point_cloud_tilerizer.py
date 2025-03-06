import warnings
from typing import List, Union

import geopandas as gpd
import numpy as np
import open3d as o3d
from laspy import CopcReader
from shapely.geometry import Point
from tqdm import tqdm

from geodataset.aoi import AOIFromPackageConfig
from geodataset.aoi.aoi_base import AOIBaseFromGeoFileInCRS
from geodataset.utils import PointCloudCOCOGenerator, strip_all_extensions_and_path
from geodataset.labels.base_labels import PolygonLabels
from geodataset.geodata.point_cloud import PointCloudTileMetadataCollection
from geodataset.tilerize.point_cloud_tilerizer import PointCloudTilerizer
from geodataset.utils.file_name_conventions import PointCloudCocoNameConvention, validate_and_convert_product_name
from pathlib import Path


class LabeledPointCloudTilerizer(PointCloudTilerizer):
    """
    Class to tilerize a labeled point cloud dataset aligned with a raster tiling grid.
    
    This class extends PointCloudTilerizer to handle point cloud data with labels,
    producing tiles that perfectly align with raster tiles.

    Parameters
    ----------
    point_cloud_path : Union[str, Path]
        Path to the point cloud file.
    labels_path : Union[str, Path]
        Path to the labels file.
    output_path : Union[str, Path]
        Path to output folder.
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
    reference_raster_path : Union[str, Path, None], optional
        Path to a reference raster for perfect alignment.
    min_intersection_ratio : float, optional
        Minimum ratio of intersection for label matching.
    ignore_tiles_without_labels : bool, optional
        Whether to ignore tiles without labels.
    geopackage_layer_name : str, optional
        Layer name in geopackage file.
    main_label_category_column_name : str, optional
        Column name for the main label category.
    other_labels_attributes_column_names : List[str], optional
        Column names for other label attributes.
    use_rle_for_labels : bool, optional
        Whether to use RLE encoding for labels.
    coco_n_workers : int, optional
        Number of workers for COCO dataset generation.
    coco_categories_list : List[dict], optional
        List of categories for COCO dataset.
    keep_dims : List[str], optional
        List of dimensions to keep.
    downsample_voxel_size : float, optional
        Voxel size for downsampling.
    verbose : bool, optional
        Enable verbose output.
    force : bool, optional
        Force tilerization even with many tiles.
    """

    def __init__(
        self,
        point_cloud_path: Union[str, Path],
        labels_path: Union[str, Path],
        output_path: Union[str, Path],
        tile_size: int,
        tile_overlap: float,
        ground_resolution: float = None,
        scale_factor: float = None,
        aois_config: Union[AOIFromPackageConfig, None] = None,
        reference_raster_path: Union[str, Path, None] = None,
        min_intersection_ratio: float = 0.9,
        ignore_tiles_without_labels: bool = False,
        geopackage_layer_name: str = None,
        main_label_category_column_name: str = "Label",
        other_labels_attributes_column_names: List[str] = None,
        use_rle_for_labels: bool = True,
        coco_n_workers: int = 1,
        coco_categories_list: List[dict] = None,
        keep_dims: List[str] = None,
        downsample_voxel_size: float = None,
        verbose: bool = False,
        force: bool = False,
    ) -> None:
        # Initialize the parent class first
        super().__init__(
            point_cloud_path=point_cloud_path,
            output_path=output_path,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            ground_resolution=ground_resolution,
            scale_factor=scale_factor,
            aois_config=aois_config,
            reference_raster_path=reference_raster_path,
            keep_dims=keep_dims,
            downsample_voxel_size=downsample_voxel_size,
            verbose=verbose,
            max_tile=50000,
            force=force,
        )
        
        # Store label-specific parameters
        self.label_path = Path(labels_path)
        self.min_intersection_ratio = min_intersection_ratio
        self.ignore_tiles_without_labels = ignore_tiles_without_labels
        self.main_label_category_column_name = main_label_category_column_name
        self.other_labels_attributes_column_names = other_labels_attributes_column_names
        self.use_rle_for_labels = use_rle_for_labels
        self.coco_n_workers = coco_n_workers
        self.coco_categories_list = coco_categories_list
        
        # Label management
        self.category_to_id_map = self.create_category_to_id_map()
        self.labels = self.load_labels(
            geopackage_layer=geopackage_layer_name,
            main_label_category_column_name=main_label_category_column_name,
            other_labels_attributes_column_names=other_labels_attributes_column_names,
            keep_categories=list(self.category_to_id_map.keys()),
        )
        
        # Validate RLE requirements
        if self.use_rle_for_labels:
            assert (
                self.tiles_metadata.height is not None and 
                self.tiles_metadata.width is not None
            ), "Height and width of tiles are required for RLE encoding"
            
        # Process AOIs for labels
        self.aoi_tiles = None
        self.aoi_labels = None
            
    def create_category_to_id_map(self):
        """Create a mapping from category names to IDs"""
        category_id_map = {}
        if self.coco_categories_list:
            for category_dict in self.coco_categories_list:
                category_id_map[category_dict["name"]] = category_dict["id"]
                for name in category_dict.get("other_names", []):
                    category_id_map[name] = category_dict["id"]

        return category_id_map

    def load_labels(
        self,
        geopackage_layer=None,
        main_label_category_column_name="labels",
        other_labels_attributes_column_names=None,
        keep_categories=None,
    ) -> PolygonLabels:
        """Load labels from file"""
        labels = PolygonLabels(
            path=self.label_path,
            geopackage_layer_name=geopackage_layer,
            main_label_category_column_name=main_label_category_column_name,
            other_labels_attributes_column_names=other_labels_attributes_column_names,
            keep_categories=keep_categories,
        )
        return labels

    def tilerize(self):
        """Generate labeled point cloud tiles"""
        self._generate_labels()
        super()._tilerize()  # Use parent _tilerize method
        super().plot_aois()
        return self.tiles_metadata

    def _generate_labels(self):
        """Generate label associations for tiles"""
        self.aoi_tiles, self.aoi_labels = self._get_aoi_labels()
        
        coco_paths, categories_coco, category_to_id_map = self._generate_coco_dataset()
        self.category_to_id_map.update(category_to_id_map)
        self.category_to_id_map[np.nan] = np.nan
        
        return coco_paths, categories_coco, category_to_id_map
        
    def _generate_coco_dataset(self):
        """Generate COCO dataset from labels"""
        coco_paths = {}
        for aoi in self.aoi_labels:
            if aoi == "all" and len(self.aoi_labels.keys()) > 1:
                # Skip 'all' if we have specific AOIs
                continue

            labels = self.aoi_labels[aoi]
            if len(labels) == 0:
                print(f"No tiles found for AOI {aoi}. Skipping...")
                continue

            # Get tiles for this AOI
            aoi_tiles_metadata = PointCloudTileMetadataCollection(
                [tile for tile in self.tiles_metadata if tile.tile_id in self.aoi_tiles[aoi]]
            )

            # Extract polygons and attributes
            polygons = [x["geometry"].to_list() for x in labels]
            
            # Handle categories
            categories_list = None
            if self.main_label_category_column_name:
                categories_list = [
                    x[self.main_label_category_column_name].to_list()
                    for x in labels
                ]

            # Handle other attributes
            other_attributes_dict_list = None
            if self.other_labels_attributes_column_names:
                other_attributes_dict_list = [
                    {
                        attribute: label[attribute].to_list()
                        for attribute in self.other_labels_attributes_column_names
                    }
                    for label in labels
                ]
                other_attributes_dict_list = [
                    [
                        {k: d[k][i] for k in d}
                        for i in range(len(next(iter(d.values()))))
                    ]
                    for d in other_attributes_dict_list
                ]

            # Generate COCO output path
            coco_output_file_path = (
                self.output_path
                / PointCloudCocoNameConvention.create_name(
                    product_name=self.tiles_metadata.product_name,
                    ground_resolution=self.ground_resolution,
                    scale_factor=self.scale_factor,
                    voxel_size=self.downsample_voxel_size,
                    fold=aoi,
                )
            )

            # Create COCO generator
            coco_generator = PointCloudCOCOGenerator(
                description=f"Point cloud dataset for {self.tiles_metadata.product_name} with fold {aoi}",
                tiles_metadata=aoi_tiles_metadata,
                polygons=polygons,
                scores=None,
                categories=categories_list,
                other_attributes=other_attributes_dict_list,
                output_path=coco_output_file_path,
                use_rle_for_labels=self.use_rle_for_labels,
                n_workers=self.coco_n_workers,
                coco_categories_list=self.coco_categories_list,
            )

            # Generate COCO dataset
            categories_coco, category_to_id_map = coco_generator.generate_coco()
            coco_paths[aoi] = coco_output_file_path

        return coco_paths, categories_coco, category_to_id_map
        
    def _find_tiles_associated_labels(self):
        """Find which labels are associated with each tile"""
        print("Finding labels associated with each tile...")

        # Create GeoDataFrame with tile geometries
        tile_ids = [tile.tile_id for tile in self.tiles_metadata]
        tile_geometries = [tile.geometry.values[0][0] for tile in self.tiles_metadata]
        tiles_gdf = gpd.GeoDataFrame(
            data={"tile_id": tile_ids, "geometry": tile_geometries},
            crs=self.tiles_metadata.crs
        )

        # Convert labels to tiles CRS
        labels_gdf = self.labels.geometries_gdf
        labels_gdf = labels_gdf.to_crs(tiles_gdf.crs)

        # Calculate intersection areas
        labels_gdf["total_instance_area"] = labels_gdf.geometry.area
        inter_polygons = gpd.overlay(
            tiles_gdf, labels_gdf, how="intersection", keep_geom_type=True
        )
        
        # Calculate intersection ratios
        inter_polygons["instance_area_within_tile"] = inter_polygons.geometry.area
        inter_polygons["intersection_ratio_tiles"] = (
            inter_polygons["instance_area_within_tile"]
            / inter_polygons["total_instance_area"]
        )
        
        # Filter by minimum intersection ratio
        significant_polygons_inter = inter_polygons[
            inter_polygons["intersection_ratio_tiles"] > self.min_intersection_ratio
        ]
        significant_polygons_inter = significant_polygons_inter.reset_index()

        return significant_polygons_inter
        
    def _remove_unlabeled_tiles(self, associated_labels):
        """Remove tiles that don't have any associated labels"""
        if not self.ignore_tiles_without_labels:
            return

        labeled_tiles = []
        for tile in self.tiles_metadata:
            individual_associated_labels = associated_labels[
                associated_labels["tile_id"] == tile.tile_id
            ]
            if len(individual_associated_labels) > 0:
                labeled_tiles.append(tile)

        # Update tiles_metadata with only labeled tiles
        tiles_removed = len(self.tiles_metadata) - len(labeled_tiles)
        if tiles_removed > 0:
            warnings.warn(f"Removed {tiles_removed} tiles without labels")
            self.tiles_metadata = PointCloudTileMetadataCollection(
                labeled_tiles, product_name=self.tiles_metadata.product_name
            )
            
    def _get_aoi_tiles(self):
        """Get the tiles for each AOI"""
        aois_gdf = self.aoi_engine.get_aoi_gdf()
        tiles_gdf = self.tiles_metadata.gdf
        aois_gdf = aois_gdf.to_crs(tiles_gdf.crs)

        # Check if tiles have AOIs assigned
        if 'aoi' not in tiles_gdf.columns or tiles_gdf['aoi'].isnull().all():
            # Remove any existing 'aoi' column to avoid conflicts
            if 'aoi' in tiles_gdf.columns:
                tiles_gdf = tiles_gdf.drop(columns=['aoi'])

            # Find intersections between tiles and AOIs
            intersections = gpd.overlay(tiles_gdf, aois_gdf, how="intersection")
            intersections["intersection_area"] = intersections.geometry.area
            
            # Find the AOI with maximum intersection for each tile
            max_intersection_per_tile = intersections.loc[
                intersections.groupby("tile_id")["intersection_area"].idxmax()
            ]
            
            # Group tiles by AOI
            aois_tiles = (
                max_intersection_per_tile.groupby("aoi")["tile_id"].apply(list).to_dict()
            )
            
            # Update tile AOI attributes
            tile_id_to_aoi = {}
            for aoi, tile_ids in aois_tiles.items():
                for tile_id in tile_ids:
                    tile_id_to_aoi[tile_id] = aoi
                    
            # Update tile metadata with AOI info
            for tile in self.tiles_metadata:
                if tile.tile_id in tile_id_to_aoi:
                    tile.aoi = tile_id_to_aoi[tile.tile_id]
        else:
            # Use existing AOI assignments
            aois_tiles = tiles_gdf.groupby("aoi")["tile_id"].apply(list).to_dict()

        return aois_tiles, aois_gdf
            
    def _get_aoi_labels(self):
        """
        Get labels for each AOI
        
        Returns
        -------
        Tuple[Dict, Dict]
            tiles by AOI, labels by AOI
        """
        # Find which labels are associated with which tiles
        tile_associated_labels = self._find_tiles_associated_labels()

        # Remove tiles without labels if needed
        if self.ignore_tiles_without_labels:
            self._remove_unlabeled_tiles(tile_associated_labels)

        # Get tiles per AOI
        aois_tiles, aois_gdf = self._get_aoi_tiles()

        # Add instance ID to each label
        tile_associated_labels["instance_id"] = tile_associated_labels.index

        # Assign labels to AOIs
        intersected_labels_aois = gpd.overlay(
            tile_associated_labels, aois_gdf, how="intersection", keep_geom_type=False
        )
        
        # Initialize result dictionaries
        final_aoi_labels = {aoi: [] for aoi in list(aois_tiles.keys()) + ["all"]}

        for aoi in aois_tiles:
            tiles_to_remove = []
            for tile_id in aois_tiles[aoi]:
                # Get labels for this tile in this AOI
                labels_crs_coords = intersected_labels_aois[
                    (intersected_labels_aois["tile_id"] == tile_id) & 
                    (intersected_labels_aois["aoi"] == aoi)
                ]
                
                # Get label IDs
                labels_ids = labels_crs_coords["instance_id"].tolist()
                
                # Get all labels for these IDs in this AOI
                labels_tiles_coords = intersected_labels_aois[
                    (intersected_labels_aois.instance_id.isin(labels_ids)) & 
                    (intersected_labels_aois["aoi"] == aoi)
                ]

                # Remove labels with zero area
                labels_tiles_coords = labels_tiles_coords[
                    labels_tiles_coords.geometry.area > 0.0
                ]

                # Remove very small bounding boxes
                labels_tiles_coords = labels_tiles_coords[
                    (
                        labels_tiles_coords.geometry.bounds["maxx"]
                        - labels_tiles_coords.geometry.bounds["minx"]
                    )
                    > 0.5
                ]
                labels_tiles_coords = labels_tiles_coords[
                    (
                        labels_tiles_coords.geometry.bounds["maxy"]
                        - labels_tiles_coords.geometry.bounds["miny"]
                    )
                    > 0.5
                ]

                # Remove tiles without labels if needed
                if self.ignore_tiles_without_labels and len(labels_tiles_coords) == 0:
                    tiles_to_remove.append(tile_id)
                    if self.verbose:
                        print(f"Removing tile {tile_id} from AOI {aoi} as it has no labels")
                    continue

                # Add labels to results
                final_aoi_labels[aoi].append(labels_tiles_coords)
                final_aoi_labels["all"].append(labels_tiles_coords)
            
            # Remove tiles without labels
            for tile_id in tiles_to_remove:
                aois_tiles[aoi].remove(tile_id)

        return aois_tiles, final_aoi_labels

    def _get_tile_labels(self, tile_id, aois_tiles, aoi_labels):
        """Get labels for a specific tile"""
        # Find which AOI the tile belongs to
        aoi = self._get_aoi_from_tile_id(tile_id, aois_tiles)
        if aoi is None:
            return gpd.GeoDataFrame()
            
        # Get the index of the labels for this tile
        ind_aoi = self._get_ind_aoi_tiles(tile_id, aoi, aoi_labels)
        
        # Return the labels if found
        return aoi_labels[aoi][ind_aoi] if ind_aoi is not None else gpd.GeoDataFrame()

    def _get_aoi_from_tile_id(self, tile_id, aois_tiles):
        """Find which AOI a tile belongs to"""
        for aoi, tile_ids in aois_tiles.items():
            if tile_id in tile_ids:
                return aoi
        
        if self.verbose:
            warnings.warn(f"Tile {tile_id} not found in any AOI")
        return None

    def _get_ind_aoi_tiles(self, tile_id, aoi, aoi_labels):
        """Find the index of labels for a tile in an AOI"""
        if aoi:
            for i, labels in enumerate(aoi_labels[aoi]):
                if tile_id in labels["tile_id"].values:
                    return i
        return None

    def _tilerize(self):
        """Generate labeled point cloud tiles"""
        # We'll override the parent's _tilerize to add label attributes to points
        new_tile_md_list = []
        
        with CopcReader.open(self.point_cloud_path) as reader:
            reader_crs = reader.header.parse_crs()
            
            for tile_md in tqdm(self.tiles_metadata, desc="Processing tiles"):
                # Get points for this tile
                data = super().query_tile(tile_md, reader)
                if len(data) == 0:
                    continue
                    
                # Convert to Open3D format
                pcd = super()._laspy_to_o3d(
                    data, 
                    self.keep_dims.copy() if type(self.keep_dims) is not str else self.keep_dims
                )

                # Apply processing steps
                if self.downsample_voxel_size:
                    pcd = super()._downsample_tile(pcd, self.downsample_voxel_size)
                pcd = self._keep_unique_points(pcd)
                pcd = self._remove_points_outside_aoi(pcd, tile_md, reader_crs)
                
                # Get labels for this tile and add them to the point cloud
                tile_labels = self._get_tile_labels(
                    tile_md.tile_id, self.aoi_tiles.copy(), self.aoi_labels.copy()
                )
                
                if not tile_labels.empty:
                    tile_labels = tile_labels.to_crs(reader_crs)
                    pcd = self._add_labels(pcd, tile_labels, reader_crs)
                else:
                    # Add empty labels
                    pcd = self._add_empty_labels(pcd)
                
                new_tile_md_list.append(tile_md)
                
                # Save the tile
                output_dir = self.pc_tiles_folder_path / f"{tile_md.aoi if tile_md.aoi else 'noaoi'}"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / tile_md.tile_name
                
                o3d.t.io.write_point_cloud(str(output_file), pcd)

        print(f"Finished tilerizing. Generated {len(new_tile_md_list)} tiles.")
        self.tiles_metadata = PointCloudTileMetadataCollection(
            new_tile_md_list, product_name=self.tiles_metadata.product_name
        )

    def _add_labels(self, pcd, tile_labels, reader_crs):
        """Add label information to each point"""
        # Convert points to GeoDataFrame
        positions = pcd.point.positions.numpy()
        geopoints = gpd.GeoDataFrame(
            geometry=[Point(x, y, z) for x, y, z in positions],
            crs=reader_crs
        )

        # Convert categories to IDs
        def category_to_id(x):
            if x in self.category_to_id_map:
                return self.category_to_id_map[x]
            return np.nan

        if not tile_labels.empty:
            # Remove level_0 column if present (from previous operations)
            if 'level_0' in tile_labels.columns:
                tile_labels.drop(columns=['level_0'], inplace=True, errors='ignore')

            # Spatial join to find which points are within which polygons
            joined_label = gpd.sjoin(
                geopoints, tile_labels, how="left", predicate="within"
            )

            # Verify points order is preserved
            points_from_joined = np.vstack(
                [point.x, point.y, point.z] 
                for point in joined_label.geometry
            ).T

            if not np.array_equal(positions, points_from_joined):
                raise ValueError("Point order was not preserved in the spatial join")

            # Convert categories to IDs
            if self.main_label_category_column_name and self.main_label_category_column_name in joined_label.columns:
                joined_label[self.main_label_category_column_name] = joined_label[self.main_label_category_column_name].apply(category_to_id)
                semantic_labels = joined_label[self.main_label_category_column_name].values.reshape((-1, 1))
            else:
                semantic_labels = np.full((positions.shape[0], 1), np.nan)

            # Get instance IDs
            if 'instance_id' in joined_label.columns:
                instance_labels = joined_label["instance_id"].values.reshape((-1, 1))
            else:
                instance_labels = np.full((positions.shape[0], 1), np.nan)
        else:
            if self.verbose:
                warnings.warn("No labels found for this tile")
            semantic_labels = np.full((positions.shape[0], 1), np.nan)
            instance_labels = np.full((positions.shape[0], 1), np.nan)

        # Add labels to point cloud
        tensor_map = {}
        for k, v in pcd.point.items():
            v_numpy = v.numpy()
            if len(v_numpy.shape) == 1:
                tensor_map[k] = v_numpy.reshape(-1, 1)
            else:
                tensor_map[k] = v_numpy

        tensor_map["semantic_labels"] = semantic_labels
        tensor_map["instance_labels"] = instance_labels

        return o3d.t.geometry.PointCloud(tensor_map)

    def _add_empty_labels(self, pcd):
        """Add empty label fields to a point cloud"""
        positions = pcd.point.positions.numpy()
        semantic_labels = np.full((positions.shape[0], 1), np.nan)
        instance_labels = np.full((positions.shape[0], 1), np.nan)

        tensor_map = {}
        for k, v in pcd.point.items():
            v_numpy = v.numpy()
            if len(v_numpy.shape) == 1:
                tensor_map[k] = v_numpy.reshape(-1, 1)
            else:
                tensor_map[k] = v_numpy

        tensor_map["semantic_labels"] = semantic_labels
        tensor_map["instance_labels"] = instance_labels

        return o3d.t.geometry.PointCloud(tensor_map)

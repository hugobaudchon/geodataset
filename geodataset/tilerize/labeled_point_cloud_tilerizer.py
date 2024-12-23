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
from geodataset.geodata.point_cloud_tile import PointCloudTileMetadataCollection
from geodataset.tilerize.point_cloud_tilerizer import PointCloudTilerizer
from geodataset.utils.file_name_conventions import PointCloudCocoNameConvention, validate_and_convert_product_name
from pathlib import Path


class LabeledPointCloudTilerizer(PointCloudTilerizer):
    """
    Class to tilerize a labeled point cloud dataset with the given AOIs.

    Parameters
    ----------
    point_cloud_path : Union[str, Path]
        Path to the point cloud file.
    labels_path : Union[str, Path]
        Path to the labels file.
    output_path : Union[str, Path]
        Path to the output folder.
    tiles_metadata : Union[TileMetadataCollection, None], optional
        Metadata of the tiles, by default None.
    coco_categories_list : List[dict], optional
        List of categories for the COCO dataset, by default None.
    aois_config : Union[AOIFromPackageConfig, None], optional
        Configuration for the AOIs, by default None.
    min_intersection_ratio : float, optional
        Minimum intersection ratio for the labels, by default 0.9.
    ignore_tiles_without_labels : bool, optional
        Whether to ignore tiles without labels, by default False.
    geopackage_layer_name : str, optional
        Name of the layer in the geopackage file, by default None.
    main_label_category_column_name : str, optional
        Name of the main label category column, by default None.
    other_labels_attributes_column_names : List[str], optional
        List of other label attributes, by default None.
    coco_n_workers : int, optional
        Number of workers for the COCO dataset, by default 1.
    tile_side_length : float, optional
        Side length of the tile, by default None.
    tile_overlap : float, optional
        Overlap of the tiles, by default 0.5.
    max_tile : int, optional
        Maximum number of tiles, by default 50000.
    keep_dims : List[str], optional
        List of dimensions to keep, by default None.
    downsample_voxel_size : float, optional
        Voxel size for downsampling, by default None.
    verbose : bool, optional
        Whether to print verbose output, by default False.
    force : bool, optional
        Whether to force the tilerization, by default False. Useful for job submissions
    """


    def __init__(
        self,
        point_cloud_path:Union[str,Path],
        labels_path:Union[str,Path],
        output_path:Union[str,Path],
        tiles_metadata: Union[PointCloudTileMetadataCollection, None] = None,
        coco_categories_list: List[dict] = None,
        aois_config: Union[AOIFromPackageConfig, None] = None,
        min_intersection_ratio: float = 0.9,
        ignore_tiles_without_labels: bool = False,
        geopackage_layer_name: str = None,
        main_label_category_column_name: str = "Label",
        other_labels_attributes_column_names: List[str] = None,
        coco_n_workers: int = 1,
        tile_overlap: float = 0.5,
        max_tile: int = 50000,
        keep_dims: List[str] = None,
        downsample_voxel_size: float = None,
        verbose: bool = False,
        force: bool = False,
        tile_side_length: float = None,
    ) -> None:
        self.point_cloud_path = Path(point_cloud_path)
        self.product_name = validate_and_convert_product_name(strip_all_extensions_and_path(self.point_cloud_path))
        self.label_path = labels_path
        self.output_path = Path(output_path)
        self.tiles_metadata = tiles_metadata
        self.min_intersection_ratio = min_intersection_ratio
        self.ignore_tiles_without_labels = ignore_tiles_without_labels
        self.main_label_category_column_name = main_label_category_column_name
        self.other_labels_attributes_column_names = other_labels_attributes_column_names
        self.aois_config = aois_config
        self.coco_n_workers = coco_n_workers
        self.coco_categories_list = coco_categories_list
        self.downsample_voxel_size = downsample_voxel_size
        self.tile_overlap = tile_overlap
        self.keep_dims = keep_dims if keep_dims is not None else "ALL"
        self.verbose = verbose
        self.max_tile = max_tile
        self.force = force

        self.tile_side_length = tile_side_length

        assert self.tile_overlap < 1.0, "Tile overlap should be less than 1.0"

        self.pc_tiles_folder_path = (
            self.output_path / f"pc_tiles_{self.downsample_voxel_size}"
            if self.downsample_voxel_size
            else self.output_path / "pc_tiles"
        )

        self.aoi_tiles = None
        self.aoi_labels = None
        self.category_to_id_map = None

        keep_categories = list(self.create_category_to_id_map().keys())

        # PolygonLabel class
        self.labels = self.load_labels(
            geopackage_layer=geopackage_layer_name,
            main_label_category_column_name=main_label_category_column_name,
            other_labels_attributes_column_names=other_labels_attributes_column_names,
            keep_categories=keep_categories,
        )

        if self.tiles_metadata is None:
            assert (
                self.tile_overlap is not None
            ), "Tile Overlap is required if tile metadata is not provided"

            self.populate_tiles_metadata()

        assert self.tiles_metadata is not None, "Tile metadata is required for RLE encoding (image height and width)"
        assert self.tiles_metadata.height is not None and self.tiles_metadata.width is not None, "Height and width of the tiles are required for RLE encoding"

        assert (
            len(self.tiles_metadata) < max_tile
        ), f"Number of max possible tiles {len(self.tiles_metadata)} exceeds the maximum number of tiles {max_tile}"

        if self.aois_config is None:
            raise Exception("Please provide an aoi_config. Currently don't support 'None'.")
        self.aoi_engine = AOIBaseFromGeoFileInCRS(aois_config)
        self.aois_in_both = self._check_aois_in_both_tiles_and_config()

        self.create_folder()

    def create_category_to_id_map(self):
        category_id_map = {}
        if self.coco_categories_list:
            for category_dict in self.coco_categories_list:
                category_id_map[category_dict["name"]] = category_dict["id"]
                for name in category_dict["other_names"]:
                    category_id_map[name] = category_dict["id"]

        return category_id_map

    def load_labels(
        self,
        geopackage_layer=None,
        main_label_category_column_name="labels",
        other_labels_attributes_column_names=None,
        keep_categories=None,
    ) -> PolygonLabels:
        labels = PolygonLabels(
            path=self.label_path,
            geopackage_layer_name=geopackage_layer,
            main_label_category_column_name=main_label_category_column_name,
            other_labels_attributes_column_names=other_labels_attributes_column_names,
            keep_categories=keep_categories,
        )
        return labels

    def _find_tiles_associated_labels(self):
        print("Finding the labels associate to each tile...")

        tile_ids = [tile.tile_id for tile in self.tiles_metadata]
        geometry = [tile.geometry.values[0][0] for tile in self.tiles_metadata]

        tiles_gdf = gpd.GeoDataFrame(
            data={"tile_id": tile_ids, "geometry": geometry}
        )  # CRS not set here

        tiles_gdf.crs = self.tiles_metadata.crs

        labels_gdf = self.labels.geometries_gdf
        labels_gdf = labels_gdf.to_crs(tiles_gdf.crs)

        labels_gdf["total_instance_area"] = labels_gdf.geometry.area
        inter_polygons = gpd.overlay(
            tiles_gdf, labels_gdf, how="intersection", keep_geom_type=True
        )
        inter_polygons["instance_area_within_tile"] = inter_polygons.geometry.area
        inter_polygons["intersection_ratio_tiles"] = (
            inter_polygons["instance_area_within_tile"]
            / inter_polygons["total_instance_area"]
        )
        significant_polygons_inter = inter_polygons[
            inter_polygons["intersection_ratio_tiles"] > self.min_intersection_ratio
        ]
        significant_polygons_inter.reset_index()

        return significant_polygons_inter

    def _remove_unlabeled_tiles(self, associated_labels):
        """
        Remove tiles that do not have any associated labels
        """

        labeled_tiles = []
        for tile in self.tiles_metadata:
            individual_associated_labels = associated_labels[
                associated_labels["tile_id"] == tile.tile_id
            ]
            if (
                self.ignore_tiles_without_labels
                and len(individual_associated_labels) == 0
            ):
                continue
            else:
                labeled_tiles.append(tile)

        # Rewrite self.tiles_metadata with only the tiles that have labels

        tiles_removed = len(self.tiles_metadata) - len(labeled_tiles)
        if tiles_removed > 0:
            warnings.warn(f"Removed {tiles_removed} tiles without labels")
            self.tiles_metadata = PointCloudTileMetadataCollection(
                labeled_tiles, product_name=self.tiles_metadata.product_name
            )

    def _get_aoi_tiles(self):
        aois_gdf = self.aoi_engine.get_aoi_gdf()
        # //NOTE -  Cannot check for color as data is not provided here and only metadata is provided.
        tiles_gdf = self.tiles_metadata.gdf
        aois_gdf = aois_gdf.to_crs(tiles_gdf.crs)

        if 'aoi' not in tiles_gdf or ('aoi' in tiles_gdf and tiles_gdf['aoi'].isnull().all()):
            # only keeping 'aoi' column from aois_gdf, ignoring the one from tiles_gdf as it is same info if set
            tiles_gdf = tiles_gdf.drop(columns=['aoi'])

            aois_gdf.crs = tiles_gdf.crs
            intersections = gpd.overlay(tiles_gdf, aois_gdf, how="intersection")
            intersections["intersection_area"] = intersections.geometry.area
            max_intersection_per_tile = intersections.loc[
                intersections.groupby("tile_id")["intersection_area"].idxmax()
            ]
            aois_tiles = (
                max_intersection_per_tile.groupby("aoi")["tile_id"].apply(list).to_dict()
            )
        else:
            # just use tiles_gdf, no need to overlay
            aois_tiles = tiles_gdf.groupby("aoi")["tile_id"].apply(list).to_dict()

        return aois_tiles, aois_gdf

    def _get_aoi_labels(self):
        tile_associated_labels = self._find_tiles_associated_labels()

        if self.ignore_tiles_without_labels:
            self._remove_unlabeled_tiles(tile_associated_labels)

        aois_tiles, aois_gdf = self._get_aoi_tiles()

        tile_associated_labels["instance_id"] = tile_associated_labels.index

        intersected_labels_aois = gpd.overlay(
            tile_associated_labels, aois_gdf, how="intersection", keep_geom_type=False
        )
        final_aoi_labels = {aoi: [] for aoi in list(aois_tiles.keys()) + ["all"]}

        for aoi in aois_tiles:
            tiles_to_remove = []
            for tile_id in aois_tiles[aoi]:
                labels_crs_coords = intersected_labels_aois[
                    intersected_labels_aois["tile_id"] == tile_id
                ]
                labels_crs_coords = labels_crs_coords[labels_crs_coords["aoi"] == aoi]
                labels_ids = labels_crs_coords["instance_id"].tolist()
                labels_tiles_coords = intersected_labels_aois[
                    intersected_labels_aois.instance_id.isin(labels_ids)
                ]
                labels_tiles_coords = labels_tiles_coords[
                    labels_tiles_coords["aoi"] == aoi
                ]

                # removing boxes that have an area of 0.0
                labels_tiles_coords = labels_tiles_coords[
                    labels_tiles_coords.geometry.area > 0.0
                ]

                # remove boxes where x2 - x1 <= 0.5 or y2 - y1 <= 0.5, as they are too small and will cause issues when rounding the coordinates (area=0)
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

                if self.ignore_tiles_without_labels and len(labels_tiles_coords) == 0:
                    tiles_to_remove.append(tile_id)
                    print(f"Removing tile {tile_id} from AOI {aoi} as it has no labels")
                    continue

                final_aoi_labels[aoi].append(labels_tiles_coords)
                final_aoi_labels["all"].append(labels_tiles_coords)
            
            for id in tiles_to_remove:
                aois_tiles[aoi].remove(id)

        return aois_tiles, final_aoi_labels

    def _get_tiles_per_aoi(self):
        """
        Get the tiles that intersect with the AOIs
        """

        tiles_per_aoi = {}
        for aoi_name in self.aoi_engine.loaded_aois:
            aoi_gdf = self.aoi_engine.loaded_aois[aoi_name]
            aoi_tiles = []
            for tile in self.tiles_metadata:
                if any(tile.geometry.intersects(aoi_gdf.geometry)):
                    aoi_tiles.append(tile)
            tiles_per_aoi[aoi_name] = aoi_tiles

        return tiles_per_aoi

    def tilerize(self):
        self._generate_labels()
        self._tilerize()
        super().plot_aois()

    def _tilerize(self):
        new_tile_md_list = []
        with CopcReader.open(self.point_cloud_path) as reader:
            reader_crs = reader.header.parse_crs()
            for tile_md in tqdm(self.tiles_metadata):
                data = super().query_tile(tile_md, reader)

                if len(data) == 0:
                    continue
                pcd = super()._laspy_to_o3d(data, self.keep_dims.copy() if type(self.keep_dims) is not str else self.keep_dims)

                if self.downsample_voxel_size:
                    pcd = super()._downsample_tile(pcd, self.downsample_voxel_size)
                pcd = self._keep_unique_points(pcd)
                pcd = self._remove_points_outside_aoi(pcd, tile_md, reader_crs)
                tile_labels = self._get_tile_labels(
                    tile_md.tile_id, self.aoi_tiles.copy(), self.aoi_labels.copy()
                )
                tile_labels = tile_labels.to_crs(reader_crs)
                pcd = self._add_labels(pcd, tile_labels, reader_crs)
                new_tile_md_list.append(tile_md)

                downsampled_tile_path = (
                    self.pc_tiles_folder_path / f"{tile_md.aoi}/{tile_md.tile_name}"
                )
                o3d.t.io.write_point_cloud(str(downsampled_tile_path), pcd)

        print(f"Finished tilerizing. Number of tiles generated: {len(new_tile_md_list)}.")
        self.tiles_metadata = PointCloudTileMetadataCollection(
            new_tile_md_list, product_name=self.tiles_metadata.product_name
        )

    def _generate_tile_metadata(self, **kwargs):
        return super()._generate_tile_metadata(**kwargs)

    def get_tile(self, **kwargs):
        return super().get_tile(**kwargs)

    def _generate_labels(self):
        """
        Generate the tiles and the COCO dataset(s) for each AOI (or for the 'all' AOI) and save everything to the disk.
        """
        self.aoi_tiles, self.aoi_labels = self._get_aoi_labels()

        coco_paths = {}
        for aoi in self.aoi_labels:
            if aoi == "all" and len(self.aoi_labels.keys()) > 1:
                # don't save the 'all' tiles if aois were provided.
                continue

            labels = self.aoi_labels[aoi]

            if len(labels) == 0:
                print(f"No tiles found for AOI {aoi}. Skipping...")
                continue

            aoi_tiles_metadata = PointCloudTileMetadataCollection(
                [tile for tile in self.tiles_metadata if tile.tile_id in self.aoi_tiles[aoi]]
            )

            polygons = [x["geometry"].to_list() for x in labels]
            categories_list = (
                [
                    x[self.labels.main_label_category_column_name].to_list()
                    for x in labels
                ]
                if self.labels.main_label_category_column_name
                else None
            )
            other_attributes_dict_list = (
                [
                    {
                        attribute: label[attribute].to_list()
                        for attribute in self.labels.other_labels_attributes_column_names
                    }
                    for label in labels
                ]
                if self.labels.other_labels_attributes_column_names
                else None
            )
            other_attributes_dict_list = (
                [
                    [
                        {k: d[k][i] for k in d}
                        for i in range(len(next(iter(d.values()))))
                    ]
                    for d in other_attributes_dict_list
                ]
                if self.labels.other_labels_attributes_column_names
                else None
            )

            # Saving the tiles

            coco_output_file_path = (
                self.output_path
                / PointCloudCocoNameConvention.create_name(
                    product_name=self.tiles_metadata.product_name,
                    ground_resolution=None,
                    scale_factor=None,
                    voxel_size=self.downsample_voxel_size,
                    fold=aoi,
                )
            )

            coco_generator = PointCloudCOCOGenerator(
                description="Dataset for the product XYZ",
                tiles_metadata=aoi_tiles_metadata,
                polygons=polygons,
                scores=None,
                categories=categories_list,
                other_attributes=other_attributes_dict_list,
                output_path=coco_output_file_path,
                n_workers=self.coco_n_workers,
                coco_categories_list=self.coco_categories_list,
            )

            categories_coco, category_to_id_map = coco_generator.generate_coco()
            coco_paths[aoi] = coco_output_file_path

            self.category_to_id_map = category_to_id_map
            self.category_to_id_map[np.nan] = np.nan

        return coco_paths, categories_coco, category_to_id_map

    def _add_labels(self, pcd, tile_labels, reader_crs):
        # Create a geodataframe for all the points

        positions = pcd.point.positions.numpy()
        geopoints = gpd.GeoDataFrame(positions)
        geopoints = gpd.GeoDataFrame(geopoints.apply(lambda x: Point(x), axis=1))
        geopoints.columns = ["points"]
        geopoints = geopoints.set_geometry("points")
        geopoints.crs = reader_crs

        def category_to_id(x):
            if x in self.category_to_id_map:
                return self.category_to_id_map[x]
            else:
                return np.nan

        if tile_labels.empty is False:
            tile_labels.drop(columns=['level_0'], errors='ignore', inplace=True)

            joined_label = tile_labels.sjoin(
                geopoints, how="right", predicate="contains"
            )

            # joined_label = joined_label[~joined_label.index.duplicated(keep='first')] # Removes mismatch in very few of the cases #//TODO: - This is because polygons overlap

            points = np.vstack(
                [
                    joined_label.geometry.x,
                    joined_label.geometry.y,
                    joined_label.geometry.z,
                ]
            ).T

            assert np.array_equal(
                positions, points
            ), f"The points are not in the same order for {tile_labels['tile_id']}"

            joined_label[self.main_label_category_column_name] = joined_label[self.main_label_category_column_name].apply(category_to_id)

            semantic_labels = joined_label[self.main_label_category_column_name].values.reshape((-1, 1))
            instance_labels = joined_label["instance_id"].values.reshape((-1, 1))

        else:
            if self.verbose:
                warnings.warn("No labels found for the tile")

            points = np.vstack(
                [geopoints.geometry.x, geopoints.geometry.y, geopoints.geometry.z]
            ).T

            assert np.array_equal(
                positions, points
            ), f"The points are not in the same order for {tile_labels['tile_id']}"

            semantic_labels = np.nan * np.ones((positions.shape[0], 1))
            instance_labels = np.nan * np.ones((positions.shape[0], 1))

        tensor_map = {}
        for k, v in pcd.point.items():
            v = v.numpy()
            if len(v.shape) == 1:
                value = v.reshape((-1, 1))
            else:
                value = v

            tensor_map[k] = value

        tensor_map["semantic_labels"] = semantic_labels
        tensor_map["instance_labels"] = instance_labels

        new_pcd = o3d.t.geometry.PointCloud(tensor_map)

        return new_pcd

    def _get_tile_labels(self, tile_id, aois_tiles, aoi_labels):
        aoi = self._get_aoi_from_tile_id(tile_id, aois_tiles)
        ind_aoi = self._get_ind_aoi_tiles(tile_id, aoi, aoi_labels)

        return aoi_labels[aoi][ind_aoi] if ind_aoi is not None else gpd.GeoDataFrame()

    def _get_aoi_from_tile_id(self, tile_id, aois_tiles):
        aoi_types = list(aois_tiles.keys())

        for aoi in aoi_types:
            tile_ids = aois_tiles[aoi]
            if tile_id in tile_ids:
                return aoi

        if self.verbose:
            warnings.warn(f"Tile {tile_id} not found in any AOI")
        return None

    def _get_ind_aoi_tiles(self, tile_id, aoi, aoi_labels):
        if aoi:
            for i, labels in enumerate(aoi_labels[aoi]):
                if tile_id in labels["tile_id"].values:
                    return i
        else:
            return None

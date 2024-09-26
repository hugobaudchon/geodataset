import warnings
from typing import List, Union

import geopandas as gpd
import numpy as np
import open3d as o3d
from laspy import CopcReader
from shapely.geometry import Point
from tqdm import tqdm

from geodataset.aoi import AOIFromPackageConfig
from geodataset.aoi.aoi_base import AOIBaseFromGeoFile
from geodataset.dataset.coco_generator import PointCloudCOCOGenerator
from geodataset.labels.base_labels import PolygonLabels
from geodataset.metadata.tile_metadata import TileMetadataCollection
from geodataset.tilerize.point_cloud_tilerizer import PointCloudTilerizer
from geodataset.utils.file_name_conventions import PointCloudCocoNameConvention


class LabeledPointCloudTilerizer(PointCloudTilerizer):
    """
    Tiler for labeled point cloud data
    """

    def __init__(
        self,
        point_cloud_path,
        labels_path,
        output_path,
        tiles_metadata: Union[TileMetadataCollection, None] = None,
        coco_categories_list: List[dict] = None,
        aois_config: Union[AOIFromPackageConfig, None] = None,
        min_intersection_ratio: float = 0.9,
        ignore_tiles_without_labels: bool = False,
        geopackage_layer_name: str = None,
        main_label_category_column: str = None,
        other_labels_attributes_column: List[str] = None,
        use_rle_for_labels: bool = True,
        coco_n_workers: int = 1,
        tile_side_length: float = None,
        tile_overlap: float = 1.0,
        max_tile: int = 50000,
        keep_dims: List[str] = None,
        downsample_voxel_size: float = None,
        verbose: bool = False,
        force: bool = False,
    ) -> None:
        self.point_cloud_path = point_cloud_path
        self.label_path = labels_path
        self.output_path = output_path
        self.tiles_metadata = tiles_metadata
        self.min_intersection_ratio = min_intersection_ratio
        self.ignore_tiles_without_labels = ignore_tiles_without_labels
        self.aoi_engine = AOIBaseFromGeoFile(aois_config)
        self.use_rle_for_labels = use_rle_for_labels
        self.coco_n_workers = coco_n_workers
        self.coco_categories_list = coco_categories_list
        self.downsample_voxel_size = downsample_voxel_size
        self.tile_overlap = tile_overlap
        self.tile_overlap = tile_overlap
        self.keep_dims = keep_dims if keep_dims is not None else "ALL"
        self.verbose = verbose
        self.max_tile = max_tile
        self.force = force
        self.tile_side_length = tile_side_length

        self.pc_tiles_folder_path = (
            self.output_path / f"labelled_pc_tiles_{self.downsample_voxel_size}"
            if self.downsample_voxel_size
            else self.output_path / "labelled_pc_tiles"
        )
        self.annotation_folder_path = self.output_path / "annotations"

        self.aoi_tiles = None
        self.aoi_labels = None
        self.category_to_id_map = None

        keep_categories = list(self.create_category_to_id_map().keys())

        # PolygonLabel class
        self.labels = self.load_labels(
            geopackage_layer=geopackage_layer_name,
            main_label_category=main_label_category_column,
            other_labels_attributes=other_labels_attributes_column,
            keep_categories=keep_categories,
        )

        if self.tiles_metadata is None:
            assert (
                self.tile_overlap is not None
            ), "Tile Overlap is required if tile metadata is not provided"

            self.populate_tiles_metadata()

        assert (
            len(self.tiles_metadata) < max_tile
        ), f"Number of max possible tiles {len(self.tiles_metadata)} exceeds the maximum number of tiles {max_tile}"

        super().create_folder()

    def create_category_to_id_map(self):
        category_id_map = {}
        for category_dict in self.coco_categories_list:
            category_id_map[category_dict["name"]] = category_dict["id"]
            for name in category_dict["other_names"]:
                category_id_map[name] = category_dict["id"]

        return category_id_map

    def load_labels(
        self,
        geopackage_layer=None,
        main_label_category="labels",
        other_labels_attributes=None,
        keep_categories=None,
    ) -> None:
        labels = PolygonLabels(
            path=self.label_path,
            geopackage_layer_name=geopackage_layer,
            main_label_category_column=main_label_category,
            other_labels_attributes_column=other_labels_attributes,
            keep_categories=keep_categories,
        )
        return labels

    def _find_tiles_associated_labels(self):
        print("Finding the labels associate to each tile...")

        tile_ids = [tile.id for tile in self.tiles_metadata]
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

        # No geometry adjustment here, as the labels are already in the same CRS as the tiles
        # //TODO: Check why it is needed in LabeledRasterTilerizer
        return significant_polygons_inter

    def _remove_unlabeled_tiles(self, associated_labels):
        """
        Remove tiles that do not have any associated labels
        """

        labeled_tiles = []
        for tile in self.tiles_metadata:
            individual_associated_labels = associated_labels[
                associated_labels["tile_id"] == tile.id
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
            self.tiles_metadata = TileMetadataCollection(
                labeled_tiles, product_name=self.tiles_metadata.product_name
            )

    def _get_aoi_labels(self):
        tile_associated_labels = self._find_tiles_associated_labels()

        if self.ignore_tiles_without_labels:
            self._remove_unlabeled_tiles(tile_associated_labels)

        aois_gdf = self.aoi_engine.get_aoi_gdf()
        # //NOTE -  Cannot check for color as data is not provided here and only metadata is provided.

        tiles_gdf = self.tiles_metadata.gdf
        aois_gdf = aois_gdf.to_crs(tiles_gdf.crs)

        # To generate aois_tiles, we need to find the tiles that intersect with the AOIs
        intersections = gpd.overlay(tiles_gdf, aois_gdf, how="intersection")
        intersections["intersection_area"] = intersections.geometry.area
        max_intersection_per_tile = intersections.loc[
            intersections.groupby("tile_id")["intersection_area"].idxmax()
        ]
        aois_tiles = (
            max_intersection_per_tile.groupby("aoi")["tile_id"].apply(list).to_dict()
        )

        tile_associated_labels["instance_id"] = tile_associated_labels.index

        intersected_labels_aois = gpd.overlay(
            tile_associated_labels, aois_gdf, how="intersection", keep_geom_type=False
        )
        final_aoi_labels = {aoi: [] for aoi in list(aois_tiles.keys()) + ["all"]}

        for aoi in aois_tiles:
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
                    print(f"Removing tile {tile_id} from AOI {aoi} as it has no labels")
                    continue

                final_aoi_labels[aoi].append(labels_tiles_coords)
                final_aoi_labels["all"].append(labels_tiles_coords)

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
            for tile_md in tqdm(self.tiles_metadata):
                data = super().query_tile(tile_md, reader)

                if len(data) == 0:
                    continue
                pcd = super()._laspy_to_o3d(data, self.keep_dims.copy())

                if self.downsample_voxel_size:
                    pcd = super()._downsample_tile(pcd, self.downsample_voxel_size)

                pcd = super()._keep_unique_points(pcd)
                tile_labels = self._get_tile_labels(
                    tile_md.id, self.aoi_tiles.copy(), self.aoi_labels.copy()
                )
                pcd = self._add_labels(pcd, tile_labels)
                new_tile_md_list.append(tile_md)

                downsampled_tile_path = (
                    self.pc_tiles_folder_path / f"{tile_md.tile_name}"
                )
                o3d.t.io.write_point_cloud(str(downsampled_tile_path), pcd)

        self.tiles_metadata = TileMetadataCollection(
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
                print(f"No tiles found for AOI {aoi}, skipping...")
                continue

            if len(labels) == 0:
                print(f"No tiles found for AOI {aoi}. Skipping...")
                continue

            tiles_metadata = TileMetadataCollection(
                [tile for tile in self.tiles_metadata if tile.id in self.aoi_tiles[aoi]]
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
                self.annotation_folder_path
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
                tiles_metadata=tiles_metadata,
                polygons=polygons,
                scores=None,
                categories=categories_list,
                other_attributes=other_attributes_dict_list,
                output_path=coco_output_file_path,
                use_rle_for_labels=self.use_rle_for_labels,
                n_workers=self.coco_n_workers,
                coco_categories_list=self.coco_categories_list,
            )

            categories_coco, category_to_id_map = coco_generator.generate_coco()
            coco_paths[aoi] = coco_output_file_path

            self.category_to_id_map = category_to_id_map
            self.category_to_id_map[np.nan] = np.nan

        return coco_paths, categories_coco, category_to_id_map

    def _add_labels(self, pcd, tile_labels):
        # Create a geodataframe for all the points

        positions = pcd.point.positions.numpy()
        geopoints = gpd.GeoDataFrame(positions)
        geopoints = gpd.GeoDataFrame(geopoints.apply(lambda x: Point(x), axis=1))
        geopoints.columns = ["points"]
        geopoints = geopoints.set_geometry("points")

        def category_to_id(x):
            if x in self.category_to_id_map:
                return self.category_to_id_map[x]
            else:
                return np.nan

        if tile_labels.empty is False:
            geopoints.crs = tile_labels.crs

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

            joined_label["Label"] = joined_label["Label"].apply(category_to_id)

            semantic_labels = joined_label["Label"].values.reshape((-1, 1))
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

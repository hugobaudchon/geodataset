from geodataset.tilerize.base_tilerizer import LabeledTilerizer
from geodataset.labels.base_labels import PolygonLabels

import geopandas as gpd
from geodataset.aoi import AOIFromPackageConfig
from shapely import box, Polygon, MultiPolygon

from typing import List, Tuple, Union
import pandas as pd
import warnings

from geodataset.metadata.tile_metadata import TileMetadata, TileMetadataCollection
from geodataset.dataset.coco_generator import PointCloudCOCOGenerator
from geodataset.utils.file_name_conventions import CocoNameConvention

from geodataset.tilerize.point_cloud_tilerizer import PointCloudTilerizer
from shapely.geometry import Point

import open3d as o3d
from tqdm import tqdm
import numpy as np
import open3d.core as o3c

from concurrent.futures import ThreadPoolExecutor

class LabeledPointCloudTilerizer(PointCloudTilerizer):
    """
    Tiler for labeled point cloud data
    """

    def __init__(self, 
                 point_cloud_path,
                 labels_path,
                 output_path,
                 tiles_metadata,
                 aois_config: Union[AOIFromPackageConfig , None] = None,
                 min_intersection_ratio: float = 0.9,
                 ignore_tiles_without_labels: bool = False,
                 geopackage_layer: str = None,
                 main_label_category_column: str = None,
                 other_labels_attributes_column: List[str] = None,
                 use_rle_for_labels: bool =True,
                 coco_n_workers: int =1,
                 coco_categories_list: Union[List[dict],None] =None,
                 donwsample_args: Union[dict, None] = None ) -> None:
        
        self.point_cloud_path = point_cloud_path
        self.label_path = labels_path
        self.output_path = output_path
        self.tiles_metadata = tiles_metadata
        self.min_intersection_ratio = min_intersection_ratio
        self.ignore_tiles_without_labels = ignore_tiles_without_labels
        self.aoi_engine =  AOIBaseFromGeoFile(aois_config)
        self.use_rle_for_labels = use_rle_for_labels
        self.coco_n_workers = coco_n_workers
        self.coco_categories_list = coco_categories_list

        self.pc_tiles_folder_path = self.output_path / "pc_tiles"
        self.annotation_folder_path = self.output_path / "annotations"

        self.dowsample_args = donwsample_args

        if donwsample_args:
            assert "voxel_size" in donwsample_args

        self.downsample_folder_path = self.output_path / f"pc_tiles_downsampled_{donwsample_args['voxel_size']}" if  self.dowsample_args else None

        self.labels = self.load_labels(geopackage_layer=geopackage_layer, main_label_category=main_label_category_column, other_labels_attributes=other_labels_attributes_column)

        self.aoi_tiles = None
        self.aois_labels = None
        self.category_to_id_map = None

    def load_labels(self, 
                     geopackage_layer= None, 
                     main_label_category= "labels",
                     other_labels_attributes= None,
                     ) -> None:
        
        labels = PolygonLabels(path=self.label_path, geopackage_layer_name=geopackage_layer, main_label_category_column= main_label_category, other_labels_attributes_column=other_labels_attributes)
        return labels
    
    def _find_tiles_associated_labels(self):
        print("Finding the labels associate to each tile...")
        
        tile_ids = [tile.id for tile in self.tiles_metadata]
        geometry = [tile.geometry.values[0][0] for tile in self.tiles_metadata]

        tiles_gdf = gpd.GeoDataFrame(data={'tile_id': tile_ids,
                                           'geometry': geometry}) #CRS not set here
        
        tiles_gdf.crs = self.tiles_metadata[0].crs

        labels_gdf = self.labels.geometries_gdf
        
        labels_gdf['label_area'] = labels_gdf.geometry.area
        inter_polygons = gpd.overlay(tiles_gdf, labels_gdf, how='intersection', keep_geom_type=True)
        inter_polygons['area'] = inter_polygons.geometry.area
        inter_polygons['intersection_ratio'] = inter_polygons['area'] / inter_polygons['label_area']
        significant_polygons_inter = inter_polygons[inter_polygons['intersection_ratio'] > self.min_intersection_ratio]
        significant_polygons_inter.reset_index()

        # No geometry adjustment here, as the labels are already in the same CRS as the tiles
        # //TODO: Check why it is needed in LabeledRasterTilerizer
        return significant_polygons_inter

    def _remove_unlabeled_tiles(self,  associated_labels):
        """
        Remove tiles that do not have any associated labels
        """

        labeled_tiles = []
        for tile in self.tiles_metadata:
            individual_associated_labels = associated_labels[associated_labels['tile_id'] == tile.id]
            if self.ignore_tiles_without_labels and len(individual_associated_labels) == 0:
                continue
            else:
                labeled_tiles.append(tile)

        # Rewrite self.tiles_metadata with only the tiles that have labels
        
        tiles_removed = len(self.tiles_metadata) - len(labeled_tiles)
        if tiles_removed > 0:
            warnings.warn(f"Removed {tiles_removed} tiles without labels")
            self.tiles_metadata = TileMetadataCollection(labeled_tiles)

    def _get_aoi_labels(self):

        tile_associated_labels = self._find_tiles_associated_labels()

        self._remove_unlabeled_tiles(tile_associated_labels)
   
        aois_gdf = self.aoi_engine.get_aoi_gdf()
        # //NOTE -  Cannot check for color as data is not provided here and only metadata is provided.

        tiles_gdf = self.tiles_metadata.gdf
        
        intersections = gpd.overlay(tiles_gdf, aois_gdf, how='intersection')
        intersections['intersection_area'] = intersections.geometry.area
        max_intersection_per_tile = intersections.loc[intersections.groupby('tile_id')['intersection_area'].idxmax()]
        aois_tiles = max_intersection_per_tile.groupby('aoi')['tile_id'].apply(list).to_dict()
        
        tile_associated_labels["label_id"] = tile_associated_labels.index
        tile_associated_labels["label_area"] = tile_associated_labels.geometry.area
        intersected_labels_aois = gpd.overlay(tile_associated_labels, aois_gdf, how='intersection', keep_geom_type=False)
        intersected_labels_aois['intersection_area'] = intersected_labels_aois.geometry.area
        intersected_labels_aois['intersection_ratio'] = intersected_labels_aois['intersection_area'] / intersected_labels_aois['label_area']
        intersected_labels_aois = intersected_labels_aois[intersected_labels_aois['intersection_ratio'] > self.min_intersection_ratio]


        final_aois_labels = {aoi: [] for aoi in list(aois_tiles.keys()) + ['all']}

        for aoi in aois_tiles:
            for tile_id in aois_tiles[aoi]:
                
                labels_crs_coords = intersected_labels_aois[intersected_labels_aois['tile_id'] == tile_id]
                labels_crs_coords = labels_crs_coords[labels_crs_coords['aoi'] == aoi]
                labels_ids = labels_crs_coords['label_id'].tolist()
                labels_tiles_coords = intersected_labels_aois[intersected_labels_aois.label_id.isin(labels_ids)]

                # removing boxes that have an area of 0.0
                labels_tiles_coords = labels_tiles_coords[labels_tiles_coords.geometry.area > 0.0]

                # remove boxes where x2 - x1 <= 0.5 or y2 - y1 <= 0.5, as they are too small and will cause issues when rounding the coordinates (area=0)
                labels_tiles_coords = labels_tiles_coords[(labels_tiles_coords.geometry.bounds['maxx'] - labels_tiles_coords.geometry.bounds['minx']) > 0.5]
                labels_tiles_coords = labels_tiles_coords[(labels_tiles_coords.geometry.bounds['maxy'] - labels_tiles_coords.geometry.bounds['miny']) > 0.5]

                if self.ignore_tiles_without_labels and len(labels_tiles_coords) == 0:
                    print(f"Removing tile {tile_id} from AOI {aoi} as it has no labels")
                    continue
            
                final_aois_labels[aoi].append(labels_tiles_coords)
                final_aois_labels['all'].append(labels_tiles_coords)

        return aois_tiles, final_aois_labels
    
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
    

    def tilerize(self, **kwargs):
        return super().tilerize(**kwargs)
    
    
    def _generate_tile_metadata(self, **kwargs):
        return super()._generate_tile_metadata(**kwargs)
    
    def get_tile(self, **kwargs):
        return super().get_tile(**kwargs)
    
    def generate_labels(self):
        """
        Generate the tiles and the COCO dataset(s) for each AOI (or for the 'all' AOI) and save everything to the disk.
        """
        self.aoi_tiles, self.aois_labels = self._get_aoi_labels()

        coco_paths = {}
        for aoi in self.aois_labels:
            if aoi == 'all' and len(self.aois_labels.keys()) > 1:
                # don't save the 'all' tiles if aois were provided.
                continue

            labels = self.aois_labels[aoi]

            if len(labels) == 0:
                print(f"No tiles found for AOI {aoi}, skipping...")
                continue

            if len(labels) == 0:
                print(f"No tiles found for AOI {aoi}. Skipping...")
                continue

            tiles_metadata = TileMetadataCollection([tile for tile in self.tiles_metadata if tile.id in self.aoi_tiles[aoi]])
            polygons = [x['geometry'].to_list() for x in labels]
            categories_list = [x[self.labels.main_label_category_column_name].to_list() for x in labels]\
                if self.labels.main_label_category_column_name else None
            other_attributes_dict_list = [{attribute: label[attribute].to_list() for attribute in
                                           self.labels.other_labels_attributes_column_names} for label in labels]\
                if self.labels.other_labels_attributes_column_names else None
            other_attributes_dict_list = [[{k: d[k][i] for k in d} for i in range(len(next(iter(d.values()))))] for d in other_attributes_dict_list]\
                if self.labels.other_labels_attributes_column_names else None

            # Saving the tiles
            
            coco_output_file_path = self.output_path / CocoNameConvention.create_name(
                product_name="PointCloud",
                ground_resolution=None,
                scale_factor=None,
                fold=aoi
            )

            coco_generator = PointCloudCOCOGenerator(
                description=f"Dataset for the product XYZ",
                tiles_metadata=tiles_metadata,
                polygons=polygons,
                scores=None,
                categories=categories_list,
                other_attributes=other_attributes_dict_list,
                output_path=coco_output_file_path,
                use_rle_for_labels=self.use_rle_for_labels,
                n_workers=self.coco_n_workers,
                coco_categories_list=self.coco_categories_list
            )

            categories_coco, category_to_id_map = coco_generator.generate_coco()
            coco_paths[aoi] = coco_output_file_path

            self.category_to_id_map = category_to_id_map
            self.category_to_id_map[np.nan] = np.nan

        return coco_paths, categories_coco, category_to_id_map


    def lazy_tilerize(self, chunk_size: int | None = 500000) -> None:
        self.create_folder()
        print("Creating tiles...")
        # self._lazy_tilerize(chunk_size)

        self.generate_labels()
        if self.dowsample_args:
            print("Downsampling the tiles...")
            self._downsample()


    def _downsample(self,):

        #//TODO - Add suppport for threading here
        # The process is completely parallelizable

        if "keep_dims" in self.dowsample_args:
            keep_dims = self.dowsample_args["keep_dims"]
        else:
            keep_dims = "ALL"

        voxel_size = self.dowsample_args["voxel_size"]

        for tile_md in tqdm(self.tiles_metadata):
            self._parallel_downsample((tile_md, keep_dims, voxel_size, self.aoi_tiles.copy(), self.aois_labels.copy()))

        # iterable = [(tile_md, keep_dims, voxel_size, self.aoi_tiles.copy(), self.aois_labels.copy()) for tile_md in self.tiles_metadata]
        # print("Iterable created")
        
        # with ThreadPoolExecutor(5) as executor:
        #     list(tqdm(executor.map(self._parallel_downsample, iterable), total=len(iterable)))


    def _parallel_downsample(self, arg):
        tile_md, keep_dims, voxel_size, aois_tiles, aois_labels = arg
        tile_labels = self._get_tile_labels(tile_md.id, aois_tiles, aois_labels)
        pc_tile_path = self.pc_tiles_folder_path / f"{tile_md.output_filename}"
        downsampled_tile = self._downsample_tile(pc_tile_path, keep_dims=keep_dims.copy(), voxel_size=voxel_size)
        if not tile_labels.empty:
            downsampled_tile = self._add_labels(downsampled_tile, tile_labels)
        downsampled_tile_path = self.downsample_folder_path / f"{tile_md.output_filename.replace('.las', '.ply')}"
        
        o3d.t.io.write_point_cloud(str(downsampled_tile_path), downsampled_tile)

    
    def _add_labels(self, pcd, tile_labels):
        
        # Create a geodataframe for all the points
        positions = pcd.point.positions.numpy()
        geopoints = gpd.GeoDataFrame(positions)
        geopoints = gpd.GeoDataFrame(geopoints.apply(lambda x: Point(x), axis=1))
        geopoints.columns = ['points']
        geopoints = geopoints.set_geometry('points')


        geopoints.crs = tile_labels.crs

        joined_label = tile_labels.sjoin(geopoints, how='right', predicate='contains')

        points = np.vstack([joined_label.geometry.x, joined_label.geometry.y, joined_label.geometry.z]).T

        assert np.array_equal(positions, points), f"The points are not in the same order for {tile_labels['tile_id']}"
        
        joined_label['Label'] = joined_label['Label'].apply(lambda x: self.category_to_id_map[x])

        joined_label = joined_label[~joined_label.index.duplicated(keep='first')] # Removes mismatch in very few of the cases #//TODO: - Check why this is happening
        
        semantic_labels = joined_label['Label'].values.reshape((-1, 1))
        instance_labels = joined_label['label_id'].values.reshape((-1, 1))

        tensor_map = {}
        for k, v in pcd.point.items():
            v = v.numpy()
            if len(v.shape) == 1:
                value = o3c.Tensor(v.reshape((-1, 1)))
            else:
                value = o3c.Tensor(v)

            tensor_map[k] = value

        tensor_map['semantic_labels'] = o3c.Tensor(semantic_labels)
        tensor_map['instance_labels'] = o3c.Tensor(instance_labels)

        new_pcd = o3d.t.geometry.PointCloud(tensor_map)

        return new_pcd


    def _get_tile_labels(self, tile_id, aois_tiles, aois_labels):

        aoi = self._get_aoi_from_tile_id(tile_id, aois_tiles)
        ind_aoi = self._get_ind_aoi_tiles(tile_id, aoi, aois_labels)
                
        return self.aois_labels[aoi][ind_aoi] if ind_aoi is not None else gpd.GeoDataFrame()
    
    def _get_aoi_from_tile_id(self, tile_id, aois_tiles):
        
        for aoi in ["train", "valid", "test"]:
            tile_ids = aois_tiles[aoi]
            if tile_id in tile_ids:
                return aoi
        
        assert KeyError(f"Tile {tile_id} not found in any AOI")

    def _get_ind_aoi_tiles(self, tile_id, aoi, aois_labels):
        for i, labels in enumerate(aois_labels[aoi]):
            if tile_id in labels['tile_id'].values:
                return i

        assert KeyError(f"Tile {tile_id} not found in any AOI")
        

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
    
    
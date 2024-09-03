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

class LabeledPointCloudTilerizer(LabeledTilerizer):
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
                 coco_categories_list: List[dict] or None =None ) -> None:
        
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


        self.labels = self._load_labels(geopackage_layer=geopackage_layer, main_label_category=main_label_category_column, other_labels_attributes=other_labels_attributes_column)

    def _load_labels(self, 
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
    
    def generate_coco_labels(self):
        """
        Generate the tiles and the COCO dataset(s) for each AOI (or for the 'all' AOI) and save everything to the disk.
        """
        aoi_tiles, aois_labels = self._get_aoi_labels()

        coco_paths = {}
        for aoi in aois_labels:
            if aoi == 'all' and len(aois_labels.keys()) > 1:
                # don't save the 'all' tiles if aois were provided.
                continue

            labels = aois_labels[aoi]

            if len(labels) == 0:
                print(f"No tiles found for AOI {aoi}, skipping...")
                continue

            if len(labels) == 0:
                print(f"No tiles found for AOI {aoi}. Skipping...")
                continue

            tiles_metadata = TileMetadataCollection([tile for tile in self.tiles_metadata if tile.id in aoi_tiles[aoi]])
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

            coco_generator.generate_coco()
            coco_paths[aoi] = coco_output_file_path

        return coco_paths


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
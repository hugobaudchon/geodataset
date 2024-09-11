import warnings
from typing import List

import xmltodict
import pandas as pd
import geopandas as gpd
from pathlib import Path

from shapely import box

from geodataset.geodata import Raster
from geodataset.utils import try_cast_multipolygon_to_polygon

from abc import ABC, abstractmethod

from typing import Union

class BaseLabels(ABC):
    @abstractmethod
    def _load_labels(self):
        pass

class PolygonLabels(BaseLabels):
    def __init__(self,
                 path: Union[str, Path],
                 geopackage_layer_name: str = None,
                 main_label_category_column: str = None,
                 other_labels_attributes_column: List[str] = None,
                 keep_categories: List[str] = None):

        self.path = Path(path)
        self.ext = self.path.suffix
        self.geopackage_layer_name = geopackage_layer_name
        self.main_label_category_column_name = main_label_category_column
        self.other_labels_attributes_column_names = other_labels_attributes_column
        self.main_label_category_column_name = main_label_category_column

        gdf = self._load_labels()
        
        self.geometries_gdf = gdf[gdf.Label.isin(keep_categories)]

    def _load_labels(self):
        # Loading the labels into a GeoDataFrame
        labels_gdf = self._load_gdf_labels()
        # Making sure we are working with Polygons and not Multipolygons
        labels_gdf = self._check_multipolygons(labels_gdf)

        labels_gdf = self._remove_overlapping_polygons(labels_gdf)
        # Making sure the labels polygons are valid (they are not None)
        labels_gdf = self._check_valid_polygons(labels_gdf)

        return labels_gdf
    
    def _load_gdf_labels(self):
        if self.ext.lower() == '.xml':
            labels_gdf = self._load_xml_labels()
        elif self.ext == '.csv':
            labels_gdf = self._load_csv_labels()
        elif self.ext in ['.geojson', '.gpkg', '.shp', ".json"]:
            labels_gdf = self._load_geopandas_labels()
        else:
            raise Exception(f'Annotation format {self.ext} is not yet supported.')
        return labels_gdf

        

    def _check_valid_polygons(self, labels_gdf):
        n_labels = len(labels_gdf)
        labels_gdf = labels_gdf.dropna(subset=['geometry'])
        if n_labels != len(labels_gdf):
            warnings.warn(f"Removed {n_labels - len(labels_gdf)} out of {n_labels} labels as they have 'None' geometries.")

        n_labels = len(labels_gdf)
        labels_gdf = labels_gdf[labels_gdf.geometry.area > 0]
        if n_labels != len(labels_gdf):
            warnings.warn(f"Removed {n_labels - len(labels_gdf)} out of {n_labels} labels as they have an area of 0.")

        return labels_gdf
    
    def _check_multipolygons(self, labels_gdf):
        if (labels_gdf['geometry'].type == 'MultiPolygon').any():
            labels_gdf['geometry'] = labels_gdf['geometry'].astype(object).apply(try_cast_multipolygon_to_polygon)
            n_poly_before = len(labels_gdf)
            labels_gdf = labels_gdf.dropna(subset=['geometry'])
            warnings.warn(f"Removed {n_poly_before - len(labels_gdf)} out of {n_poly_before} labels as they are MultiPolygons"
                          f" that can't be cast to Polygons.")
        return labels_gdf
    
    def _load_geopandas_labels(self):
        # Load polygons

        if self.ext in ['.geojson', '.gpkg', '.shp', '.json']:
            labels_gdf = gpd.read_file(self.path, layer=self.geopackage_layer_name)

        else:
            raise ValueError("Unsupported file format for polygons. Please use GeoJSON (.geojson), JSON (.json), GPKG (.gpkg) or Shapefile (.shp).")

        if self.main_label_category_column_name:
            assert self.main_label_category_column_name in labels_gdf, \
                f'Could not find the main category {self.main_label_category_column_name}'\
                f' in the geopackage. Please manually double check the geopackage polygons attributes'\
                f' or change the value of parameter \'main_label_category_column_name\'.'\
                f' The columns of the geopackages are: {labels_gdf.columns}'

        if self.other_labels_attributes_column_names:
            for attribute in self.other_labels_attributes_column_names:
                assert attribute in labels_gdf, \
                    f'Could not find the attribute {attribute}' \
                    f' in the geopackage. Please manually double check the geopackage polygons attributes' \
                    f' or remove the attribute from the parameter \'other_labels_attributes_column_names\'.' \
                    f' The columns of the geopackages are: {labels_gdf.columns}'

        return labels_gdf

    def _load_xml_labels(self):
        with open(self.path, 'r') as annotation_file:
            annotation = xmltodict.parse(annotation_file.read())
        labels_bboxes = []
        labels_main_categories = []
        labels_other_attributes = {attribute: [] for attribute in self.other_labels_attributes_column_names}\
            if self.other_labels_attributes_column_names else {}
        if isinstance(annotation['annotation']['object'], list):
            for bbox in annotation['annotation']['object']:
                xmin = int(bbox['bndbox']['xmin'])
                ymin = int(bbox['bndbox']['ymin'])
                xmax = int(bbox['bndbox']['xmax'])
                ymax = int(bbox['bndbox']['ymax'])
                labels_bboxes.append(box(xmin, ymin, xmax, ymax))
                labels_main_categories, labels_other_attributes = self._find_label_attributes_in_xml(
                    bbox_xml_object=bbox,
                    labels_main_categories=labels_main_categories,
                    labels_other_attributes=labels_other_attributes
                )
        else:
            bbox = annotation['annotation']['object']
            xmin = int(bbox['bndbox']['xmin'])
            ymin = int(bbox['bndbox']['ymin'])
            xmax = int(bbox['bndbox']['xmax'])
            ymax = int(bbox['bndbox']['ymax'])
            labels_bboxes.append(box(xmin, ymin, xmax, ymax))
            labels_main_categories, labels_other_attributes = self._find_label_attributes_in_xml(
                bbox_xml_object=bbox,
                labels_main_categories=labels_main_categories,
                labels_other_attributes=labels_other_attributes
            )

        labels_gdf = gpd.GeoDataFrame(geometry=labels_bboxes)
        # Adding main categories and other attributes in labels_gdf
        if self.main_label_category_column_name:
            labels_gdf['main_category'] = pd.Series(labels_main_categories)
        if self.other_labels_attributes_column_names:
            for attribute, values in labels_other_attributes.items():
                labels_gdf[attribute] = pd.Series(values)

        return labels_gdf

    def _find_label_attributes_in_xml(self,
                                      bbox_xml_object: dict,
                                      labels_main_categories: list,
                                      labels_other_attributes: dict):
        if self.main_label_category_column_name:
            if self.main_label_category_column_name in bbox_xml_object:
                labels_main_categories.append(bbox_xml_object[self.main_label_category_column_name])
            else:
                raise Exception(f'Could not find the main category {self.main_label_category_column_name}'
                                f' in the annotation. Please manually double check the XML labels file'
                                f' or change the value of parameter \'main_label_category_column_name\'.')

        if self.other_labels_attributes_column_names:
            for attribute in self.other_labels_attributes_column_names:
                if attribute in bbox_xml_object:
                    labels_other_attributes[attribute].append(bbox_xml_object[attribute])
                else:
                    raise Exception(f'Could not find the attribute {attribute} in the annotation.'
                                    f' Please manually double check the XML labels file or remove that attribute'
                                    f' value from parameter \'other_labels_attributes_column_names\'.')

        return labels_main_categories, labels_other_attributes

    def _load_csv_labels(self):
        labels_df = pd.read_csv(self.path)

        raster_name_prefix = self.associated_raster.name.split('.')[-2]

        if 'img_name' in labels_df and 'Xmin' in labels_df and raster_name_prefix in set(labels_df['img_name']):
            # For wwf_ecuador cleaned annotations
            labels_df = labels_df[labels_df['img_name'] == raster_name_prefix]
            labels_bbox = labels_df[['Xmin', 'Ymin', 'Xmax', 'Ymax']].values.tolist()
        else:
            labels_df = labels_df[labels_df['img_path'] == self.associated_raster.path]
            labels_bbox = labels_df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()

        labels_gdf = gpd.GeoDataFrame(labels_df, geometry=[box(*bbox) for bbox in labels_bbox])
        if self.main_label_category_column_name:
            if self.main_label_category_column_name in labels_df:
                labels_gdf[self.main_label_category_column_name] = labels_df[self.main_label_category_column_name]
            else:
                raise Exception(f'Could not find the main category column {self.main_label_category_column_name}'
                                f' in the csv. Please manually double check the CSV columns'
                                f' or change the value of parameter \'main_label_category_column_name\'.'
                                f' The columns of the CSV are: {labels_df.columns}')

        if self.other_labels_attributes_column_names:
            for attribute in self.other_labels_attributes_column_names:
                if attribute in labels_df:
                    labels_gdf[attribute] = labels_df[attribute]
                else:
                    raise Exception(f'Could not find the attribute column {attribute} in the CSV.'
                                    f' Please manually double check the CSV columns or remove that attribute'
                                    f' value from parameter \'other_labels_attributes_column_names\'.'
                                    f' The columns of the CSV are: {labels_df.columns}')

        return labels_gdf


    def _remove_overlapping_polygons(self, labels_gdf):

        labels_gdf['id'] = labels_gdf.index

        overlap_polygons = labels_gdf.sjoin(labels_gdf, how='inner', predicate="overlaps")
        
        ind = overlap_polygons.id_left.values
        labels_gdf = labels_gdf[~(labels_gdf.id.isin(ind))]
        print(f"Removing {ind.shape[0]} overlapping polygons")


        overlay = gpd.overlay(labels_gdf, labels_gdf, how='intersection')
        
        overlap_set = set()
        for ind, each in overlay.iterrows():
            if(each.id_1 != each.id_2):
                overlap_set.add(each.id_1)
                overlap_set.add(each.id_2)

        ind = list(overlap_set)
        labels_gdf = labels_gdf[~(labels_gdf.id.isin(ind))]

        print(f"Removing {len(ind)} overlapping polygons")

        return labels_gdf

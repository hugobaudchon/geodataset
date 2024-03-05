import warnings
from typing import List

import xmltodict
import pandas as pd
import geopandas as gpd
from pathlib import Path

from shapely import box, Polygon, MultiPolygon

from geodataset.geodata import Raster


class RasterPolygonLabels:
    def __init__(self,
                 path: Path,
                 associated_raster: Raster,
                 ground_resolution: float = None,
                 scale_factor: float = None,
                 main_label_category_column_name: str = None,
                 other_labels_attributes_column_names: List[str] = None):

        self.path = path
        self.ext = self.path.suffix
        self.associated_raster = associated_raster
        self.ground_resolution = ground_resolution
        self.scale_factor = scale_factor
        self.main_label_category_column_name = main_label_category_column_name
        self.other_labels_attributes_column_names = other_labels_attributes_column_names

        assert self.ground_resolution == self.associated_raster.ground_resolution, \
            "The specified ground_resolution for the labels and Raster are different."
        assert self.scale_factor == self.associated_raster.scale_factor, \
            "The specified scale_factor for the labels and Raster are different."

        self.labels_gdf = self._load_labels()

    def _load_labels(self):
        # Loading the labels into a GeoDataFrame
        if self.ext.lower() == '.xml':
            labels_gdf = self._load_xml_labels()
        elif self.ext == '.csv':
            labels_gdf = self._load_csv_labels()
        elif self.ext in ['.geojson', '.gpkg', '.shp']:
            labels_gdf = self._load_geopandas_labels()
        else:
            raise Exception(f'Annotation format {self.ext} is not yet supported.')

        # Making sure we are working with Polygons and not Multipolygons
        if (labels_gdf['geometry'].type == 'MultiPolygon').any():
            labels_gdf['geometry'] = labels_gdf['geometry'].astype(object).apply(self.try_cast_multipolygon_to_polygon)
            n_poly_before = len(labels_gdf)
            labels_gdf = labels_gdf.dropna(subset=['geometry'])
            warnings.warn(f"Removed {n_poly_before - len(labels_gdf)} out of {n_poly_before} labels as they are MultiPolygons"
                          f" that can't be cast to Polygons.")

        # Making sure the labels and associated raster CRS are matching.
        labels_gdf = self.associated_raster.adjust_geometries_to_raster_crs_if_necessary(gdf=labels_gdf)

        # Scaling the geometries to pixel coordinates aligned with the Raster
        labels_gdf = self.associated_raster.adjust_geometries_to_raster_pixel_coordinates(gdf=labels_gdf)

        # Checking if most of the labels are intersecting the Raster.
        # If not, something probably went wrong with the CRS, transform or scaling factor.
        raster_gdf = gpd.GeoDataFrame(data={'geometry': [box(0,
                                                             0,
                                                             self.associated_raster.metadata['height'],
                                                             self.associated_raster.metadata['width'])]})
        labels_gdf['label_area'] = labels_gdf.geometry.area
        intersecting_polygons = gpd.overlay(raster_gdf, labels_gdf, how='intersection')
        labels_in_raster_ratio = 1 - (len(intersecting_polygons) - len(labels_gdf))/len(labels_gdf)
        assert labels_in_raster_ratio > 0.95, (f"{(1-len(labels_gdf))*100}% of labels are not in the raster,"
                                               f" the raster and labels are probably not aligned anymore"
                                               f" due to CRS, transform or scale_factor.")

        return labels_gdf

    def _load_geopandas_labels(self):
        # Load polygons
        if self.ext in ['.geojson', '.gpkg', '.shp']:
            labels_gdf = gpd.read_file(self.path)
        else:
            raise ValueError("Unsupported file format for polygons. Please use GeoJSON (.geojson), GPKG (.gpkg) or Shapefile (.shp).")

        if self.main_label_category_column_name:
            assert self.main_label_category_column_name in labels_gdf, \
                f'Could not find the main category {self.main_label_category_column_name}'\
                f' in the geopackage. Please manually double check the geopackage polygons attributes'\
                f' or change the value of parameter \'main_label_category_column_name\'.'\
                f' The columns of the geopackages are: {labels_gdf.columns}'

        if self.other_labels_attributes_column_names:
            for attribute in self.other_labels_attributes_column_names:
                assert self.main_label_category_column_name in labels_gdf, \
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

    @staticmethod
    def try_cast_multipolygon_to_polygon(geometry):
        if isinstance(geometry, Polygon):
            return geometry
        elif isinstance(geometry, MultiPolygon):
            polygons = list(geometry.geoms)
            if len(polygons) == 1:
                return Polygon(polygons[0])
            else:
                return None
        else:
            # Return None if the geometry is neither Polygon nor MultiPolygon
            return None



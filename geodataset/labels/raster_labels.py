import warnings
from functools import partial

import rasterio
import xmltodict
import pandas as pd
import geopandas as gpd
from pathlib import Path

from shapely import box, Polygon, MultiPolygon
from shapely.ops import transform

from geodataset.geodata import Raster


class RasterPolygonLabels:
    SUPPORTED_TASKS = ['detection', 'segmentation']

    def __init__(self,
                 path: Path,
                 associated_raster: Raster,
                 task: str,
                 scale_factor: float = 1.0):

        self.path = path
        self.ext = self.path.suffix
        self.associated_raster = associated_raster
        self.scale_factor = scale_factor
        self.task = task

        assert task in self.SUPPORTED_TASKS, \
            f"The 'task' value provided is not valid. Valid values are {self.SUPPORTED_TASKS}"
        assert self.scale_factor == self.associated_raster.scale_factor, \
            "The specified scale_factor for the labels and Raster are different."

        self.labels_gdf = self._load_labels()

    def _load_labels(self):
        # Loading the labels into a GeoDataFrame
        if self.ext.lower() == '.xml':
            labels_gdf, is_bbox = self._load_xml_labels()
        elif self.ext == '.csv':
            labels_gdf, is_bbox = self._load_csv_labels()
        elif self.ext in ['.geojson', '.gpkg', '.shp']:
            labels_gdf, is_bbox = self._load_geopandas_labels()
        else:
            raise Exception(f'Annotation format {self.ext} is not yet supported.')

        # Making sure the labels and associated raster CRS are matching.
        if labels_gdf.crs != self.associated_raster.metadata['crs']:
            if labels_gdf.crs and self.associated_raster.metadata['crs']:
                labels_gdf.set_crs(self.associated_raster.metadata['crs'])
            elif labels_gdf.crs and not self.associated_raster.metadata['crs']:
                raise Exception(f"The labels have a CRS but no the Raster."
                                f" Please verify the correct raster path was set")

        # Making sure we are working with Polygons and not Multipolygons
        if (labels_gdf['geometry'].type == 'MultiPolygon').any():
            labels_gdf['geometry'] = labels_gdf['geometry'].astype(object).apply(self.try_cast_multipolygon_to_polygon)
            n_poly_before = len(labels_gdf)
            labels_gdf = labels_gdf.dropna(subset=['geometry'])
            warnings.warn(f"Removed {n_poly_before - len(labels_gdf)} out of {n_poly_before} labels as they are MultiPolygons"
                          f" that can't be cast to Polygons.")

        # Scaling the geometries to pixel coordinates aligned with the Raster
        if labels_gdf.crs:
            # If the labels have a CRS, their geometries are in CRS coordinates,
            # so we need to apply the inverse of the Raster transform to get pixel coordinates.
            # This also applies the scaling_factor as the Raster is supposedly already scaled too.
            inverse_transform = ~self.associated_raster.metadata['transform']

            def transform_coord(x, y, transform_fct):
                # Applying the inverse transform to the coordinate
                x, y = transform_fct * (x, y)
                return x, y

            labels_gdf['geometry'] = labels_gdf['geometry'].astype(object).apply(
                lambda geom: transform(partial(transform_coord, transform_fct=inverse_transform), geom)
            )
            labels_gdf.crs = None
        else:
            # If the labels don't have a CRS, we expect them to already be in pixel coordinates.
            # So we just need to apply the scaling factor.
            labels_gdf['geometry'] = labels_gdf['geometry'].astype(object).apply(
                lambda geom: Polygon([(x * self.scale_factor, y * self.scale_factor) for x, y in geom.exterior.coords])
            )

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
        return labels_gdf, None#TODO

    def _load_xml_labels(self):
        with open(self.path, 'r') as annotation_file:
            annotation = xmltodict.parse(annotation_file.read())
        labels_bboxes = []
        if isinstance(annotation['annotation']['object'], list):
            for bbox in annotation['annotation']['object']:
                xmin = int(bbox['bndbox']['xmin'])
                ymin = int(bbox['bndbox']['ymin'])
                xmax = int(bbox['bndbox']['xmax'])
                ymax = int(bbox['bndbox']['ymax'])
                labels_bboxes.append(box(xmin, ymin, xmax, ymax))
        else:
            xmin = int(annotation['annotation']['object']['bndbox']['xmin'])
            ymin = int(annotation['annotation']['object']['bndbox']['ymin'])
            xmax = int(annotation['annotation']['object']['bndbox']['xmax'])
            ymax = int(annotation['annotation']['object']['bndbox']['ymax'])
            labels_bboxes.append(box(xmin, ymin, xmax, ymax))

        labels_gdf = gpd.GeoDataFrame(geometry=labels_bboxes)
        is_bbox = True

        return labels_gdf, is_bbox

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

        if 'group' in labels_df.columns:                    # TODO add parameter to specify category columns
            categories = labels_df['group'].to_numpy()
        else:
            categories = [None] * len(labels_bbox)

        if 'AGB' in labels_df.columns:                      # TODO add parameter to specify category columns
            agb = labels_df['AGB'].to_numpy()
        else:
            agb = None

        labels_gdf = gpd.GeoDataFrame(labels_df, geometry=[box(*bbox) for bbox in labels_bbox])
        is_bbox = True

        return labels_gdf, is_bbox

    def find_associated_labels(self,
                               window: rasterio.windows.Window,
                               min_intersection_ratio: float):
        intersecting_cropped_polygons = []
        for label in self.labels:
            # Calling label.is_bbox_in_window and get_cropped_bbox_polygon_label
            # instead of is_in_window and get_cropped_polygon_label as we are in a detection task and only care about bbox.
            # Those methods are much faster (about x5) than their counterparts.

            if label.is_bbox_in_window(window=window, min_intersection_ratio=min_intersection_ratio):
                intersecting_cropped_polygons.append(label.get_cropped_bbox_polygon_label(window=window))

        return intersecting_cropped_polygons

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



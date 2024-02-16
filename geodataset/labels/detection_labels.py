from abc import ABC, abstractmethod

import rasterio
import shapely
import xmltodict
import pandas as pd
import geopandas as gpd
from pathlib import Path

from geodataset.geodata import Raster
from geodataset.geodata.label import PolygonLabel
from geodataset.utils.geometry import try_cast_multipolygon_to_polygon


class RasterLabels(ABC):
    def __init__(self,
                 path: Path,
                 associated_raster: Raster,
                 scale_factor: float):
        self.path = path
        self.ext = path.suffix
        self.associated_raster = associated_raster
        self.scale_factor = scale_factor

        assert self.associated_raster.scale_factor == self.scale_factor, \
            ("The specified labels scale factor is different than the one used for the raster,"
             " this would render them unaligned.")

        self.labels = self._load_labels()

    @abstractmethod
    def _load_labels(self):
        pass

    @abstractmethod
    def find_associated_labels(self,
                               window: rasterio.windows.Window,
                               min_intersection_ratio: float):
        pass

    def _load_geopandas_labels(self):
        """
        Load polygons from a GeoJSON, GPKG or Shapefile, find their bounding box, and convert to pixel coordinates using a TIFF file.

        Parameters:
        - polygon_file_path: Path to the GeoJSON or GPKG file containing polygons.
        - tif_file_path: Path to a TIFF file for extracting CRS and converting coordinates to pixels.

        Returns:
        - A GeoDataFrame with polygons converted to pixel coordinates based on the TIFF file's CRS.
        """

        assert self.associated_raster.transform is not None, f"A {self.ext} label file was specified but the associated_geo_data does not contain a Transform."
        assert self.associated_raster.crs is not None, f"A {self.ext} label file was specified but the associated_geo_data does not contain a CRS."

        # Load polygons
        if self.ext in ['.geojson', '.gpkg', '.shp']:
            polygons = gpd.read_file(self.path)
        else:
            raise ValueError("Unsupported file format for polygons. Please use GeoJSON (.geojson), GPKG (.gpkg) or Shapefile (.shp).")

        labels = []
        n_multi_polygons_removed = 0
        for _, polygon in polygons.iterrows():
            if type(polygon['geometry']) is shapely.MultiPolygon:
                try:
                    polygon['geometry'] = try_cast_multipolygon_to_polygon(polygon['geometry'])
                except Exception:
                    n_multi_polygons_removed += 1
                    continue

            label = PolygonLabel(polygon=polygon['geometry'],
                                 category=polygon['Label'] if 'Label' in polygon else None,
                                 crs=polygons.crs)
            label.apply_crs(self.associated_raster.crs)
            # No need to call label.apply_scale_factor(...) as the raster transform already contains the scale_factor
            label.apply_crs_to_pixel_transform(self.associated_raster.transform)
            labels.append(label)

        print(f"Found {len(labels)} labels (polygons). "
              f"Had to remove {n_multi_polygons_removed} MultiPolygons as they are not supported yet.")

        return labels

    @staticmethod
    def _get_pixel_bbox(geom):
        minx, miny, maxx, maxy = geom.bounds
        return [minx, miny, maxx, maxy]


class RasterDetectionLabels(RasterLabels):
    def __init__(self,
                 path: Path,
                 associated_raster: Raster,
                 scale_factor: float):
        super().__init__(path, associated_raster, scale_factor)

    def _load_labels(self):
        if self.ext.lower() == '.xml':
            labels = self._load_xml_labels()
        elif self.ext == '.csv':
            labels = self._load_csv_labels()
        elif self.ext in ['.geojson', '.gpkg', '.shp']:
            labels = self._load_geopandas_labels()
        else:
            raise Exception('Annotation format {} not supported yet.'.format(self.ext))

        return labels

    def _load_xml_labels(self):
        with open(self.path, 'r') as annotation_file:
            annotation = xmltodict.parse(annotation_file.read())
        labels = []
        if isinstance(annotation['annotation']['object'], list):
            for bbox in annotation['annotation']['object']:
                xmin = int(bbox['bndbox']['xmin'])
                ymin = int(bbox['bndbox']['ymin'])
                xmax = int(bbox['bndbox']['xmax'])
                ymax = int(bbox['bndbox']['ymax'])

                label = PolygonLabel.from_bbox(bbox=[xmin, ymin, xmax, ymax], category=None)
                label.apply_scale_factor(self.scale_factor)
                labels.append(label)
        else:
            xmin = int(annotation['annotation']['object']['bndbox']['xmin'])
            ymin = int(annotation['annotation']['object']['bndbox']['ymin'])
            xmax = int(annotation['annotation']['object']['bndbox']['xmax'])
            ymax = int(annotation['annotation']['object']['bndbox']['ymax'])

            label = PolygonLabel.from_bbox(bbox=[xmin, ymin, xmax, ymax], category=None)
            label.apply_scale_factor(self.scale_factor)
            labels.append(label)

        return labels

    def _load_csv_labels(self):
        annots = pd.read_csv(self.path)

        file_name_prefix = self.associated_raster.name.split('.')[-2]

        if 'img_name' in annots and 'Xmin' in annots and file_name_prefix in set(annots['img_name']):
            annots = annots[annots['img_name'] == file_name_prefix]
            labels_bbox = annots[['Xmin', 'Ymin', 'Xmax', 'Ymax']].values.tolist()
        else:
            annots = annots[annots['img_path'] == self.associated_raster.path]
            labels_bbox = annots[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()

        if 'group' in annots.columns:
            categories = annots['group'].to_numpy()
        else:
            categories = [None] * len(labels_bbox)

        if 'AGB' in annots.columns:
            agb = annots['AGB'].to_numpy()
        else:
            agb = None

        labels = []
        for label_bbox, category in zip(labels_bbox, categories):
            label = PolygonLabel.from_bbox(bbox=label_bbox, category=category)
            label.apply_scale_factor(self.scale_factor)
            labels.append(label)

        return labels

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


class RasterSegmentationLabels(RasterLabels):
    def __init__(self,
                 path: Path,
                 associated_raster: Raster,
                 scale_factor: float):
        super().__init__(path, associated_raster, scale_factor)

    def _load_labels(self):
        if self.ext in ['.geojson', '.gpkg', '.shp']:
            labels = self._load_geopandas_labels()
        else:
            raise Exception('Annotation format {} not supported yet.'.format(self.ext))

        return labels

    def find_associated_labels(self,
                               window: rasterio.windows.Window,
                               min_intersection_ratio: float):

        intersecting_cropped_polygons = []
        for label in self.labels:
            if label.is_in_window(window=window, min_intersection_ratio=min_intersection_ratio):
                intersecting_cropped_polygons.append(label.get_cropped_polygon_label(window=window))

        return intersecting_cropped_polygons

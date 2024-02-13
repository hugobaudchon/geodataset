import xmltodict
import pandas as pd
import geopandas as gpd
from pathlib import Path

from geodataset.geodata import Raster


class RasterDetectionLabels:
    def __init__(self, path: Path, associated_geo_data: Raster, scale_factor: float):
        self.path = path
        self.ext = path.suffix
        self.associated_geo_data = associated_geo_data
        self.scale_factor = scale_factor

        (self.labels,
         self.categories,
         self.agb) = self._load_labels()

    def _load_labels(self):
        if self.ext.lower() == '.xml':
            labels, categories, agb = self._load_xml_labels()
        elif self.ext == '.csv':
            labels, categories, agb = self._load_csv_labels()
        elif self.ext in ['.geojson', '.gpkg', '.shp']:
            labels, categories, agb = self._load_geopandas_labels()
        else:
            raise Exception('Annotation format {} not supported yet.'.format(self.ext))

        return labels, categories, agb

    def _load_xml_labels(self):
        with open(self.path, 'r') as annotation_file:
            annotation = xmltodict.parse(annotation_file.read())
        labels = []
        if isinstance(annotation['annotation']['object'], list):
            for bbox in annotation['annotation']['object']:
                xmin = bbox['bndbox']['xmin']
                ymin = bbox['bndbox']['ymin']
                xmax = bbox['bndbox']['xmax']
                ymax = bbox['bndbox']['ymax']
                labels.append([float(xmin), float(ymin), float(xmax), float(ymax)])
        else:
            xmin = annotation['annotation']['object']['bndbox']['xmin']
            ymin = annotation['annotation']['object']['bndbox']['ymin']
            xmax = annotation['annotation']['object']['bndbox']['xmax']
            ymax = annotation['annotation']['object']['bndbox']['ymax']
            labels.append([float(xmin), float(ymin), float(xmax), float(ymax)])
        categories = None
        agb = None

        return labels, categories, agb

    def _load_csv_labels(self):
        annots = pd.read_csv(self.path)

        file_name_prefix = self.associated_geo_data.name.split('.')[-2]

        if 'img_name' in annots and 'Xmin' in annots and file_name_prefix in set(annots['img_name']):
            annots = annots[annots['img_name'] == file_name_prefix]
            labels = annots[['Xmin', 'Ymin', 'Xmax', 'Ymax']].values.tolist()
        else:
            annots = annots[annots['img_path'] == self.associated_geo_data.path]
            labels = annots[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()

        if 'group' in annots.columns:
            categories = annots['group'].to_numpy()
        else:
            categories = None
        if 'AGB' in annots.columns:
            agb = annots['AGB'].to_numpy()
        else:
            categories = None

        return labels, categories, agb

    def _load_geopandas_labels(self):
        """
        Load polygons from a GeoJSON, GPKG or Shapefile, find their bounding box, and convert to pixel coordinates using a TIFF file.

        Parameters:
        - polygon_file_path: Path to the GeoJSON or GPKG file containing polygons.
        - tif_file_path: Path to a TIFF file for extracting CRS and converting coordinates to pixels.

        Returns:
        - A GeoDataFrame with polygons converted to pixel coordinates based on the TIFF file's CRS.
        """

        assert self.associated_geo_data.transform is not None, f"A {self.ext} label file was specified but the associated_geo_data does not contain a Transform."
        assert self.associated_geo_data.crs is not None, f"A {self.ext} label file was specified but the associated_geo_data does not contain a CRS."

        # Load polygons
        if self.ext in ['.geojson', '.gpkg', '.shp']:
            polygons = gpd.read_file(self.path)
        else:
            raise ValueError("Unsupported file format for polygons. Please use GeoJSON (.geojson), GPKG (.gpkg) or Shapefile (.shp).")

        # Convert polygons to the same CRS as the TIFF
        if polygons.crs != self.associated_geo_data.crs:
            polygons = polygons.to_crs(self.associated_geo_data.crs)

        # Calculate bounding box for each polygon and convert to pixel coordinates
        def get_pixel_bbox(geom):
            minx, miny, maxx, maxy = geom.bounds
            # Convert the bounding box corners to pixel coordinates
            top_left = ~self.associated_geo_data.transform * (minx, maxy)
            bottom_right = ~self.associated_geo_data.transform * (maxx, miny)
            return [top_left[0], bottom_right[1], bottom_right[0], top_left[1]]

        labels_pixel_bounds = polygons.geometry.apply(get_pixel_bbox).tolist()
        categories = None
        agb = None

        return labels_pixel_bounds, categories, agb

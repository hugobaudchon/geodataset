import warnings
from typing import List

import xmltodict
import pandas as pd
import geopandas as gpd
from pathlib import Path

from shapely import box

from geodataset.geodata import Raster
from geodataset.utils import try_cast_multipolygon_to_polygon


class RasterPolygonLabels:
    """
    Class to handle the loading and processing of polygon labels associated with a :class:`~geodataset.geodata.Raster`.
    The labels will automatically be adjusted to the associated :class:`~geodataset.geodata.Raster`
    pixel coordinate system.
    For example, this class is instantiated by the :class:`~geodataset.tilerize.LabeledRasterTilerizer` to align the Raster and labels.

    Parameters
    ----------
    path: str or Path
        Path to the labels file. Supported formats are: .gpkg, .geojson, .shp, .xml, .csv.
        For .xml and .csv files, only specific formats are supported.
        Supported .xml and .csv files will also be converted to a GeoDataFrame for downstream uses.
    associated_raster: :class:`~geodataset.geodata.Raster`
        The instanced :class:`~geodataset.geodata.Raster` object associated with the labels.
        The labels will automatically be adjusted to this :class:`~geodataset.geodata.Raster` pixel coordinate system.
    geopackage_layer_name : str, optional
        The name of the layer in the geopackage file to use as labels. Only used if the labels_path is a .gpkg, .geojson
        or .shp file. Only useful when the labels geopackage file contains multiple layers.
    main_label_category_column_name : str, optional
        The name of the column in the labels file that contains the main category of the labels.
    other_labels_attributes_column_names : list of str, optional
        The names of the columns in the labels file that contains other attributes of the labels, which should be kept
        as a dictionary in the COCO annotations data.
    """

    def __init__(self,
                 path: str or Path or None,
                 associated_raster: Raster,
                 labels_gdf: gpd.GeoDataFrame = None,
                 geopackage_layer_name: str = None,
                 main_label_category_column_name: str = None,
                 other_labels_attributes_column_names: List[str] = None):
        if path:
            self.path = Path(path)
            self.ext = self.path.suffix
        else:
            self.path = path
            self.ext = None

        self.associated_raster = associated_raster
        self.labels_gdf = labels_gdf
        self.geopackage_layer_name = geopackage_layer_name
        self.ground_resolution = self.associated_raster.ground_resolution
        self.scale_factor = self.associated_raster.scale_factor
        self.main_label_category_column_name = main_label_category_column_name
        self.other_labels_attributes_column_names = other_labels_attributes_column_names

        assert not (self.path and self.labels_gdf is not None) and (self.path or self.labels_gdf is not None),\
            'Either path or labels_gdf must be passed to RasterPolygonLabels, but not both.'

        self.geometries_gdf = self._load_labels()

    def _load_labels(self) -> gpd.GeoDataFrame:
        # Loading the labels into a GeoDataFrame
        if self.labels_gdf is not None:
            labels_gdf = self.labels_gdf
        else:
            if self.ext.lower() == '.xml':
                labels_gdf = self._load_xml_labels()
            elif self.ext == '.csv':
                labels_gdf = self._load_csv_labels()
            elif self.ext in ['.geojson', '.gpkg', '.shp', ".json"]:
                labels_gdf = self._load_geopandas_labels()
            else:
                raise Exception(f'Annotation format {self.ext} is not yet supported.')

        # Making sure we are working with Polygons and not Multipolygons
        if (labels_gdf['geometry'].type == 'MultiPolygon').any():
            labels_gdf['geometry'] = labels_gdf['geometry'].astype(object).apply(
                lambda geom: try_cast_multipolygon_to_polygon(geom, strategy="largest_part")
            )
            n_poly_before = len(labels_gdf)

            labels_gdf = labels_gdf.dropna(subset=['geometry'])
            warnings.warn(f"Removed {n_poly_before - len(labels_gdf)} out of {n_poly_before} labels as they are MultiPolygons"
                          f" that can't be cast to Polygons.")

        # Strip Z (3D) if present, to avoid downstream 2D-only affine ops
        has_z = labels_gdf['geometry'].astype(object).apply(lambda g: getattr(g, "has_z", False))
        if has_z.any():
            n_z = int(has_z.sum())
            warnings.warn(f"Detected {n_z} geometries with Z-coordinates; dropping the Z dimension.")
            from shapely.ops import transform
            labels_gdf.loc[has_z, 'geometry'] = labels_gdf.loc[has_z, 'geometry'].apply(
                lambda geom: transform(lambda x, y, z=None: (x, y), geom)
            )

        # Making sure the labels and associated raster CRS are matching.
        labels_gdf = self.associated_raster.adjust_geometries_to_raster_crs_if_necessary(gdf=labels_gdf)

        # Making sure the labels polygons are valid (they are not None)
        n_labels = len(labels_gdf)
        labels_gdf = labels_gdf.dropna(subset=['geometry'])
        if n_labels != len(labels_gdf):
            warnings.warn(f"Removed {n_labels - len(labels_gdf)} out of {n_labels} labels as they have 'None' geometries.")

        # Scaling the geometries to pixel coordinates aligned with the Raster
        labels_gdf = self.associated_raster.adjust_geometries_to_raster_pixel_coordinates(gdf=labels_gdf)

        # Making sure the polygons have an area > 0
        n_labels = len(labels_gdf)
        labels_gdf = labels_gdf[labels_gdf.geometry.area > 0]
        if n_labels != len(labels_gdf):
            warnings.warn(f"Removed {n_labels - len(labels_gdf)} out of {n_labels} labels as they have an area of 0.")

        # Checking if most of the labels are intersecting the Raster.
        # If not, something probably went wrong with the CRS, transform or scaling factor.
        raster_gdf = gpd.GeoDataFrame(data={'geometry': [box(0,
                                                             0,
                                                             self.associated_raster.metadata['width'],
                                                             self.associated_raster.metadata['height'])]})

        labels_in_raster_gdf = labels_gdf[labels_gdf.intersects(raster_gdf.unary_union)]

        if len(labels_gdf) == 0:
            raise ValueError("Found no geometries in the labels geopackage file.")
        elif len(labels_in_raster_gdf) == 0:
            raise ValueError("Found no geometries in the labels geopackage file that intersect the raster.")
        elif len(labels_in_raster_gdf) / len(labels_gdf) < 0.10:
            warnings.warn(f"Less than 10% of the labels are intersecting the Raster."
                          f" Something probably went wrong with the CRS, transform or scaling factor.")

        return labels_in_raster_gdf

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
            labels_gdf[self.main_label_category_column_name] = pd.Series(labels_main_categories)
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

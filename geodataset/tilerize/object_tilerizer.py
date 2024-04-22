from functools import partial
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
from shapely import box
from shapely.affinity import translate
import geopandas as gpd

from geodataset.aoi import AOIConfig
from geodataset.geodata import Raster
from geodataset.labels import RasterPolygonLabels


class ObjectTilerizer:
    def __init__(self,
                 raster_path: Path,
                 labels_path: Path,
                 output_path: Path,
                 aois_config: AOIConfig or None,
                 ground_resolution: float or None,
                 scale_factor: float or None,
                 min_intersection_ratio: float,
                 output_size: int,
                 categories_coco: List[dict]):

        self.rasters_labels_configs = rasters_labels_configs
        self.gpkg_aoi = gpkg_aoi
        self.ground_resolution = ground_resolution
        self.scale_factor = scale_factor
        self.min_intersection_ratio = min_intersection_ratio
        self.output_size = output_size
        self.categories_coco = categories_coco
        self.augment_data = augment_data

        self.categories_mapping = self._setup_categories()
        self.rasters, self.polygons_labels, self.category_columns = self._load_rasters()
        self.aoi_polygons = self._load_aoi_polygon_for_each_raster()
        self.masks = self._load_masks_for_aoi()

    def _setup_categories(self):
        categories_mapping = {}
        for category in self.categories_coco:
            categories_mapping[category['name']] = category['id']
            if category['other_names']:
                for other_name in category['other_names']:
                    categories_mapping[other_name] = category['id']

        return categories_mapping

    def _load_rasters(self):
        rasters = {}
        polygons_labels = {}
        category_columns = {}
        for raster_config in self.rasters_labels_configs:
            raster = Raster(path=raster_config['raster_path'],
                            ground_resolution=self.ground_resolution,
                            scale_factor=self.scale_factor)
            labels = RasterPolygonLabels(path=raster_config['labels_path'],
                                         associated_raster=raster,
                                         main_label_category_column_name=raster_config['main_label_category_column_name'],
                                         other_labels_attributes_column_names=None)

            rasters[raster_config['raster_path']] = raster
            polygons_labels[raster_config['raster_path']] = labels.geometries_gdf
            category_columns[raster_config['raster_path']] = labels.main_label_category_column_name

        return rasters, polygons_labels, category_columns

    def _load_aoi_polygon_for_each_raster(self):
        aoi_polygon = gpd.read_file(self.gpkg_aoi)
        aoi_polygons = {}
        for raster_path in self.rasters.keys():
            aoi_polygon_for_raster = aoi_polygon.copy()
            aoi_polygon_for_raster = self.rasters[raster_path].adjust_geometries_to_raster_crs_if_necessary(gdf=aoi_polygon_for_raster)
            aoi_polygon_for_raster = self.rasters[raster_path].adjust_geometries_to_raster_pixel_coordinates(gdf=aoi_polygon_for_raster)
            aoi_polygons[raster_path] = aoi_polygon_for_raster
        return aoi_polygons

    @staticmethod
    def create_centroid_centered_mask(polygon, output_size):
        binary_mask = np.zeros((output_size, output_size), dtype=np.uint8)
        x, y = polygon.centroid.coords[0]
        x, y = int(x), int(y)
        mask_box = box(x - 0.5 * output_size,
                       y - 0.5 * output_size,
                       x + 0.5 * output_size,
                       y + 0.5 * output_size)

        polygon_intersection = mask_box.intersection(polygon)

        translated_polygon_intersection = translate(
            polygon_intersection,
            xoff=-mask_box.bounds[0],
            yoff=-mask_box.bounds[1]
        )

        contours = np.array(translated_polygon_intersection.exterior.coords).reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(binary_mask, [contours], 1)

        return binary_mask, mask_box

    def _load_masks_for_aoi(self):
        id = 0
        masks = {}
        for raster_path in self.rasters.keys():
            raster = self.rasters[raster_path]
            polygons = self.polygons_labels[raster_path].copy()

            # Get the polygons masks inside the aoi using min_intersection_ratio
            aoi_polygon = self.aoi_polygons[raster_path]
            polygons['polygons_id'] = polygons.index
            polygons['polygons_area'] = polygons.geometry.area

            intersections = gpd.overlay(polygons, aoi_polygon, how='intersection')
            intersections['intersection_area'] = intersections.geometry.area
            intersections['intersecting_ratio'] = intersections['intersection_area'] / intersections['polygons_area']

            # Filter geometries based on the threshold percentage
            intersections = intersections[intersections['intersecting_ratio'] > self.min_intersection_ratio]
            polygons = polygons.loc[polygons['polygons_id'].isin(intersections['polygons_id'])]

            polygons['centroid'] = polygons.centroid
            polygons['temp'] = polygons['geometry'].astype(object).apply(
                partial(self.create_centroid_centered_mask, output_size=self.output_size)
            )
            # Split the temporary column into the desired two columns
            polygons['mask'] = polygons['temp'].astype(object).apply(lambda x: x[0])
            polygons['mask_box'] = polygons['temp'].astype(object).apply(lambda x: x[1])

            # Drop the temporary column
            polygons.drop(columns=['temp'], inplace=True)

            for _, polygon_row in polygons.iterrows():
                mask_box = polygon_row['mask_box']
                mask_bounds = mask_box.bounds
                mask_bounds = [int(x) for x in mask_bounds]

                data = raster.data[
                       :3,                                  # removing Alpha channel if present
                       mask_bounds[1]:mask_bounds[3],
                       mask_bounds[0]:mask_bounds[2]
                ]

                masked_data = data * polygon_row['mask']
                masked_data = masked_data / 255.0
                # masked_data += 1.0 - polygon_row['mask']

                label = polygon_row[self.category_columns[raster_path]]
                if label in self.categories_mapping:
                    masks[id] = {
                        'masked_data': masked_data,
                        'label': self.categories_mapping[label]
                    }
                    id += 1

        return masks

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        mask_data = self.masks[idx]
        label = mask_data['label']
        data = mask_data['masked_data']
        if self.augment_data:
            data = self.augmentation(image=data.transpose(1, 2, 0))['image']
            data = data.transpose(2, 0, 1)
        return data, label

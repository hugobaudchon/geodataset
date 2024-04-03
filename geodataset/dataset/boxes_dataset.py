from pathlib import Path
from typing import List

import numpy as np
from shapely import box
from shapely.affinity import translate

from geodataset.dataset.base_dataset import BaseDataset
from geodataset.geodata import Raster
from geodataset.labels import RasterPolygonLabels


class BoxesDataset(BaseDataset):
    def __init__(self,
                 raster_path: Path,
                 boxes_path: Path,
                 padding_percentage: float or None,
                 min_pixel_padding: int or None,
                 ground_resolution: float = None,
                 scale_factor: float = None,
                 ):

        self.raster_path = raster_path
        self.boxes_path = boxes_path
        self.ground_resolution = ground_resolution
        self.scale_factor = scale_factor
        self.padding_percentage = padding_percentage
        self.min_pixel_padding = min_pixel_padding

        assert not (ground_resolution and scale_factor), ("Both a ground_resolution and a scale_factor were provided."
                                                          " Please only specify one.")

        self.raster = self._load_raster()
        self.boxes = self._load_boxes(main_label_category_column_name=None,
                                      other_labels_attributes_column_names=None)

    def _load_raster(self):
        raster = Raster(path=self.raster_path,
                        ground_resolution=self.ground_resolution,
                        scale_factor=self.scale_factor)
        return raster

    def _load_polygons(self,
                       main_label_category_column_name: str or None,
                       other_labels_attributes_column_names: List[str] or None):

        labels = RasterPolygonLabels(path=self.boxes_path,
                                     associated_raster=self.raster,
                                     main_label_category_column_name=main_label_category_column_name,
                                     other_labels_attributes_column_names=other_labels_attributes_column_names)

        return labels

    def __getitem__(self, idx: int):
        # Extract the geometry at the given index
        geom = self.boxes.geometries_gdf.iloc[idx].geometry

        # Calculate padding based on the padding percentage and minimum pixel padding
        minx, miny, maxx, maxy = geom.bounds
        x_padding = max(self.min_pixel_padding, (
                    maxx - minx) * self.padding_percentage / 2) if self.padding_percentage else self.min_pixel_padding
        y_padding = max(self.min_pixel_padding, (
                    maxy - miny) * self.padding_percentage / 2) if self.padding_percentage else self.min_pixel_padding

        # Adjust the bounds with initial padding
        padded_minx, padded_miny, padded_maxx, padded_maxy = minx - x_padding, miny - y_padding, maxx + x_padding, maxy + y_padding

        # Determine the current dimensions including initial padding
        current_width = padded_maxx - padded_minx
        current_height = padded_maxy - padded_miny

        # Calculate additional padding required to make the subarray square
        if current_width > current_height:
            difference = current_width - current_height
            padded_miny -= difference / 2
            padded_maxy += difference / 2
        elif current_height > current_width:
            difference = current_height - current_width
            padded_minx -= difference / 2
            padded_maxx += difference / 2

        padded_minx, padded_miny, padded_maxx, padded_maxy = (int(padded_minx), int(padded_miny),
                                                              int(padded_maxx), int(padded_maxy))

        # Extract the subarray from the raster using the pixel bounds
        # Ensure to handle the case where bounds may go beyond the raster dimensions
        subarray_miny = int(max(padded_miny, 0))
        subarray_minx = int(max(padded_minx, 0))
        subarray_maxy = int(max(subarray_miny, min(padded_maxy, self.raster.data.shape[1])))
        subarray_maxx = int(max(subarray_minx, min(padded_maxx, self.raster.data.shape[2])))

        subarray = self.raster.data[:, subarray_miny:subarray_maxy, subarray_minx:subarray_maxx]

        final_padding = ((0, 0),
                         (subarray_miny - padded_miny, padded_maxy - subarray_maxy),
                         (subarray_minx - padded_minx, padded_maxx - subarray_maxx))

        subarray = np.pad(subarray, final_padding, mode='constant', constant_values=0)

        # Adjust the original geometry to the new image's coordinate system
        # This is done by translating the original geometry to the origin based on the minx and miny of the padded box
        adjusted_box = translate(box(*geom.bounds), xoff=-padded_minx, yoff=-padded_miny)

        return subarray, adjusted_box, (padded_minx, padded_miny)

    def __len__(self):
        return len(self.boxes.geometries_gdf)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


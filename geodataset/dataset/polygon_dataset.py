from pathlib import Path
from typing import List

import numpy as np
from shapely.affinity import translate
from shapely.geometry import box

from geodataset.dataset.base_dataset import BaseDataset
from geodataset.geodata import Raster
from geodataset.labels import RasterPolygonLabels


class FixedImageSizePolygonDataset(BaseDataset):
    def __init__(self,
                 raster_path: Path,
                 polygons_path: Path,
                 image_size: int,
                 ground_resolution: float = None,
                 scale_factor: float = None,
                 ):
        self.raster_path = raster_path
        self.polygons_path = polygons_path
        self.image_size = image_size
        self.ground_resolution = ground_resolution
        self.scale_factor = scale_factor

        assert not (ground_resolution and scale_factor), ("Both a ground_resolution and a scale_factor were provided."
                                                          " Please only specify one.")

        self.raster = self._load_raster()
        self.polygons = self._load_polygons()

    def _load_raster(self):
        raster = Raster(path=self.raster_path,
                        ground_resolution=self.ground_resolution,
                        scale_factor=self.scale_factor)
        return raster

    def _load_polygons(self):
        labels = RasterPolygonLabels(path=self.polygons_path,
                                     associated_raster=self.raster)
        return labels

    def __getitem__(self, idx: int):
        # Extract the polygon geometry at the given index, already in pixel coordinates
        geom = self.polygons.geometries_gdf.iloc[idx].geometry

        # Get the bounding box of the polygon
        minx, miny, maxx, maxy = geom.bounds

        # Center of the polygon
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2

        # Calculate the bounds for cropping around the center
        half_size = self.image_size / 2
        crop_minx = int(max(center_x - half_size, 0))
        crop_miny = int(max(center_y - half_size, 0))
        crop_maxx = int(min(center_x + half_size, self.raster.data.shape[2]))
        crop_maxy = int(min(center_y + half_size, self.raster.data.shape[1]))

        # Crop the raster to these bounds
        cropped_raster = self.raster.data[:, crop_miny:crop_maxy, crop_minx:crop_maxx]

        # Create a new box for the cropped area and translate the geometry to be centered
        translated_geom = translate(geom, xoff=-crop_minx, yoff=-crop_miny)

        return cropped_raster, translated_geom, (crop_minx, crop_miny)

    def __len__(self):
        return len(self.polygons.geometries_gdf)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

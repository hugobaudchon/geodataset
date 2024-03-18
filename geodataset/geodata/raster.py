import warnings
from functools import partial
from pathlib import Path

import numpy as np
import rasterio.windows
import geopandas as gpd
from shapely import Polygon
from shapely.ops import transform

from geodataset.geodata.base_geodata import BaseGeoData
from geodataset.geodata.tile import Tile
from geodataset.utils import read_raster, apply_affine_transform


class Raster(BaseGeoData):
    """
    Typically a .tif or .png raster with or without CRS/transform data.
    """
    def __init__(self, path: Path, ground_resolution: float = None, scale_factor: float = None):
        self.path = path
        self.name = path.name
        self.ext = path.suffix

        assert not (ground_resolution and scale_factor), ("Both a ground_resolution and a scale_factor were provided."
                                                          " Please only specify one.")
        self.ground_resolution = ground_resolution
        self.scale_factor = scale_factor

        (self.data,
         self.metadata,
         self.x_scale_factor,
         self.y_scale_factor) = self._load_data()

    def _load_data(self):
        data, metadata, x_scale_factor, y_scale_factor = read_raster(
            path=self.path,
            ground_resolution=self.ground_resolution,
            scale_factor=self.scale_factor
        )

        if 'crs' not in metadata:
            metadata['crs'] = None
        if 'transform' not in metadata:
            metadata['transform'] = None

        if not metadata['crs']:
            warnings.warn(f'Could not find a CRS in the raster file {self.name}.')
        if not metadata['transform']:
            warnings.warn(f'Could not find a transform in the raster file {self.name}.')

        return data, metadata, x_scale_factor, y_scale_factor

    def get_tile(self,
                 window: rasterio.windows.Window,
                 product_name: str,
                 tile_id: int or None) -> Tile or None:
        tile_data = self.data[
                    :,
                    window.row_off:window.row_off + window.height,
                    window.col_off:window.col_off + window.width]

        pad_row = window.width - tile_data.shape[1]
        pad_col = window.height - tile_data.shape[2]

        if pad_row > 0 or pad_col > 0:
            padding = ((0, 0), (0, pad_row), (0, pad_col))  # No padding for bands, pad rows and columns as needed
            tile_data = np.pad(tile_data, padding, mode='constant', constant_values=0)

        window_transform = rasterio.windows.transform(window, self.metadata['transform'])

        tile_metadata = {
            'driver': 'GTiff',
            'height': window.height,
            'width': window.width,
            'count': self.metadata['count'],
            'dtype': self.metadata['dtype'],
            'crs': self.metadata['crs'],
            'transform': window_transform
        }

        tile = Tile(data=tile_data, metadata=tile_metadata, product_name=product_name,
                    ground_resolution=self.ground_resolution, scale_factor=self.scale_factor,
                    row=window.row_off, col=window.col_off, tile_id=tile_id)

        return tile

    def adjust_geometries_to_raster_crs_if_necessary(self, gdf: gpd.GeoDataFrame):
        if gdf.crs != self.metadata['crs']:
            if gdf.crs and self.metadata['crs']:
                gdf = gdf.to_crs(self.metadata['crs'])
            elif gdf.crs and not self.metadata['crs']:
                raise Exception(f"The geometries have a CRS but not the Raster."
                                f" Please verify the correct raster path was set")
        return gdf

    def adjust_geometries_to_raster_pixel_coordinates(self, gdf: gpd.GeoDataFrame):
        # Scaling the geometries to pixel coordinates aligned with the Raster
        if gdf.crs:
            # If the labels have a CRS, their geometries are in CRS coordinates,
            # so we need to apply the inverse of the Raster transform to get pixel coordinates.
            # This also applies the scaling_factor as the Raster is supposedly already scaled too.
            inverse_transform = ~self.metadata['transform']

            gdf['geometry'] = gdf['geometry'].astype(object).apply(
                lambda geom: apply_affine_transform(geom, inverse_transform)
            )
            gdf.crs = None
        elif self.scale_factor:
            # If the labels don't have a CRS, we expect them to already be in pixel coordinates.
            # So we just need to apply the scaling factor.
            gdf['geometry'] = gdf['geometry'].astype(object).apply(
                lambda geom: Polygon(
                    [(x * self.scale_factor, y * self.scale_factor) for x, y in geom.exterior.coords])
            )
        elif self.ground_resolution:
            gdf['geometry'] = gdf['geometry'].astype(object).apply(
                lambda geom: Polygon(
                    [(x * self.x_scale_factor, y * self.y_scale_factor) for x, y in geom.exterior.coords])
            )

        return gdf


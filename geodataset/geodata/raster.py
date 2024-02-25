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
from geodataset.utils import read_raster


class Raster(BaseGeoData):
    """
    Typically a .tif or .png raster with or without CRS/transform data.
    """
    def __init__(self, path: Path, scale_factor: float = 1.0):
        self.path = path
        self.name = path.name
        self.ext = path.suffix

        self.scale_factor = scale_factor

        (self.data,
         self.metadata) = self._load_data()

    def _load_data(self):
        data, metadata = read_raster(path=self.path, scale_factor=self.scale_factor)
        if 'crs' not in metadata:
            metadata['crs'] = None
        if 'transform' not in metadata:
            metadata['transform'] = None

        if not metadata['crs']:
            warnings.warn(f'Could not find a CRS in the raster file {self.name}.')
        if not metadata['transform']:
            warnings.warn(f'Could not find a transform in the raster file {self.name}.')

        return data, metadata

    def get_tile(self,
                 window: rasterio.windows.Window,
                 dataset_name: str,
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

        tile = Tile(data=tile_data, metadata=tile_metadata, dataset_name=dataset_name,
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

            def transform_coord(x, y, transform_fct):
                # Applying the inverse transform to the coordinate
                x, y = transform_fct * (x, y)
                return x, y

            gdf['geometry'] = gdf['geometry'].astype(object).apply(
                lambda geom: transform(partial(transform_coord, transform_fct=inverse_transform), geom)
            )
            gdf.crs = None
        else:
            # If the labels don't have a CRS, we expect them to already be in pixel coordinates.
            # So we just need to apply the scaling factor.
            gdf['geometry'] = gdf['geometry'].astype(object).apply(
                lambda geom: Polygon(
                    [(x * self.scale_factor, y * self.scale_factor) for x, y in geom.exterior.coords])
            )

        return gdf


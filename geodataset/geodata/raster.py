import warnings
from pathlib import Path
import rasterio.windows

from geodataset.geodata.base_geodata import BaseGeoData
from geodataset.geodata.tile import Tile
from geodataset.utils.io import read_raster


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
         self.crs,
         self.transform,
         self.metadata) = self._load_data()

    def _load_data(self):
        data, metadata = read_raster(path=self.path, scale_factor=self.scale_factor)

        if 'crs' not in metadata or not metadata['crs']:
            warnings.warn(f'Could not find a CRS in the raster file {self.name}.')
            crs = None
        else:
            crs = metadata['crs']
        if 'transform' not in metadata or not metadata['transform']:
            warnings.warn(f'Could not find a transform in the raster file {self.name}.')
            transform = None
        else:
            transform = metadata['transform']

        return data, crs, transform, metadata

    def get_tile(self,
                 window: rasterio.windows.Window,
                 dataset_name: str) -> Tile or None:
        tile_data = self.data[
                       :,
                       window.row_off:window.row_off + window.height,
                       window.col_off:window.col_off + window.width]

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
                    row=window.row_off, col=window.col_off)

        return tile



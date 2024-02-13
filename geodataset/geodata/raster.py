import warnings
from pathlib import Path

from geodataset.geodata.base_geodata import BaseGeoData
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


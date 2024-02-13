from pathlib import Path
import rasterio

from geodataset.utils.io import read_raster, read_point_cloud


class BaseGeoData:
    """
    Contains base methods and attributes for any type of Geo Data (raster or point cloud)
    """
    def __init__(self, path: Path):
        self.data = None
        self.crs = None
        self.transform = None
        self.path = path
        self.name = path.name
        self.ext = path.suffix

        self._load_data()

    def _load_data(self):
        pass

    def get_resampled_data(self, scale_factor: int):
        pass


class GeoReferencedRaster(BaseGeoData):
    """
    Typically a .tif raster with a CRS.
    """
    def __init__(self, path: Path):
        super().__init__(path=path)

    def _load_data(self):
        data, crs, transform = read_raster(path=self.path, ext=self.ext)

        assert crs is not None, ('Could not find a crs in the source file, '
                                 'please verify the file or create a NonGeoReferencedRaster instead.')
        assert transform is not None, ('Could not find a transform in the source file, '
                                       'please verify the file or create a NonGeoReferencedRaster instead.')

        self.data = data
        self.crs = crs
        self.transform = transform


class NonGeoReferencedRaster(BaseGeoData):
    """
    Typically a .png raster or a .tif without a CRS.
    """
    def __init__(self, path: Path):
        super().__init__(path=path)

    def _load_data(self):
        data, crs, transform = read_raster(path=self.path, ext=self.ext)

        self.data = data
        self.crs = None
        self.transform = None


class PointCloud(BaseGeoData):
    """
    Typically a .las or .laz point cloud file.
    """
    def __init__(self, path: Path):
        super().__init__(path=path)

    def _load_data(self):
        data, crs, transform = read_point_cloud(path=self.path, ext=self.ext)

        self.data = data
        self.crs = crs
        self.transform = transform

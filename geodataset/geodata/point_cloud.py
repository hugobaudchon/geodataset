import warnings
from pathlib import Path

import numpy as np
import rasterio

from geodataset.geodata.base_geodata import BaseGeoData
from geodataset.utils.io import read_point_cloud


class PointCloud(BaseGeoData):
    """
    Typically a .las or .laz point cloud file.
    """
    def __init__(self, path: Path):
        self.path = path
        self.name = path.name
        self.ext = path.suffix

        self.data, self.crs, self.transpose = self._load_data()

    def _load_data(self):
        data, crs, transform = read_point_cloud(path=self.path)

        if not crs:
            warnings.warn(f'Could not find a CRS in the raster file {self.name}.')
        if not transform:
            warnings.warn(f'Could not find a transform in the raster file {self.name}.')

        return data, crs, transform

    def _create_sub_pointcloud(self, nb_max_points=100000):
        if self.data is None:
            print('*****')
            print('Cannot create sub point cloud, the following file does not exists: {}'.format(self.path))
            print('*****')
            return None
        x_window, y_window = self._define_pc_window(nb_max_points)
        num_rows = int(np.ceil(self.data.x.max() - self.data.x.min()))
        num_cols = int(np.ceil(self.data.y.max() - self.data.y.min()))
        samples = []
        print('Full area size: ({}, {})'.format(num_rows, num_cols))
        print('Desired x window size: ', x_window)
        print('Desired y window size: ', y_window)

        print('Saving sub point clouds')
        for row in range(0, num_rows, x_window):
            for col in range(0, num_cols, y_window):
                window = rasterio.windows.Window(col, row, y_window, x_window)
                x_min = self.data.x.min() + window.row_off
                y_min = self.data.y.min() + window.col_off
                sub_pc = self.data[np.where((self.data.x >= x_min) &
                                            (self.data.x < x_min + window.height) &
                                            (self.data.y >= y_min) &
                                            (self.data.y < y_min + window.width))[0]]

                samples.append(sub_pc)

        return samples

    def _define_pc_window(self, nb_max_points):
        total_nb_points = len(self.data.x)
        x_min = self.data.x.min()
        x_max = self.data.x.max()
        y_min = self.data.y.min()
        y_max = self.data.y.max()
        x_dist = 1
        y_dist = 1
        if total_nb_points <= nb_max_points:
            x_dist = int(np.ceil(x_max - x_min))
            y_dist = int(np.ceil(y_max - y_min))
            return x_dist, y_dist
        while len(np.where((self.data.x <= x_min + x_dist) & (self.data.y <= y_min + y_dist))[0]) <= nb_max_points:
            x_dist += 1
            y_dist += 1
        # remove the last iteration
        x_dist -= 1
        y_dist -= 1
        return x_dist, y_dist

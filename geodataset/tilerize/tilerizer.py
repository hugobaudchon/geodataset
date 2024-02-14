import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

from geodataset.geodata import Raster
from geodataset.labels import RasterDetectionLabels
from geodataset.utils.io import save_tile


class RasterDetectionTilerizer:
    def __init__(self,
                 dataset_name: str,
                 raster_path: Path,
                 labels_path: Path,
                 output_path: Path,
                 scale_factor=1.0):
        """
        raster_path: Path,
            path to the raster (.tif, .png...)
        labels_path: Path,
            path to the labels (.geojson, .gpkg, .csv...)
        output_path: Path,
            path to parent folder where to save the image tiles and associated labels
        scale_factor: float
            Rescaling the data (change pixel resolution)
        intersection_ratio: float,
            When finding the associated labels to a tile, this ratio will specify the minimal required intersection
            ratio between a candidate polygon and the tile in order to keep this polygon as a label for that tile.
        ignore_mostly_black_or_white_tiles: bool,
            Whether to ignore (skip) mostly black or white (>50%) tiles
        """
        self.dataset_name = dataset_name
        self.raster_path = raster_path
        self.labels_path = labels_path
        self.scale_factor = scale_factor
        self.min_intersection_ratio = min_intersection_ratio
        self.ignore_mostly_black_or_white_tiles = ignore_mostly_black_or_white_tiles

        self.output_path = output_path
        self.tiles_path = self.output_path / 'tiles'
        self.tiles_path.mkdir(parents=True, exist_ok=True)

        (self.raster,
         self.labels) = self._load_data()

    def _load_data(self):
        raster = Raster(path=self.raster_path,
                        scale_factor=self.scale_factor)
        labels = RasterDetectionLabels(path=self.labels_path,
                                       associated_raster=raster,
                                       scale_factor=self.scale_factor)
        return raster, labels

    def _write_paths(self, sample_paths):
        dataset_paths = pd.DataFrame(sample_paths, columns=['paths'])
        file_name = self.dataset_name + '_paths.csv'
        dataset_paths.to_csv(self.tiles_path.parent / file_name, index=False)

    def create_tiles(self, tile_size=1024, overlap=0, start_counter_tile=0):
        width = self.raster.metadata['width']
        height = self.raster.metadata['height']
        print('Desired tile size: ', tile_size)
        tile_id = start_counter_tile
        print('Saving tiles')
        samples = []
        for row in range(0, width, int((1 - overlap) * tile_size)):
            print(f'\t Row {row}/{width}')
            for col in range(0, height, int((1 - overlap) * tile_size)):
                window = rasterio.windows.Window(col, row, tile_size, tile_size)
                tile = self.raster.data[
                         :,
                         window.row_off:window.row_off + window.height,
                         window.col_off:window.col_off + window.width]

                if self.ignore_mostly_black_or_white_tiles:
                    # If it's >50% black pixels or white pixels, just continue. No point segmenting it.
                    if np.sum(tile == 0) / (tile_size * tile_size * self.raster.metadata['count']) > 0.5:
                        continue
                    if np.sum(tile == 255) / (tile_size * tile_size * self.raster.metadata['count']) > 0.5:
                        continue

                window_transform = rasterio.windows.transform(window, self.raster.metadata['transform'])

                tile_metadata = {
                    'driver': 'GTiff',
                    'height': tile_size,
                    'width': tile_size,
                    'count': self.raster.metadata['count'],
                    'dtype': self.raster.metadata['dtype'],
                    'crs': self.raster.metadata['crs'],
                    'transform': window_transform
                }

                save_tile(output_folder=self.output_path,
                          tile=tile,
                          tile_metadata=tile_metadata,
                          row=row,
                          col=col,
                          dataset_name=self.dataset_name,
                          fold='TEST')

        tile_id += 1

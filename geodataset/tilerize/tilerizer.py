import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

from geodataset.geodata import Raster
from geodataset.labels import RasterDetectionLabels


class RasterDetectionTilerizer:
    def __init__(self,
                 dataset_name: str,
                 raster_path: Path,
                 labels_path: Path,
                 output_path: Path,
                 scale_factor: float = 1.0,
                 min_intersection_ratio: float = 0.9,
                 ignore_tiles_without_labels: bool = False,
                 ignore_mostly_black_or_white_tiles: bool = True):
        """
        raster_path: Path,
            Path to the raster (.tif, .png...).
        labels_path: Path,
            Path to the labels (.geojson, .gpkg, .csv...).
        output_path: Path,
            Path to parent folder where to save the image tiles and associated labels.
        scale_factor: float,
            Scale factor for rescaling the data (change pixel resolution).
        intersection_ratio: float,
            When finding the associated labels to a tile, this ratio will specify the minimal required intersection
            ratio between a candidate polygon and the tile in order to keep this polygon as a label for that tile.
        ignore_tiles_without_labels: bool,
            Whether to ignore (skip) tiles that don't have any associated labels.
        ignore_mostly_black_or_white_tiles: bool,
            Whether to ignore (skip) mostly black or white (>50%) tiles.
        """
        self.dataset_name = dataset_name
        self.raster_path = raster_path
        self.labels_path = labels_path
        self.scale_factor = scale_factor
        self.min_intersection_ratio = min_intersection_ratio
        self.ignore_tiles_without_labels = ignore_tiles_without_labels
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
        print('Raster size: ', (width, height))
        print('Desired tile size: ', tile_size)
        tile_id = start_counter_tile
        print('Saving tiles')
        samples = []
        for row in range(0, width, int((1 - overlap) * tile_size)):
            print(f'\t Row {row}/{width}')
            for col in range(0, height, int((1 - overlap) * tile_size)):
                window = rasterio.windows.Window(col, row, tile_size, tile_size)
                tile = self.raster.get_tile(window=window,
                                            dataset_name=self.dataset_name)
                if self.ignore_mostly_black_or_white_tiles:
                    # If it's >50% black pixels or white pixels, just continue. No point segmenting it.
                    if np.sum(tile.data == 0) / (tile_size * tile_size * self.raster.metadata['count']) > 0.5:
                        continue
                    if np.sum(tile.data == 255) / (tile_size * tile_size * self.raster.metadata['count']) > 0.5:
                        continue

                associated_labels = self.labels.find_associated_labels(window=window,
                                                                       min_intersection_ratio=self.min_intersection_ratio)

                if self.ignore_tiles_without_labels and not associated_labels:
                    continue

                tile.save(output_folder=self.tiles_path)

        tile_id += 1

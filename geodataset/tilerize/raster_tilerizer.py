from typing import List
import numpy as np
import rasterio
from pathlib import Path

from geodataset.geodata import Raster
from geodataset.geodata.tile import Tile


class RasterTilerizer:
    def __init__(self,
                 dataset_name: str,
                 raster_path: Path,
                 output_path: Path,
                 scale_factor: float = 1.0,
                 ignore_mostly_black_or_white_tiles: bool = True):
        """
        raster_path: Path,
            Path to the raster (.tif, .png...).
        output_path: Path,
            Path to parent folder where to save the image tiles and associated labels.
        scale_factor: float,
            Scale factor for rescaling the data (change pixel resolution).
        ignore_mostly_black_or_white_tiles: bool,
            Whether to ignore (skip) mostly black or white (>50%) tiles.
        """
        self.dataset_name = dataset_name
        self.raster_path = raster_path
        self.scale_factor = scale_factor
        self.ignore_mostly_black_or_white_tiles = ignore_mostly_black_or_white_tiles

        self.output_path = output_path
        self.tiles_path = self.output_path / self.dataset_name / 'tiles'
        self.tiles_path.mkdir(parents=True, exist_ok=True)

        self.raster = self._load_data()

    def _load_data(self):
        raster = Raster(path=self.raster_path,
                        scale_factor=self.scale_factor)
        return raster

    def _create_tiles(self, tile_size, overlap) -> List[Tile]:
        width = self.raster.metadata['width']
        height = self.raster.metadata['height']
        print('Raster size: ', (height, width))
        print('Desired tile size: ', tile_size)
        print('Creating tiles and finding their associated labels...')
        tiles = []
        for row in range(0, height, int((1 - overlap) * tile_size)):
            print(f'\t Row {row}/{height}')
            for col in range(0, width, int((1 - overlap) * tile_size)):
                window = rasterio.windows.Window(col, row, tile_size, tile_size)
                tile = self.raster.get_tile(window=window,
                                            dataset_name=self.dataset_name)
                if self.ignore_mostly_black_or_white_tiles:
                    # If it's >50% black pixels or white pixels, just continue. No point segmenting it.
                    if np.sum(tile.data == 0) / (tile_size * tile_size * self.raster.metadata['count']) > 0.5:
                        continue
                    if np.sum(tile.data == 255) / (tile_size * tile_size * self.raster.metadata['count']) > 0.5:
                        continue

                tiles.append(tile)

        return tiles

    def generate_tiles(self, tile_size=1024, overlap=0):
        tiles = self._create_tiles(tile_size=tile_size, overlap=overlap)

        # Save the tile images
        for tile in tiles:
            tile.save(output_folder=self.tiles_path)

        print(f"The tiles has been saved to {self.tiles_path}.")

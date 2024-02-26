from abc import ABC
from typing import List
import numpy as np
import rasterio
from pathlib import Path
from typing import cast

from geodataset.aoi import AOIGenerator, AOIFromPackage
from geodataset.aoi import AOIConfig, AOIGeneratorConfig, AOIFromPackageConfig
from geodataset.geodata import Raster
from geodataset.geodata.tile import Tile
from geodataset.utils import save_aois_tiles_picture


class BaseRasterTilerizer(ABC):
    def __init__(self,
                 dataset_name: str,
                 raster_path: Path,
                 output_path: Path,
                 scale_factor: float = 1.0,
                 ignore_black_white_alpha_tiles_threshold: float = 0.8):
        """
        raster_path: Path,
            Path to the raster (.tif, .png...).
        output_path: Path,
            Path to parent folder where to save the image tiles and associated labels.
        scale_factor: float,
            Scale factor for rescaling the data (change pixel resolution).
        ignore_black_white_alpha_tiles_threshold: bool,
            Whether to ignore (skip) mostly black or white (>ignore_black_white_alpha_tiles_threshold%) tiles.
        """
        self.dataset_name = dataset_name
        self.raster_path = raster_path
        self.scale_factor = scale_factor
        self.ignore_black_white_alpha_tiles_threshold = ignore_black_white_alpha_tiles_threshold

        self.output_path = output_path / self.dataset_name
        self.tiles_path = self.output_path / 'tiles'
        self.tiles_path.mkdir(parents=True, exist_ok=True)

        self.raster = self._load_raster()

    def _load_raster(self):
        raster = Raster(path=self.raster_path,
                        scale_factor=self.scale_factor)
        return raster

    def _create_tiles(self,
                      tile_size: int,
                      overlap: float,
                      start_counter_tile: int) -> List[Tile]:
        width = self.raster.metadata['width']
        height = self.raster.metadata['height']

        print('Raster size: ', (height, width))
        print('Desired tile size: ', tile_size)
        print('Creating tiles and finding their associated labels...')

        tile_id_counter = start_counter_tile
        tiles = []
        for row in range(0, height, int((1 - overlap) * tile_size)):
            print(f'\t Row {row}/{height}')
            for col in range(0, width, int((1 - overlap) * tile_size)):
                window = rasterio.windows.Window(col, row, tile_size, tile_size)
                tile = self.raster.get_tile(window=window,
                                            dataset_name=self.dataset_name,
                                            tile_id=tile_id_counter)

                if self._check_skip_tile(tile=tile, tile_size=tile_size):
                    continue

                tiles.append(tile)
                tile_id_counter += 1

        return tiles

    def _get_tiles_per_aoi(self,
                           aois_config: AOIConfig,
                           tile_size: int,
                           overlap: float,
                           start_counter_tile: int):

        tiles = self._create_tiles(tile_size=tile_size, overlap=overlap, start_counter_tile=start_counter_tile)

        if aois_config is not None:
            if type(aois_config) is AOIGeneratorConfig:
                aoi_engine = AOIGenerator(tiles=tiles,
                                          tile_size=tile_size,
                                          tile_overlap=overlap,
                                          aois_config=cast(AOIGeneratorConfig, aois_config))
            elif type(aois_config) is AOIFromPackageConfig:
                aoi_engine = AOIFromPackage(tiles=tiles,
                                            tile_size=tile_size,
                                            tile_overlap=overlap,
                                            aois_config=cast(AOIFromPackageConfig, aois_config),
                                            associated_raster=self.raster,
                                            scale_factor=self.scale_factor)
            else:
                raise Exception(f'aois_config type unsupported: {type(aois_config)}')

            aois_tiles = aoi_engine.get_aoi_tiles()
        else:
            aois_tiles = {}

        aois_tiles['all'] = tiles

        return aois_tiles

    def _check_skip_tile(self, tile, tile_size):
        is_rgb = self.raster.data.shape[0] == 3
        is_rgba = self.raster.data.shape[0] == 4
        skip_ratio = self.ignore_black_white_alpha_tiles_threshold

        # Checking if the tile has more than a certain ratio of white, black, or alpha pixels.
        if is_rgb:
            if np.sum(tile.data == 0) / (tile_size * tile_size * 3) > skip_ratio:
                return True
            if np.sum(tile.data == 255) / (tile_size * tile_size * 3) > skip_ratio:
                return True
        elif is_rgba:
            if np.sum(tile.data[:-1] == 0) / (tile_size * tile_size * 3) > skip_ratio:
                return True
            if np.sum(tile.data[:-1] == 255) / (tile_size * tile_size * 3) > skip_ratio:
                return True
            if np.sum(tile.data[-1] == 0) / (tile_size * tile_size * 3) > skip_ratio:
                return True

        return False


class RasterTilerizer(BaseRasterTilerizer):
    def __init__(self,
                 dataset_name: str,
                 raster_path: Path,
                 output_path: Path,
                 scale_factor: float = 1.0,
                 ignore_black_white_alpha_tiles_threshold: float = 0.8):
        super().__init__(dataset_name=dataset_name,
                         raster_path=raster_path,
                         output_path=output_path,
                         scale_factor=scale_factor,
                         ignore_black_white_alpha_tiles_threshold=ignore_black_white_alpha_tiles_threshold)

    def generate_tiles(self, tile_size=1024, overlap=0, start_counter_tile=1, aois_config: AOIConfig = None):
        aois_tiles = self._get_tiles_per_aoi(aois_config=aois_config,
                                             tile_size=tile_size,
                                             overlap=overlap,
                                             start_counter_tile=start_counter_tile)

        tile_coordinate_step = int((1 - overlap) * tile_size)
        save_aois_tiles_picture(aois_tiles=aois_tiles,
                                save_path=self.output_path / 'aois_tiles.png',
                                tile_coordinate_step=tile_coordinate_step)

        for aoi in aois_tiles:
            if aoi == 'all' and len(aois_tiles.keys()) > 1:
                # don't save the 'all' tiles if aois were provided.
                continue

            # Save the tile images
            for tile in aois_tiles[aoi]:
                tile.save(output_folder=self.tiles_path)

        print(f"The tiles has been saved to {self.tiles_path}.")

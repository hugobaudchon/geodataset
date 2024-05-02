import concurrent
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import numpy as np
import rasterio
from shapely import box

from geodataset.utils import TileNameConvention, PolygonTileNameConvention


class Tile:
    def __init__(self,
                 data: np.ndarray,
                 metadata: dict,
                 product_name: str,
                 ground_resolution: float,
                 scale_factor: float,
                 row: int,
                 col: int,
                 tile_id: int):
        self.data = data
        self.metadata = metadata
        self.product_name = product_name
        self.row = row
        self.col = col
        self.tile_id = tile_id

        assert not (ground_resolution and scale_factor), ("Both a ground_resolution and a scale_factor were provided."
                                                          " Please only specify one.")
        self.scale_factor = scale_factor
        self.ground_resolution = ground_resolution

    @classmethod
    def from_path(cls, path: Path, tile_id):
        data, metadata, product_name, ground_resolution, scale_factor, row, col = Tile.load_tile(path)

        tile = cls(data=data,
                   metadata=metadata,
                   product_name=product_name,
                   ground_resolution=ground_resolution,
                   scale_factor=scale_factor,
                   row=row,
                   col=col,
                   tile_id=tile_id)

        return tile

    @staticmethod
    def load_tile(path: Path):
        ext = path.suffix
        if ext != '.tif':
            raise Exception(f'The tile extension should be \'.tif\'.')

        with rasterio.open(path) as src:
            data = src.read()
            metadata = src.profile

        product_name, ground_resolution, scale_factor, row, col = TileNameConvention.parse_name(path.name)

        return data, metadata, product_name, ground_resolution, scale_factor, row, col

    def save(self, output_folder: Path):
        assert output_folder.exists(), f"The output folder {output_folder} doesn't exist yet."

        tile_name = self.generate_name()

        with rasterio.open(
                output_folder / tile_name,
                'w',
                **self.metadata) as tile_raster:
            tile_raster.write(self.data)

    def generate_name(self):
        return TileNameConvention.create_name(product_name=self.product_name,
                                              ground_resolution=self.ground_resolution,
                                              scale_factor=self.scale_factor,
                                              row=self.row,
                                              col=self.col)

    def get_bbox(self):
        minx = self.col
        maxx = self.col + self.metadata['width']
        miny = self.row
        maxy = self.row + self.metadata['height']
        return box(minx, miny, maxx, maxy)


class PolygonTile:
    def __init__(self,
                 data: np.ndarray,
                 metadata: dict,
                 product_name: str,
                 ground_resolution: float,
                 scale_factor: float,
                 polygon_id: int):

        self.data = data
        self.metadata = metadata
        self.product_name = product_name
        self.polygon_id = polygon_id

        assert not (ground_resolution and scale_factor), ("Both a ground_resolution and a scale_factor were provided."
                                                          " Please only specify one.")
        self.scale_factor = scale_factor
        self.ground_resolution = ground_resolution

    @classmethod
    def from_path(cls, path: Path):
        data, metadata, product_name, ground_resolution, scale_factor, polygon_id = PolygonTile.load_tile(path)

        tile = cls(data=data,
                   metadata=metadata,
                   product_name=product_name,
                   ground_resolution=ground_resolution,
                   scale_factor=scale_factor,
                   polygon_id=polygon_id)

        return tile

    @staticmethod
    def load_tile(path: Path):
        ext = path.suffix
        if ext != '.tif':
            raise Exception(f'The tile extension should be \'.tif\'.')

        with rasterio.open(path) as src:
            data = src.read()
            metadata = src.profile

        product_name, ground_resolution, scale_factor, polygon_id = PolygonTileNameConvention.parse_name(path.name)

        return data, metadata, product_name, ground_resolution, scale_factor, polygon_id

    def save(self, output_folder: Path):
        assert output_folder.exists(), f"The output folder {output_folder} doesn't exist yet."

        tile_name = self.generate_name()

        with rasterio.open(
                output_folder / tile_name,
                'w',
                **self.metadata) as tile_raster:
            tile_raster.write(self.data)

    def generate_name(self):
        return PolygonTileNameConvention.create_name(
            product_name=self.product_name,
            ground_resolution=self.ground_resolution,
            scale_factor=self.scale_factor,
            polygon_id=self.polygon_id
        )


class TileSaver:
    def __init__(self, tiles_path: Path, n_workers: int):
        self.tiles_path = tiles_path
        self.n_workers = n_workers

    def save_tile(self, tile: Tile or PolygonTile):
        try:
            tile.save(output_folder=self.tiles_path)
        except Exception as e:
            print(f"Error saving tile {tile.tile_id}: {str(e)}")

    def save_all_tiles(self, tiles: List[Tile or PolygonTile]):
        # Use ThreadPoolExecutor to manage threads
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit tasks to the executor
            futures = [executor.submit(self.save_tile, tile) for tile in tiles]

            # Optionally, you can wait for all futures to complete and handle exceptions
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # If there was any exception it will be re-raised here
                except Exception as e:
                    print(f"An error occurred: {e}")

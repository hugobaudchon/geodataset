from pathlib import Path
import numpy as np
import rasterio

from geodataset.utils import TileNameConvention


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



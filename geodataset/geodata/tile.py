from pathlib import Path
import numpy as np
import rasterio

from geodataset.utils import TileNameConvention


class Tile:
    def __init__(self,
                 data: np.ndarray,
                 metadata: dict,
                 product_name: str,
                 scale_factor: float,
                 row: int,
                 col: int,
                 tile_id: int):
        self.data = data
        self.metadata = metadata
        self.product_name = product_name
        self.scale_factor = scale_factor
        self.row = row
        self.col = col
        self.tile_id = tile_id

    @classmethod
    def from_path(cls, path: Path, tile_id):
        data, metadata, product_name, scale_factor, row, col = Tile.load_tile(path)

        tile = cls(data=data,
                   metadata=metadata,
                   product_name=product_name,
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

        product_name, scale_factor, row, col = TileNameConvention.parse_name(path.name)

        return data, metadata, product_name, scale_factor, row, col

    def save(self, output_folder: Path):
        assert output_folder.exists(), f"The output folder {output_folder} doesn't exist yet."

        tile_name = TileNameConvention.create_name(product_name=self.product_name,
                                                   scale_factor=self.scale_factor,
                                                   row=self.row,
                                                   col=self.col)

        with rasterio.open(
                output_folder / tile_name,
                'w',
                **self.metadata) as tile_raster:
            tile_raster.write(self.data)

    def to_coco(self):
        """
        Generate a COCO-format dictionary for the tile image.

        Returns:
            dict: A dictionary formatted according to COCO specifications for an image.
        """

        # Extract entries from the metadata
        width = self.metadata['width']
        height = self.metadata['height']

        # Generate file name
        tile_name = TileNameConvention.create_name(product_name=self.product_name,
                                                   scale_factor=self.scale_factor,
                                                   row=self.row,
                                                   col=self.col)

        # Construct the COCO representation for the image
        coco_image = {
            "id": self.tile_id,
            "width": width,
            "height": height,
            "file_name": tile_name,
        }

        return coco_image



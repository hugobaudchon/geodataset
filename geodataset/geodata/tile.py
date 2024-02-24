import re
from pathlib import Path
import numpy as np
import rasterio


class Tile:
    TILE_NAME_PLACEHOLDERS_STRING = 'tile_{}_{}_{}.tif'
    TILE_NAME_REGEX_CONVENTION = r'tile_([a-zA-Z0-9]+)_(\d+)_(\d+)\.tif'

    def __init__(self,
                 data: np.ndarray,
                 metadata: dict,
                 dataset_name: str,
                 row: int,
                 col: int,
                 tile_id: int):
        self.data = data
        self.metadata = metadata
        self.dataset_name = dataset_name
        self.row = row
        self.col = col
        self.tile_id = tile_id

    @classmethod
    def from_path(cls, path: Path, tile_id):
        data, metadata, dataset_name, row, col = Tile.load_tile(path)

        return cls(data=data, metadata=metadata, dataset_name=dataset_name, row=row, col=col, tile_id=tile_id)

    @staticmethod
    def load_tile(path: Path):
        name = path.name
        ext = path.suffix
        if ext != '.tif':
            raise Exception(f'The tile extension should be \'.tif\'.')
        if not re.match(Tile.TILE_NAME_REGEX_CONVENTION, name):
            raise Exception(f'The tile name does not follow the convention '
                            f'\'{Tile.TILE_NAME_REGEX_CONVENTION}\'.')

        with rasterio.open(path) as src:
            data = src.read()
            metadata = src.profile

        dataset_name, row, col = name.split("_")[1:-1]

        return data, metadata, dataset_name, row, col

    def _get_tile_file_name(self):
        return Tile.TILE_NAME_PLACEHOLDERS_STRING.format(self.dataset_name, self.row, self.col)

    def save(self, output_folder: Path):
        assert output_folder.exists(), f"The output folder {output_folder} doesn't exist yet."

        tile_name = self._get_tile_file_name()

        assert re.match(Tile.TILE_NAME_REGEX_CONVENTION, tile_name), \
            (f'The generated tile_name \'{tile_name}\' doesn\'t respect the convention \'{Tile.TILE_NAME_REGEX_CONVENTION}\'.'
             f' Please make sure the dataset_name \'{self.dataset_name}\' only consists of characters and numbers.')

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
        crs = self.metadata['crs']
        transform = self.metadata['transform']

        # Generate file name using the internal method
        file_name = self._get_tile_file_name()

        # Construct the COCO representation for the image
        coco_image = {
            "id": self.tile_id,
            "width": width,
            "height": height,
            "file_name": file_name,
            "crs": {
                "format": ("EPSG" if crs.to_epsg() else "WKT") if crs else None,
                "value": (crs.to_epsg() if crs.to_epsg() else crs.to_wkt()) if crs else None
            },
            "affine_transform": tuple(transform) if transform else None
        }

        return coco_image



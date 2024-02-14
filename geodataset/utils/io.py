import re
from pathlib import Path

import laspy
import numpy as np
import rasterio
from rasterio.enums import Resampling


TILE_NAME_PLACEHOLDERS_STRING = 'tile_{}_{}_{}.tif'      # don't modify
TILE_NAME_REGEX_CONVENTION = r'tile_([a-zA-Z0-9]+)_(\d+)_(\d+)\.tif'     # don't modify


def read_raster(path: Path, scale_factor: float = 1.0):
    ext = path.suffix
    if ext in ['.tif', '.png', '.jpg']:
        with rasterio.open(path) as src:
            # resample data to target shape
            data = src.read(
                out_shape=(src.count,
                           int(src.height * scale_factor),
                           int(src.width * scale_factor)),
                resampling=Resampling.bilinear)

            if src.transform:
                # scale image transform
                new_transform = src.transform * src.transform.scale(
                    (src.width / data.shape[-1]),
                    (src.height / data.shape[-2]))
            else:
                new_transform = None

            new_profile = src.profile
            new_profile.update(transform=new_transform,
                               driver='GTiff',
                               height=data.shape[-2],
                               width=data.shape[-1])
    else:
        raise Exception(f'Data format {ext} not supported yet.')

    return data, new_profile


def read_point_cloud(path: Path):
    ext = path.suffix
    if ext in ['.las', '.laz']:
        with laspy.open(str(path)) as pc_file:
            data = pc_file.read()
            crs = None
            transform = None
            for vlr in data.vlrs:
                if isinstance(vlr, laspy.vlrs.known.WktCoordinateSystemVlr):
                    crs = vlr.parse_crs()
    else:
        raise Exception(f'Data format {ext} not supported yet.')

    return data, crs, transform


def save_tile(output_folder: Path,
              tile: np.ndarray,
              tile_metadata: dict,
              row: int,
              col: int,
              dataset_name: str):
    assert output_folder.exists(), f"The output folder {output_folder} doesn't exist yet."

    tile_name = TILE_NAME_PLACEHOLDERS_STRING.format(dataset_name, row, col)
    assert re.match(TILE_NAME_REGEX_CONVENTION, tile_name), \
        (f'The generated tile_name \n\t\'{tile_name}\'\n doesn\'t respect the convention \n\t\'{TILE_NAME_REGEX_CONVENTION}\'.\n'
         f' Please make sure the dataset_name \'{dataset_name}\' only consists of characters and numbers.')

    print(output_folder / tile_name)

    with rasterio.open(
            output_folder / tile_name,
            'w',
            **tile_metadata) as tile_raster:
        tile_raster.write(tile)


def load_tile(path: Path):
    name = path.name
    ext = path.suffix
    if ext != '.tif':
        raise Exception(f'The tile extension should be \'.tif\'.')
    if not re.match(TILE_NAME_REGEX_CONVENTION, name):
        raise Exception(f'The tile name does not follow the convention '
                        f'\'{TILE_NAME_REGEX_CONVENTION}\'.')

    with rasterio.open(path) as src:
        data = src.read()
        metadata = src.profile

    dataset_name, row, col = name.split("_")[1:-1]

    return data, metadata, dataset_name, row, col

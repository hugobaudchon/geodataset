from pathlib import Path

import laspy
import numpy as np
import rasterio
from PIL import Image
from einops import einops


def read_raster(path: Path, ext: str):
    if ext == '.png':
        with Image.open(path) as img:
            data = np.array(img)
            data = einops.rearrange(data, 'h w c -> c h w')
        crs = None
        transform = None
    elif ext == '.tif':
        with rasterio.open(path) as src:
            data = src.read()
            metadata = src.profile
            crs = metadata['crs']
            transform = metadata['transform']
    else:
        raise Exception(f'Data format {ext} not supported yet.')

    return data, crs, transform


def read_point_cloud(path: Path, ext: str):
    if ext in ('.las', '.laz'):
        with laspy.open(str(path)) as pc_file:
            data = pc_file.read()
        metadata = None
    else:
        raise Exception(f'Data format {ext} not supported yet.')

    crs = None              # TODO
    transform = None        # TODO

    return data, crs, transform

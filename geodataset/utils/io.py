from pathlib import Path

import laspy
import rasterio
from rasterio.enums import Resampling


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

import warnings
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import rasterio.windows
import geopandas as gpd
from shapely import Polygon, box
from shapely.affinity import translate

from geodataset.geodata.base_geodata import BaseGeoData
from geodataset.geodata.tile import Tile, PolygonTile
from geodataset.utils import read_raster, apply_affine_transform, validate_and_convert_product_name, \
    strip_all_extensions


class Raster(BaseGeoData):
    """
    Class responsible for loading a .tif or .png raster with or without CRS/transform data,
    while resampling it to match a given ground_resolution or scale_factor.
    It contains methods to align geometries in GeoDataFrame objects in a given CRS to the resampled raster's
    CRS or pixel coordinate system.

    Parameters
    ----------
    path: str or pathlib.Path
        The path to the raster file.
    output_name_suffix: str, optional
        A suffix to add to the output name of the tiles generated from this raster.
    ground_resolution : float
        The ground resolution in meter per pixel desired when loading the raster.
        Only one of ground_resolution and scale_factor can be set at the same time.
    scale_factor : float
        Scale factor for rescaling the data (change pixel resolution).
        Only one of ground_resolution and scale_factor can be set at the same time.
    """
    def __init__(self,
                 path: str or Path,
                 output_name_suffix: str = None,
                 ground_resolution: float = None,
                 scale_factor: float = None):
        self.path = Path(path)
        self.name = path.name
        self.ext = path.suffix
        self.product_name = validate_and_convert_product_name(strip_all_extensions(self.path))
        self.output_name = self.product_name + (f"_{output_name_suffix}" if output_name_suffix else "")

        assert not (ground_resolution and scale_factor), ("Both a ground_resolution and a scale_factor were provided."
                                                          " Please only specify one.")
        self.ground_resolution = ground_resolution
        self.scale_factor = scale_factor

        (self.data,
         self.metadata,
         self.x_scale_factor,
         self.y_scale_factor) = self._load_data()

    def _load_data(self):
        data, metadata, x_scale_factor, y_scale_factor = read_raster(
            path=self.path,
            ground_resolution=self.ground_resolution,
            scale_factor=self.scale_factor
        )

        if 'crs' not in metadata:
            metadata['crs'] = None
        if 'transform' not in metadata:
            metadata['transform'] = None

        if not metadata['crs']:
            warnings.warn(f'Could not find a CRS in the raster file {self.name}.')
        if not metadata['transform']:
            warnings.warn(f'Could not find a transform in the raster file {self.name}.')

        return data, metadata, x_scale_factor, y_scale_factor

    def get_tile(self,
                 window: rasterio.windows.Window,
                 tile_id: int or None) -> Tile or None:

        """
        Generates and returns a Tile from a window of the Raster.

        Parameters
        ----------
        window: rasterio.windows.Window
            The window to generate the Tile from.
        tile_id: int or None
            The id of the tile. Used in the tile name, and should be unique across a dataset.

        Returns
        -------
        Tile
        """

        tile_data = self.data[
                    :,
                    window.row_off:window.row_off + window.height,
                    window.col_off:window.col_off + window.width]

        pad_row = window.width - tile_data.shape[1]
        pad_col = window.height - tile_data.shape[2]

        if pad_row > 0 or pad_col > 0:
            padding = ((0, 0), (0, pad_row), (0, pad_col))  # No padding for bands, pad rows and columns as needed
            tile_data = np.pad(tile_data, padding, mode='constant', constant_values=0)

        window_transform = rasterio.windows.transform(window, self.metadata['transform'])

        tile_metadata = {
            'driver': 'GTiff',
            'height': window.height,
            'width': window.width,
            'count': self.metadata['count'],
            'dtype': self.metadata['dtype'],
            'crs': self.metadata['crs'],
            'transform': window_transform
        }

        tile = Tile(data=tile_data, metadata=tile_metadata, output_name=self.output_name,
                    ground_resolution=self.ground_resolution, scale_factor=self.scale_factor,
                    row=window.row_off, col=window.col_off, tile_id=tile_id)

        return tile

    def get_polygon_tile(self,
                         polygon: Polygon,
                         polygon_id: int,
                         tile_size: int,
                         use_variable_tile_size: bool,
                         variable_tile_size_pixel_buffer: int) -> Tuple[PolygonTile, Polygon]:

        """
        Generates and returns a PolygonTile from a Polygon geometry, along with the possibly cropped original Polygon.
        The PolygonTile (object representing an image containing the pixels of the Raster where the Polygon is)
        will be centered on the Polygon's centroid.
        The pixels of the generated PolygonTile that are outside the Polygon will be blacked-out (masked).

        Parameters
        ----------
        polygon: Polygon
            The Polygon geometry to generate the PolygonTile from.
        polygon_id: int
            The id of the polygon. Used in the tile name, and should be unique across a dataset.
        tile_size: int
            The size of the tile in pixels. If the polygon extent is larger than this value,
            then the returned Polygon will be cropped to the tile extent.
            If not, the returned Polygon will be identical to the source Polygon.
        use_variable_tile_size: bool
            If True, the tile size will be automatically adjusted to the Polygon's extent,
            up to the tile_size value, which in this case acts as a 'maximum size'.
            This avoids having too many blacked-out pixels around the Polygon of interest.
            It should be combined with the variable_tile_size_pixel_buffer parameter to add a
            fixed pixel buffer around the Polygon.
            This parameter is recommended if saving disk space is a concern.
        variable_tile_size_pixel_buffer: int
            The number of blacked-out pixels to add around the Polygon's extent as a buffer.
            Only used when use_variable_tile_size is set to 'True'.

        Returns
        -------
        Tuple[PolygonTile, Polygon]
        """

        x, y = polygon.centroid.coords[0]
        x, y = int(x), int(y)

        if use_variable_tile_size:
            max_dist_to_polygon_border = max([x - polygon.bounds[0], polygon.bounds[2] - x, y - polygon.bounds[1], polygon.bounds[3] - y])
            final_tile_size = int(min(tile_size, max_dist_to_polygon_border * 2 + variable_tile_size_pixel_buffer * 2))
        else:
            # Finding the box centered on the polygon's centroid
            final_tile_size = tile_size

        binary_mask = np.zeros((final_tile_size, final_tile_size), dtype=np.uint8)
        mask_box = box(x - 0.5 * final_tile_size,
                       y - 0.5 * final_tile_size,
                       x + 0.5 * final_tile_size,
                       y + 0.5 * final_tile_size)

        # Making sure the polygon is valid
        polygon = polygon.buffer(0)

        polygon_intersection = mask_box.intersection(polygon)

        translated_polygon_intersection = translate(
            polygon_intersection,
            xoff=-mask_box.bounds[0],
            yoff=-mask_box.bounds[1]
        )

        # Making sure the polygon is a Polygon and not a GeometryCollection (GeometryCollection can happen on Raster edges due to intersection)
        if translated_polygon_intersection.geom_type != 'Polygon':
            # Get the Polygon with the largest area
            translated_polygon_intersection = max(translated_polygon_intersection.geoms, key=lambda x: x.area)

        contours = np.array(translated_polygon_intersection.exterior.coords).reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(binary_mask, [contours], 1)

        # Getting the pixels from the raster
        mask_bounds = mask_box.bounds
        mask_bounds = [int(x) for x in mask_bounds]

        # Handling bounds outside the data
        start_row = max(mask_bounds[1], 0)
        end_row = min(mask_bounds[3], self.data.shape[1])
        start_col = max(mask_bounds[0], 0)
        end_col = min(mask_bounds[2], self.data.shape[2])

        data = self.data[
               :,
               start_row:end_row,
               start_col:end_col
               ]

        # Padding to tile_size if necessary
        pre_row_pad = max(0, -mask_bounds[1])
        post_row_pad = max(0, mask_bounds[3] - self.data.shape[1])
        pre_col_pad = max(0, -mask_bounds[0])
        post_col_pad = max(0, mask_bounds[2] - self.data.shape[2])

        if final_tile_size - (data.shape[1] + pre_row_pad + post_row_pad) == 1:
            post_row_pad += 1
        if final_tile_size - (data.shape[2] + pre_col_pad + post_col_pad) == 1:
            post_col_pad += 1

        data = np.pad(data, [(0, 0), (pre_row_pad, post_row_pad), (pre_col_pad, post_col_pad)],
                      mode='constant', constant_values=0)

        # Masking the pixels around the Polygon
        masked_data = data * binary_mask

        # Creating the PolygonTile, with the appropriate metadata
        window = rasterio.windows.Window(
            mask_box.bounds[0],
            mask_box.bounds[1],
            width=final_tile_size,
            height=final_tile_size
        )
        window_transform = rasterio.windows.transform(window, self.metadata['transform'])

        tile_metadata = {
            'driver': 'GTiff',
            'height': window.height,
            'width': window.width,
            'count': self.metadata['count'],
            'dtype': self.metadata['dtype'],
            'crs': self.metadata['crs'],
            'transform': window_transform
        }

        polygon_tile = PolygonTile(
            data=masked_data,
            metadata=tile_metadata,
            output_name=self.output_name,
            ground_resolution=self.ground_resolution,
            scale_factor=self.scale_factor,
            polygon_id=polygon_id
        )

        return polygon_tile, translated_polygon_intersection

    def adjust_geometries_to_raster_crs_if_necessary(self, gdf: gpd.GeoDataFrame):
        """
        Adjusts the geometries in a GeoDataFrame to the CRS of the Raster.

        Parameters
        ----------
        gdf: gpd.GeoDataFrame
            The GeoDataFrame containing the geometries to adjust.

        Returns
        -------
        gpd.GeoDataFrame
        """
        if gdf.crs != self.metadata['crs']:
            if gdf.crs and self.metadata['crs']:
                gdf = gdf.to_crs(self.metadata['crs'])
            elif gdf.crs and not self.metadata['crs']:
                raise Exception(f"The geometries have a CRS but not the Raster."
                                f" Please verify the correct raster path was set")
        return gdf

    def adjust_geometries_to_raster_pixel_coordinates(self, gdf: gpd.GeoDataFrame):
        """
        Adjusts the geometries in a GeoDataFrame to the pixel coordinates of the Raster.
        This should be called after calling the method adjust_geometries_to_raster_crs_if_necessary.

        Parameters
        ----------
        gdf: gpd.GeoDataFrame
            The GeoDataFrame containing the geometries to adjust.

        Returns
        -------
        gpd.GeoDataFrame
        """
        # Scaling the geometries to pixel coordinates aligned with the Raster
        if gdf.crs:
            # If the labels have a CRS, their geometries are in CRS coordinates,
            # so we need to apply the inverse of the Raster transform to get pixel coordinates.
            # This also applies the scaling_factor as the Raster is supposedly already scaled too.
            inverse_transform = ~self.metadata['transform']

            gdf['geometry'] = gdf['geometry'].astype(object).apply(
                lambda geom: apply_affine_transform(geom, inverse_transform)
            )
            gdf.crs = None
        elif self.scale_factor:
            # If the labels don't have a CRS, we expect them to already be in pixel coordinates.
            # So we just need to apply the scaling factor.
            gdf['geometry'] = gdf['geometry'].astype(object).apply(
                lambda geom: Polygon(
                    [(x * self.scale_factor, y * self.scale_factor) for x, y in geom.exterior.coords])
            )
        elif self.ground_resolution:
            gdf['geometry'] = gdf['geometry'].astype(object).apply(
                lambda geom: Polygon(
                    [(x * self.x_scale_factor, y * self.y_scale_factor) for x, y in geom.exterior.coords])
            )

        return gdf


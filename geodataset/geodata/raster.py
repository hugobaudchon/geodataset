import concurrent
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely import box
from shapely.affinity import translate
from shapely.geometry import Polygon

from geodataset.utils import read_raster, apply_affine_transform, validate_and_convert_product_name, \
    strip_all_extensions_and_path, TileNameConvention, PolygonTileNameConvention


class Raster:
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
        self.name = self.path.name
        self.ext = self.path.suffix
        self.product_name = validate_and_convert_product_name(strip_all_extensions_and_path(self.path))
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

    def create_tile_metadata(self, window: Window, tile_id: int) -> "RasterTileMetadata" or None:

        """
        Generates and returns a Tile from a window of the Raster.

        Parameters
        ----------
        window: Window
            The window to generate the Tile from.

        tile_id: int
            The id of the tile.

        Returns
        -------
        RasterTileMetadata or None
        """

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

        tile = RasterTileMetadata(
            associated_raster=self, mask=None, metadata=tile_metadata,
            ground_resolution=self.ground_resolution, scale_factor=self.scale_factor,
            row=window.row_off, col=window.col_off, tile_id=tile_id
        )

        return tile

    def get_polygon_tile(self,
                         polygon: Polygon,
                         polygon_id: int,
                         polygon_aoi: str,
                         tile_size: int,
                         use_variable_tile_size: bool,
                         variable_tile_size_pixel_buffer: int) -> Tuple["RasterPolygonTileMetadata", Polygon]:

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
        polygon_aoi: str
            The Area of Interest (AOI) the polygon belongs to. Used in the tile name.
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
        Tuple[RasterPolygonTileMetadata, Polygon]
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
        # Use an empty Polygon as the default value
        if translated_polygon_intersection.geom_type != 'Polygon':
            # Get the Polygon with the largest area
            translated_polygon_intersection = max(translated_polygon_intersection.geoms, key=lambda x: x.area, default=Polygon())

        # Ensure the result has an exterior before accessing its coordinates
        if not translated_polygon_intersection.is_empty:
            contours = np.array(translated_polygon_intersection.exterior.coords).reshape((-1, 1, 2)).astype(np.int32)
        else:
            # Handle the case when the intersection is empty (e.g., set contours to an empty array)
            contours = np.array([])

        # Check if contours is not empty before calling cv2.fillPoly
        if contours.size > 0:
            cv2.fillPoly(binary_mask, [contours], 1)
        else:
            # Handle the case where there are no contours (e.g., skip filling the polygon)
            pass

        # Getting the pixels from the raster
        mask_bounds = mask_box.bounds
        mask_bounds = [int(x) for x in mask_bounds]

        # Handling bounds outside the data
        start_row = mask_bounds[1]
        end_row = mask_bounds[3]
        start_col = mask_bounds[0]
        end_col = mask_bounds[2]

        # Padding to tile_size if necessary
        pre_row_pad = max(0, -mask_bounds[1])
        post_row_pad = max(0, mask_bounds[3] - self.data.shape[1])
        pre_col_pad = max(0, -mask_bounds[0])
        post_col_pad = max(0, mask_bounds[2] - self.data.shape[2])

        if final_tile_size - (end_row - start_row + pre_row_pad + post_row_pad) == 1:
            post_row_pad += 1
        if final_tile_size - (end_col - start_col + pre_col_pad + post_col_pad) == 1:
            post_col_pad += 1

        # Padding the mask to the final tile size
        start_row = max(0, start_row - pre_row_pad)
        end_row = min(self.data.shape[1], end_row + post_row_pad)
        start_col = max(0, start_col - pre_col_pad)
        end_col = min(self.data.shape[2], end_col + post_col_pad)
        binary_mask = np.pad(binary_mask, ((pre_row_pad, post_row_pad), (pre_col_pad, post_col_pad)),
                             mode='constant', constant_values=0)

        # Creating the PolygonTile, with the appropriate metadata
        window = Window(
            col_off=start_col,
            row_off=start_row,
            width=end_col - start_col,
            height=end_row - start_row
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

        polygon_tile = RasterPolygonTileMetadata(
            associated_raster=self,
            mask=binary_mask,
            metadata=tile_metadata,
            output_name=self.output_name,
            ground_resolution=self.ground_resolution,
            scale_factor=self.scale_factor,
            row=window.row_off,
            col=window.col_off,
            polygon_id=polygon_id,
            aoi=polygon_aoi
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

    def revert_polygons_pixels_coordinates_to_crs(self, gdf: gpd.GeoDataFrame):
        """
        Reverts the geometries in a GeoDataFrame from pixel coordinates to the CRS of the Raster.

        Parameters
        ----------
        gdf: gpd.GeoDataFrame
            The GeoDataFrame containing the geometries to revert.

        Returns
        -------
        gpd.GeoDataFrame
        """

        reverted_gdf = gdf.copy()

        assert reverted_gdf.crs is None, ("The geometries should be in pixel coordinates to revert them to a CRS,"
                                 " and so the input CRS should be None.")

        reverted_gdf['geometry'] = reverted_gdf['geometry'].astype(object).apply(
            lambda geom: apply_affine_transform(geom, self.metadata['transform'])
        )

        reverted_gdf.crs = self.metadata['crs']

        return reverted_gdf


class RasterTileMetadata:
    """
    Class to represent a tile of a raster image. Generally generated by a :class:`~geodataset.geodata.raster.Raster`.
    Does not copy the tile data, only the metadata, in order to reduce memory consumption.

    Parameters
    ----------
    associated_raster: Raster
        The Raster object associated with the tile.
    mask: np.ndarray or None
        A binary mask to apply to the tile. If None, no mask will be applied.
    metadata: dict
        The metadata of the raster tile. Should be compatible with rasterio.open(...) function.
        Examples of such metadata keys are 'crs', 'transform', 'width', 'height'...
    ground_resolution : float, optional
        The ground resolution in meter per pixel desired when loading the raster.
        Only one of ground_resolution and scale_factor can be set at the same time.
        Used for naming the .tif file when saving the tile.
    scale_factor : float, optional
        Scale factor for rescaling the data (change pixel resolution).
        Only one of ground_resolution and scale_factor can be set at the same time.
        Used for naming the .tif file when saving the tile
    row: int
        The row index of the tile in the raster (the top left pixel row).
        Used for naming the .tif file when saving the tile.
    col: int
        The column index of the tile in the raster (the top left pixel row).
        Used for naming the .tif file when saving the tile.
    aoi: str, optional
        The Area of Interest (AOI) the tile belongs to.
        Used for naming the .tif file when saving the tile.
    tile_id: int, optional
        The id of the tile.
        Used for naming the .tif file when saving the tile.
    """
    def __init__(self,
                 associated_raster: Raster,
                 mask: np.ndarray or None,
                 metadata: dict,
                 ground_resolution: float,
                 scale_factor: float,
                 row: int,
                 col: int,
                 aoi: str = None,
                 tile_id: int = None):

        self.associated_raster = associated_raster
        self.mask = None
        self.metadata = metadata
        self.row = row
        self.col = col
        self.aoi = aoi
        self.tile_id = tile_id

        assert not (ground_resolution and scale_factor), ("Both a ground_resolution and a scale_factor were provided."
                                                          " Please only specify one.")
        self.scale_factor = scale_factor
        self.ground_resolution = ground_resolution

        if mask is not None:
            self.update_mask(mask)

    def get_pixel_data(self):
        window = Window(self.col, self.row, self.metadata['width'], self.metadata['height'])

        tile_data = self.associated_raster.data.read(window=window)

        pad_row = self.metadata['height'] - tile_data.shape[1]
        pad_col = self.metadata['width'] - tile_data.shape[2]

        if pad_row > 0 or pad_col > 0:
            # RasterTileMetadata is always padded at the bottom and right because when tiled,
            # the minimum possible row and col are 0, so the top and left are always within the Raster,
            # but the bottom and right could go outside the Raster bounds.
            padding = ((0, 0), (0, pad_row), (0, pad_col))
            tile_data = np.pad(tile_data, padding, mode='constant', constant_values=0)

        if self.mask is not None:
            tile_data = tile_data * self.mask

        return tile_data

    def update_mask(self, additional_mask: np.ndarray):
        """
        Update the mask of the tile.

        Parameters
        ----------
        additional_mask: np.ndarray
            The additional, new mask to apply to the tile. Will be combined with the existing mask with an OR operation.
        """

        additional_mask = additional_mask.astype(np.bool_)

        if self.mask is not None:
            # combining masks
            self.mask = self.mask | additional_mask
        else:
            self.mask = additional_mask

    @staticmethod
    def _load_tile(path: Path):
        ext = path.suffix
        if ext != '.tif':
            raise Exception(f'The tile extension should be \'.tif\'.')

        with rasterio.open(path) as src:
            data = src.read()
            metadata = src.profile

        product_name, ground_resolution, scale_factor, row, col, aoi = TileNameConvention.parse_name(path.name)

        return data, metadata, product_name, ground_resolution, scale_factor, row, col, aoi

    def save(self, output_folder: str or Path):
        """
        Save the tile as a .tif file in the output_folder.

        Parameters
        ----------
        output_folder: str or pathlib.Path
            The path to the output folder where the tile should be saved.
        """

        output_folder = Path(output_folder)
        assert output_folder.exists(), f"The output folder {output_folder} doesn't exist yet."

        tile_name = self.generate_name()

        with rasterio.open(
                output_folder / tile_name,
                'w',
                **self.metadata,
                # compress='zstd',  # Lossless compression
                # predictor=2,  # For integer data; use predictor=3 for floating point if needed
                # tiled=True  # Enables tiling, which can improve compression efficiency
        ) as tile_raster:
            tile_raster.write(self.get_pixel_data())

    def generate_name(self):
        """
        Generate the name of the tile based on its metadata.

        Returns
        -------
        str
            The name of the tile.
        """
        return TileNameConvention.create_name(product_name=self.associated_raster.output_name,
                                              ground_resolution=self.ground_resolution,
                                              scale_factor=self.scale_factor,
                                              row=self.row,
                                              col=self.col,
                                              aoi=self.aoi)

    def get_bbox(self):
        """
        Get the bounding box of the tile in its original Raster coordinates.

        Returns
        -------
        shapely.geometry.box
            The bounding box of the tile.
        """
        minx = self.col
        maxx = self.col + self.metadata['width']
        miny = self.row
        maxy = self.row + self.metadata['height']
        return box(minx, miny, maxx, maxy)

    def copy_with_aoi_and_id(self, new_aoi: str, new_id: int):
        """
        Create a copy of the tile with a new AOI.

        Parameters
        ----------
        new_aoi: str
            The new AOI to assign to the tile.
        new_id: int
            The new ID to assign to the tile.

        Returns
        -------
        RasterTileMetadata
            The new tile with the new AOI.
        """

        return RasterTileMetadata(associated_raster=self.associated_raster,
                                  mask=self.mask,
                                  metadata=self.metadata,
                                  ground_resolution=self.ground_resolution,
                                  scale_factor=self.scale_factor,
                                  row=self.row,
                                  col=self.col,
                                  aoi=new_aoi,
                                  tile_id=new_id)


class RasterPolygonTileMetadata:
    """
    Class to represent a single polygon label tile of a raster image. Generally generated by a :class:`~geodataset.geodata.raster.Raster`.

    Parameters
    ----------
    associated_raster: Raster
        The Raster object associated with the tile.
    mask: np.ndarray or None
        A binary mask to apply to the tile. If None, no mask will be applied.
    metadata: dict
        The metadata of the raster image. Should be compatible with rasterio.open(...) function.
        Examples of such metadata keys are 'crs', 'transform', 'width', 'height'...
    output_name: str
        The name of the output product. Usually the same as the Raster name.
        Used as the prefix for the final .tif file name when saving the tile.
    ground_resolution : float, optional
        The ground resolution in meter per pixel desired when loading the raster.
        Only one of ground_resolution and scale_factor can be set at the same time.
        Used for naming the .tif file when saving the tile.
    scale_factor : float, optional
        Scale factor for rescaling the data (change pixel resolution).
        Only one of ground_resolution and scale_factor can be set at the same time.
        Used for naming the .tif file when saving the tile.
    row: int
        The row index of the tile in the raster (the top left pixel row).
        Used for naming the .tif file when saving the tile.
    col: int
        The column index of the tile in the raster (the top left pixel row).
        Used for naming the .tif file when saving the tile.
    polygon_id: int
        The unique identifier of the polygon.
    """
    def __init__(self,
                 associated_raster: Raster,
                 mask: np.ndarray or None,
                 metadata: dict,
                 output_name: str,
                 ground_resolution: float,
                 scale_factor: float,
                 row: int,
                 col: int,
                 aoi: str = None,
                 polygon_id: int = None):

        self.associated_raster = associated_raster
        self.mask = mask
        self.metadata = metadata
        self.output_name = output_name
        self.row = row
        self.col = col
        self.aoi = aoi
        self.polygon_id = polygon_id

        assert not (ground_resolution and scale_factor), ("Both a ground_resolution and a scale_factor were provided."
                                                          " Please only specify one.")
        self.scale_factor = scale_factor
        self.ground_resolution = ground_resolution

    def get_pixel_data(self):
        window = Window(self.col, self.row, self.metadata['width'], self.metadata['height'])
        
        tile_data = self.associated_raster.data.read(window=window)

        if self.mask is not None:
            tile_data = tile_data * self.mask

        return tile_data

    def update_mask(self, additional_mask: np.ndarray):
        """
        Update the mask of the tile.

        Parameters
        ----------
        additional_mask: np.ndarray
            The additional, new mask to apply to the tile. Will be combined with the existing mask with an OR operation.
        """
        if self.mask is not None:
            # combining masks
            self.mask = self.mask | additional_mask
        else:
            self.mask = additional_mask

    @staticmethod
    def _load_tile(path: Path):
        ext = path.suffix
        if ext != '.tif':
            raise Exception(f'The tile extension should be \'.tif\'.')

        with rasterio.open(path) as src:
            data = src.read()
            metadata = src.profile

        product_name, ground_resolution, scale_factor, polygon_id, aoi = PolygonTileNameConvention.parse_name(path.name)

        return data, metadata, product_name, ground_resolution, scale_factor, polygon_id, aoi

    def save(self, output_folder: Path):
        """
        Save the tile as a .tif file in the output_folder.
        """
        assert output_folder.exists(), f"The output folder {output_folder} doesn't exist yet."

        tile_name = self.generate_name()

        with rasterio.open(
                output_folder / tile_name,
                'w',
                **self.metadata,
                # compress='zstd',  # Lossless compression
                # predictor=2,  # For integer data; use predictor=3 for floating point if needed
                # tiled=True  # Enables tiling, which can improve compression efficiency
        ) as tile_raster:
            tile_raster.write(self.get_pixel_data())

    def generate_name(self):
        """
        Generate the name of the tile based on its metadata.

        Returns
        -------
        str
            The name of the tile.
        """
        return PolygonTileNameConvention.create_name(
            product_name=self.output_name,
            ground_resolution=self.ground_resolution,
            scale_factor=self.scale_factor,
            polygon_id=self.polygon_id,
            aoi=self.aoi
        )


class RasterTileSaver:
    """
    Class to save multiple tiles in parallel using ThreadPoolExecutor.

    Parameters
    ----------
    tiles_path: str or pathlib.Path
        The path to the folder where the tiles should be saved.
    n_workers: int
        The number of workers to use for saving the tiles.
    """
    def __init__(self, n_workers: int):
        self.n_workers = n_workers

    def save_tile(self, tile: RasterTileMetadata or RasterPolygonTileMetadata, output_folder: Path):
        """
        Save a single tile.

        Parameters
        ----------
        tile: RasterTile or RasterPolygonTileMetadata
            The tile to save.
        """
        try:
            tile.save(output_folder=output_folder)
        except Exception as e:
            print(f"Error saving tile {tile.generate_name()}: {str(e)}")

    def save_all_tiles(self, tiles: List[RasterTileMetadata or RasterPolygonTileMetadata], output_folder: Path):
        """
        Save all the tiles in parallel using ThreadPoolExecutor.

        Parameters
        ----------
        tiles: List[RasterTileMetadata or RasterPolygonTileMetadata]
            The list of tiles to save.
        """
        # Use ThreadPoolExecutor to manage threads
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit tasks to the executor
            futures = [executor.submit(self.save_tile, tile, output_folder) for tile in tiles]

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred: {e}")

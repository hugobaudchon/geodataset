import concurrent
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from math import floor
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
    strip_all_extensions_and_path, TileNameConvention, fix_geometry_collection, try_cast_multipolygon_to_polygon

_thread_ds = threading.local()


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
    temp_dir: str or pathlib.Path
        Temporary directory to store the resampled Raster, if it is too big to fit in memory.
    """
    def __init__(self,
                 path: str or Path,
                 output_name_suffix: str = None,
                 ground_resolution: float = None,
                 scale_factor: float = None,
                 temp_dir: str or Path = './tmp'):
        self.path = Path(path)
        self.name = self.path.name
        self.ext = self.path.suffix
        self.product_name = validate_and_convert_product_name(strip_all_extensions_and_path(self.path))
        self.output_name = self.product_name + (f"_{output_name_suffix}" if output_name_suffix else "")
        self.temp_dir = Path(temp_dir)

        assert not (ground_resolution and scale_factor), ("Both a ground_resolution and a scale_factor were provided."
                                                          " Please only specify one.")
        self.ground_resolution = ground_resolution
        self.scale_factor = scale_factor

        (self.data,
         self.metadata,
         self.x_scale_factor,
         self.y_scale_factor,
         self.temp_path) = self._load_data()

    def _load_data(self):
        data, metadata, x_scale_factor, y_scale_factor, temp_path = read_raster(
            path=self.path,
            ground_resolution=self.ground_resolution,
            scale_factor=self.scale_factor,
            temp_dir=self.temp_dir
        )

        if 'crs' not in metadata:
            metadata['crs'] = None
        if 'transform' not in metadata:
            metadata['transform'] = None

        if not metadata['crs']:
            warnings.warn(f'Could not find a CRS in the raster file {self.name}.')
        if not metadata['transform']:
            warnings.warn(f'Could not find a transform in the raster file {self.name}.')

        return data, metadata, x_scale_factor, y_scale_factor, temp_path

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
        Generates and returns a RasterTileMetadata from a Polygon geometry, along with the possibly cropped original Polygon.
        The RasterTileMetadata (object representing an image containing the pixels extent and metadata of the Raster where the Polygon is)
        will be centered on the Polygon's centroid.

        Parameters
        ----------
        polygon: Polygon
            The Polygon geometry to generate the RasterTileMetadata from.
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
            It should be combined with the variable_tile_size_pixel_buffer parameter to add a
            fixed pixel buffer around the Polygon.
            This parameter is recommended if saving disk space is a concern.
        variable_tile_size_pixel_buffer: int
            The number of pixels to add around the Polygon's extent as a buffer.
            Only used when use_variable_tile_size is set to 'True'.

        Returns
        -------
        Tuple[RasterPolygonTileMetadata, Polygon]
        """

        # find largest polygon and center tile around it
        polygon = fix_geometry_collection(polygon)
        polygon = try_cast_multipolygon_to_polygon(polygon, strategy='largest_part')
        x, y = polygon.centroid.coords[0]
        cx, cy = polygon.centroid.coords[0]

        if use_variable_tile_size:
            # Variable tile size centered on the polygon centroid, with a minimum size of tile_size and that doesn't go outside the raster bounds
            max_dist_to_polygon_border = max(
                [x - polygon.bounds[0], polygon.bounds[2] - cx, cy - polygon.bounds[1], polygon.bounds[3] - cy])
            final_tile_size = int(min(tile_size, max_dist_to_polygon_border * 2 + variable_tile_size_pixel_buffer * 2))
        else:
            # Fixed tile size centered on the polygon centroid
            final_tile_size = tile_size

        col_off = int(floor(cx - final_tile_size / 2))
        row_off = int(floor(cy - final_tile_size / 2))
        window = Window(col_off=col_off, row_off=row_off, width=final_tile_size, height=final_tile_size)
        binary_mask = np.zeros((final_tile_size, final_tile_size), dtype=np.uint8)

        window_transform = rasterio.windows.transform(window, self.metadata['transform'])
        tile_metadata = {
            'driver': 'GTiff',
            'height': final_tile_size,
            'width': final_tile_size,
            'count': self.metadata['count'],
            'dtype': self.metadata['dtype'],
            'crs': self.metadata['crs'],
            'transform': window_transform
        }

        # Ensure polygon validity
        polygon = polygon.buffer(0)

        # Clip to the window bounds (in pixel coords), then make sure there is still only one polygon
        win_bbox = box(col_off, row_off, col_off + final_tile_size, row_off + final_tile_size)
        inter = polygon.intersection(win_bbox)
        inter = fix_geometry_collection(inter)
        inter = try_cast_multipolygon_to_polygon(inter, strategy='largest_part')

        # Translate the polygon into the tile frame of reference
        translated_inter = translate(inter, xoff=-col_off, yoff=-row_off)

        # Ensure the result has an exterior before accessing its coordinates
        if not translated_inter.is_empty:
            contours = np.array(translated_inter.exterior.coords).reshape((-1, 1, 2)).astype(np.int32)
        else:
            # Handle the case when the intersection is empty (e.g., set contours to an empty array)
            contours = np.array([])

        # Check if contours is not empty before calling cv2.fillPoly
        if contours.size > 0:
            cv2.fillPoly(binary_mask, [contours], 1)
        else:
            # Handle the case where there are no contours (e.g., skip filling the polygon)
            pass

        # Creating the RasterTileMetadata, with the appropriate metadata
        polygon_tile = RasterTileMetadata(
            associated_raster=self,
            mask=binary_mask,
            metadata=tile_metadata,
            ground_resolution=self.ground_resolution,
            scale_factor=self.scale_factor,
            row=window.row_off,
            col=window.col_off,
            tile_id=polygon_id,
            aoi=polygon_aoi
        )

        return polygon_tile, translated_inter

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

    def get_pixel_data(self, apply_mask: bool = True):
        """
        Get the pixel data of the tile.

        Parameters
        ----------
        apply_mask: bool
            Whether to apply the mask to the tile data. Default is True.

        Returns
        -------
        np.ndarray
        """
        window = Window(self.col, self.row, self.metadata['width'], self.metadata['height'])

        # Using a thread-local storage to avoid opening the raster file multiple times in parallel threads
        if not hasattr(_thread_ds, "ds"):
            ds_name = self.associated_raster.data.name
            _thread_ds.ds = rasterio.open(ds_name)
        ds = _thread_ds.ds

        tile_data = ds.read(window=window, boundless=True, masked=False, fill_value=0)
        # tile_data = self.associated_raster.data.read(window=window)

        if self.mask is not None and apply_mask:
            tile_data = tile_data * self.mask

        return tile_data

    def update_mask(self, additional_mask: np.ndarray):
        """
        Update the mask of the tile.

        Parameters
        ----------
        additional_mask: np.ndarray
            The additional, new mask to apply to the tile. Will be combined with the existing mask with an AND operation.
        """

        additional_mask = additional_mask.astype(np.bool_)

        if self.mask is not None:
            # combining masks
            self.mask = self.mask & additional_mask
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

    def save(self,
             output_folder: str or Path,
             apply_mask: bool = True):
        """
        Save the tile as a .tif file in the output_folder.

        Parameters
        ----------
        output_folder: str or pathlib.Path
            The path to the output folder where the tile should be saved.
        apply_mask: bool
            Whether to apply the mask to the tile data before saving it. Default is True.
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
            tile_raster.write(self.get_pixel_data(apply_mask=apply_mask))

            # Make sure to also copy the colorinterp from the parent raster
            parent_ci = self.associated_raster.data.colorinterp
            tile_raster.colorinterp = parent_ci

    def generate_name(self):
        """
        Generate the name of the tile based on its metadata.

        Returns
        -------
        str
            The name of the tile.
        """
        return TileNameConvention.create_name(
            product_name=self.associated_raster.output_name,
            ground_resolution=self.ground_resolution,
            scale_factor=self.scale_factor,
            row=self.row,
            col=self.col,
            aoi=self.aoi,
            width=self.metadata['width'],
            height=self.metadata['height'],
            tile_id=self.tile_id
        )

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

        return RasterTileMetadata(
            associated_raster=self.associated_raster,
            mask=self.mask,
            metadata=self.metadata,
            ground_resolution=self.ground_resolution,
            scale_factor=self.scale_factor,
            row=self.row,
            col=self.col,
            aoi=new_aoi,
            tile_id=new_id
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

    def save_tile(self,
                  tile: RasterTileMetadata,
                  output_folder: Path,
                  apply_mask: bool = True):
        """
        Save a single tile.

        Parameters
        ----------
        tile: RasterTile or RasterPolygonTileMetadata
            The tile to save.
        output_folder: Path
            The path to the folder where the tile should be saved.
        apply_mask: bool
            Whether to apply the mask to the tile data before saving it. Default is True.
        """
        try:
            tile.save(output_folder=output_folder, apply_mask=apply_mask)
        except Exception as e:
            print(f"Error saving tile {tile.generate_name()}: {str(e)}")

    def save_all_tiles(self,
                       tiles: List[RasterTileMetadata],
                       output_folder: Path,
                       apply_mask: bool = True):
        """
        Save all the tiles in parallel using ThreadPoolExecutor.

        Parameters
        ----------
        tiles: List[RasterTileMetadata or RasterPolygonTileMetadata]
            The list of tiles to save.
        output_folder: Path
            The path to the folder where the tiles should be saved.
        apply_mask: bool
            Whether to apply the mask to the tile data before saving it. Default is True.
        """
        # Use ThreadPoolExecutor to manage threads
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit tasks to the executor
            futures = [executor.submit(self.save_tile, tile, output_folder, apply_mask) for tile in tiles]

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred: {e}")

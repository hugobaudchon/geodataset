from abc import ABC
from typing import List
import numpy as np
from pathlib import Path
from typing import cast

from geopandas import GeoDataFrame
from rasterio.windows import Window
from tqdm import tqdm

from geodataset.aoi import AOIGeneratorForTiles, AOIFromPackageForTiles
from geodataset.aoi import AOIConfig, AOIGeneratorConfig, AOIFromPackageConfig
from geodataset.aoi.aoi_base import DEFAULT_AOI_NAME
from geodataset.geodata import Raster, RasterTileMetadata
from geodataset.utils import save_aois_tiles_picture, AoiTilesImageConvention, assert_raster_exists
from geodataset.utils.file_name_conventions import AoiGeoPackageConvention


class BaseRasterTilerizer(ABC):
    """
    Base class for raster tilerizers.

    Parameters
    ----------
    raster_path : str or pathlib.Path
        Path to the raster (.tif, .png...).
    tile_size : int
        The size of the tiles in pixels (tile_size, tile_size).
    tile_overlap : float
        The overlap between the tiles (0 <= overlap < 1).
    global_aoi : str or pathlib.Path or geopandas.GeoDataFrame, optional
        Path to the global AOI file, or directly a GeoDataFrame.
        If provided, only the tiles intersecting this AOI will be kept, even if some tiles are inside one of the aois
        in aois_config (if AOIFromPackageConfig).

        This parameter can be really useful to create a kfold dataset in association with an AOIGeneratorConfig config like this:

        aois_config = AOIGeneratorConfig(aois={
                'zone1': {'percentage': 0.2, 'position': 1, 'actual_name': f'train{kfold_id}'},
                'zone2': {'percentage': 0.2, 'position': 2, 'actual_name': f'train{kfold_id}'},
                'zone3': {'percentage': 0.2, 'position': 3, 'actual_name': f'valid{kfold_id}'},
                'zone4': {'percentage': 0.2, 'position': 4, 'actual_name': f'train{kfold_id}'},
                'zone5': {'percentage': 0.2, 'position': 5, 'actual_name': f'train{kfold_id}'}
            },
            aoi_type='band'
        )
    aois_config : :class:`~geodataset.aoi.AOIGeneratorConfig` or :class:`~geodataset.aoi.AOIFromPackageConfig` or None
        An instance of AOIConfig to use, or None if all tiles should be kept in a DEFAULT_AOI_NAME AOI.
    ground_resolution : float, optional
        The ground resolution in meter per pixel desired when loading the raster.
        Only one of ground_resolution and scale_factor can be set at the same time.
    scale_factor : float, optional
        Scale factor for rescaling the data (change pixel resolution).
        Only one of ground_resolution and scale_factor can be set at the same time.
    output_name_suffix : str, optional
        Suffix to add to the output file names.
    ignore_black_white_alpha_tiles_threshold : float, optional
        Threshold ratio of black or white pixels in a tile to skip it.
    temp_dir : str or pathlib.Path
        Temporary directory to store the resampled Raster, if it is too big to fit in memory.
    """

    def __init__(self,
                 raster_path: str or Path,
                 tile_size: int,
                 tile_overlap: float,
                 global_aoi: str or Path or GeoDataFrame = None,
                 aois_config: AOIConfig = None,
                 ground_resolution: float = None,
                 scale_factor: float = None,
                 output_name_suffix: str = None,
                 ignore_black_white_alpha_tiles_threshold: float = 0.8,
                 temp_dir: str or Path = './tmp'):

        self.raster_path = str(raster_path)
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.tile_coordinate_step = int((1 - self.tile_overlap) * self.tile_size)
        self.buffer_overlap_rows = self.tile_size - self.tile_coordinate_step

        self.global_aoi = global_aoi
        self.aois_config = aois_config
        self.output_name_suffix = output_name_suffix
        self.ignore_black_white_alpha_tiles_threshold = ignore_black_white_alpha_tiles_threshold
        self.temp_dir = Path(temp_dir)

        self.scale_factor = scale_factor
        self.ground_resolution = ground_resolution

        self._check_parameters()
        self._check_aois_config()

        self.raster = self._load_raster()

    def _check_parameters(self):
        assert assert_raster_exists(self.raster_path), \
            f"Raster file not found at {self.raster_path}."
        assert isinstance(self.tile_size, int) and self.tile_size > 0, \
            "The tile size must be and integer greater than 0."
        assert isinstance(self.tile_overlap, float) and 0 <= self.tile_overlap < 1, \
            "The tile overlap must be between 0 and 1."
        assert not (self.ground_resolution and self.scale_factor), \
            "Both a ground_resolution and a scale_factor were provided. Please only specify one."

    def _check_aois_config(self):
        if self.aois_config is None:
            self.aois_config = AOIGeneratorConfig(aois={DEFAULT_AOI_NAME: {'percentage': 1, 'position': 1}}, aoi_type='band')
            print('No AOIs provided, all tiles will be kept in the DEFAULT_AOI_NAME AOI.')

    def _load_raster(self):
        raster = Raster(path=self.raster_path,
                        output_name_suffix=self.output_name_suffix,
                        ground_resolution=self.ground_resolution,
                        scale_factor=self.scale_factor,
                        temp_dir=self.temp_dir)
        return raster

    def _compute_skip_condition(self, data: np.array) -> np.array:
        """
        Runs the skip logic on an entire raster array and returns a 2D boolean mask.
        True means the pixel *is* a skip pixel (black, white, or alpha).
        """
        # (Handling 1-band case where read() might return 2D)
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)

        num_bands = data.shape[0]

        if num_bands == 3:
            black = np.all(data == 0, axis=0)
            white = np.all(data == 255, axis=0)
            return (black | white)
        elif num_bands == 4:
            black = np.all(data[:-1] == 0, axis=0)
            white = np.all(data[:-1] == 255, axis=0)
            alpha = data[-1] == 0
            return (black | white | alpha)
        else:
            # Handle 1-band or other cases
            black = (data[0] == 0)
            white = (data[0] == 255)  # Assuming 255 is also "empty"
            return (black | white)

    def _create_tiles(self) -> List[RasterTileMetadata]:
        if not self.raster.data.name.startswith('/vsimem/'):
            print("Warning: Rolling buffer optimization is fastest when raster is in-memory.")

        width = self.raster.metadata['width']
        height = self.raster.metadata['height']
        num_bands = self.raster.data.count

        print('Raster size: ', (height, width))
        print('Desired tile size: ', self.tile_size)
        print('Creating tiles and finding their associated labels...')

        tile_id_counter = 0
        tiles = []

        # 1. Check if we even need to run this logic
        if self.ignore_black_white_alpha_tiles_threshold >= 1.0:
            print("Skip threshold is >= 1.0, creating all tiles without checking.")
            for row in tqdm(range(0, height, self.tile_coordinate_step), desc="Processing rows"):
                for col in range(0, width, self.tile_coordinate_step):
                    window = Window(col, row, width=self.tile_size, height=self.tile_size)
                    tile = self.raster.create_tile_metadata(window=window, tile_id=tile_id_counter)
                    tiles.append(tile)
                    tile_id_counter += 1
            return tiles

        # 2. Initialize the rolling buffers
        data_buffer = np.zeros((num_bands, self.tile_size, width), dtype=self.raster.metadata['dtype'])
        mask_buffer = np.zeros((self.tile_size, width), dtype=bool)

        # 3. Pre-fill the buffer with the first strip (0 to tile_size)
        rows_to_read = min(self.tile_size, height)
        new_data_window = Window(0, 0, width, rows_to_read)
        new_data = self.raster.data.read(window=new_data_window)

        # Place data in the buffer
        data_buffer[:, 0:rows_to_read, :] = new_data

        # Compute and place mask
        new_mask = self._compute_skip_condition(new_data)
        mask_buffer[0:rows_to_read, :] = new_mask

        rows_read_so_far = rows_to_read

        # 4. --- Process the first strip (row = 0) ---
        for col in range(0, width, self.tile_coordinate_step):
            col_end = col + self.tile_size
            tile_mask_slice = mask_buffer[:, col:col_end]

            tile_size_h, tile_size_w = tile_mask_slice.shape
            if tile_size_h == 0 or tile_size_w == 0: continue

            num_pixels = tile_size_h * tile_size_w
            if (np.sum(tile_mask_slice) / num_pixels) >= self.ignore_black_white_alpha_tiles_threshold:
                continue

            window = Window(col, 0, width=self.tile_size, height=self.tile_size)
            tile = self.raster.create_tile_metadata(window=window, tile_id=tile_id_counter)
            tiles.append(tile)
            tile_id_counter += 1

        # 5. --- Main loop for all subsequent rows ---
        for row in tqdm(range(self.tile_coordinate_step, height, self.tile_coordinate_step), desc="Processing rows"):

            rows_to_read = min(self.tile_coordinate_step, height - rows_read_so_far)

            if rows_to_read <= 0:
                break  # We've read the whole raster

            # Roll the buffers
            if self.buffer_overlap_rows > 0:
                # --- CORRECTED SLICE on axis 1 ---
                data_buffer[:, 0:self.buffer_overlap_rows, :] = data_buffer[:, self.tile_coordinate_step:, :]
                mask_buffer[0:self.buffer_overlap_rows, :] = mask_buffer[self.tile_coordinate_step:, :]

            # Read new data
            new_data_window = Window(0, rows_read_so_far, width, rows_to_read)
            new_data = self.raster.data.read(window=new_data_window)

            # Place new data in the end of the buffer
            # --- CORRECTED SLICE on axis 1 ---
            data_buffer[:, self.buffer_overlap_rows: self.buffer_overlap_rows + rows_to_read, :] = new_data

            # Compute and place new mask
            new_mask = self._compute_skip_condition(new_data)
            mask_buffer[self.buffer_overlap_rows: self.buffer_overlap_rows + rows_to_read, :] = new_mask

            rows_read_so_far += rows_to_read

            # --- Column Loop (now uses the rolled buffer) ---
            for col in range(0, width, self.tile_coordinate_step):

                col_end = col + self.tile_size
                tile_mask_slice = mask_buffer[:, col:col_end]

                tile_size_h, tile_size_w = tile_mask_slice.shape
                if tile_size_h == 0 or tile_size_w == 0:
                    continue

                num_pixels = tile_size_h * tile_size_w
                if (np.sum(tile_mask_slice) / num_pixels) >= self.ignore_black_white_alpha_tiles_threshold:
                    continue

                window = Window(col, row, width=self.tile_size, height=self.tile_size)
                tile = self.raster.create_tile_metadata(window=window, tile_id=tile_id_counter)
                tiles.append(tile)
                tile_id_counter += 1

        return tiles

    def _get_tiles_per_aoi(self, tiles: List[RasterTileMetadata]):
        print('Assigning the tiles to the aois...')

        if self.aois_config is not None:
            if type(self.aois_config) is AOIGeneratorConfig:
                aoi_engine = AOIGeneratorForTiles(
                    tiles=tiles,
                    tile_coordinate_step=self.tile_coordinate_step,
                    associated_raster=self.raster,
                    global_aoi=self.global_aoi,
                    aois_config=cast(AOIGeneratorConfig, self.aois_config),
                    ignore_black_white_alpha_tiles_threshold=self.ignore_black_white_alpha_tiles_threshold
                )
            elif type(self.aois_config) is AOIFromPackageConfig:
                aoi_engine = AOIFromPackageForTiles(
                    tiles=tiles,
                    tile_coordinate_step=self.tile_coordinate_step,
                    associated_raster=self.raster,
                    global_aoi=self.global_aoi,
                    aois_config=cast(AOIFromPackageConfig, self.aois_config),
                    ground_resolution=self.ground_resolution,
                    scale_factor=self.scale_factor
                )
            else:
                raise Exception(f'aois_config type unsupported: {type(self.aois_config)}')

            aois_tiles, aois_gdf = aoi_engine.get_aoi_tiles()

            # When the tiles are assigned to AOIs, sometimes part of them are black-out
            # (pixels within the tile that are outside the assigned AOI),
            # so we need to check again if the tile has too many black pixels, and remove it if yes.
            final_aois_tiles = {aoi: [] for aoi in aois_tiles}
            for aoi in aois_tiles:
                for tile in aois_tiles[aoi]:
                    if tile.mask is not None and self._check_skip_tile(tile=tile):
                        continue
                    else:
                        final_aois_tiles[aoi].append(tile)
        else:
            final_aois_tiles = {}
            aois_gdf = None

        return final_aois_tiles, aois_gdf

    def _check_skip_tile_data(self, tile_data: np.array) -> bool:
        """
        Checks a tile's NumPy array data to see if it should be skipped.
        This is now only used by _get_tiles_per_aoi (which is fine).
        """
        skip_ratio = self.ignore_black_white_alpha_tiles_threshold

        # Get actual tile dimensions from the array
        if len(tile_data.shape) == 2:  # Handle 1-band case
            tile_data = np.expand_dims(tile_data, axis=0)

        num_bands, tile_size_h, tile_size_w = tile_data.shape

        # Skip if the tile is effectively empty (e.g., a sliver at the edge)
        if tile_size_h == 0 or tile_size_w == 0:
            return True

        num_pixels = tile_size_h * tile_size_w

        if num_bands == 3:
            black = np.all(tile_data == 0, axis=0)
            white = np.all(tile_data == 255, axis=0)
            if np.sum(black | white) / num_pixels >= skip_ratio:
                return True
        elif num_bands == 4:
            black = np.all(tile_data[:-1] == 0, axis=0)
            white = np.all(tile_data[:-1] == 255, axis=0)
            alpha = tile_data[-1] == 0
            if np.sum(black | white | alpha) / num_pixels >= skip_ratio:
                return True

        # Handle 1-band or other cases
        elif num_bands == 1:
            black = (tile_data[0] == 0)
            white = (tile_data[0] == 255)  # Assuming 255 is also "empty"
            if np.sum(black | white) / num_pixels >= skip_ratio:
                return True

        return False

    def _check_skip_tile(self, tile: RasterTileMetadata):
        # This function now performs the I/O...
        tile_data = tile.get_pixel_data()

        return self._check_skip_tile_data(tile_data)


class BaseDiskRasterTilerizer(BaseRasterTilerizer, ABC):
    """
    A base class for tilerizers that save tiles to the disk.

    Parameters
    ---------------------
    raster_path : str or pathlib.Path
        Path to the raster (.tif, .png...).
    output_path : str or pathlib.Path
        Path to parent folder where to save the image tiles and associated labels.
    tile_size : int
        The wanted size of the tiles (tile_size, tile_size).
    tile_overlap : float
        The overlap between the tiles (should be 0 <= overlap < 1).
    global_aoi : str or pathlib.Path or geopandas.GeoDataFrame, optional
        Path to the global AOI file, or directly a GeoDataFrame.
        If provided, only the tiles intersecting this AOI will be kept, even if some tiles are inside one of the aois
        in aois_config (if AOIFromPackageConfig).

        This parameter can be really useful to create a kfold dataset in association with an AOIGeneratorConfig config like this:

        aois_config = AOIGeneratorConfig(aois={
                'zone1': {'percentage': 0.2, 'position': 1, 'actual_name': f'train{kfold_id}'},
                'zone2': {'percentage': 0.2, 'position': 2, 'actual_name': f'train{kfold_id}'},
                'zone3': {'percentage': 0.2, 'position': 3, 'actual_name': f'valid{kfold_id}'},
                'zone4': {'percentage': 0.2, 'position': 4, 'actual_name': f'train{kfold_id}'},
                'zone5': {'percentage': 0.2, 'position': 5, 'actual_name': f'train{kfold_id}'}
            },
            aoi_type='band'
        )
    aois_config : :class:`~geodataset.aoi.AOIGeneratorConfig` or :class:`~geodataset.aoi.AOIFromPackageConfig` or None
        An instance of AOIConfig to use, or None if all tiles should be kept in a DEFAULT_AOI_NAME AOI.
    ground_resolution : float
        The ground resolution in meter per pixel desired when loading the raster.
        Only one of ground_resolution and scale_factor can be set at the same time.
    scale_factor : float
        Scale factor for rescaling the data (change pixel resolution).
        Only one of ground_resolution and scale_factor can be set at the same time.
    output_name_suffix : str
        Suffix to add to the output file names.
    ignore_black_white_alpha_tiles_threshold : float
        Threshold ratio of black, white or transparent pixels in a tile to skip it
    temp_dir : str or pathlib.Path
        Temporary directory to store the resampled Raster, if it is too big to fit in memory.
    output_dtype : str
        The data type to use when saving the tile. If None, the original data type will
        be used. Currently supported values are None and 'uint8' (0-255).
    """

    def __init__(self,
                 raster_path: str,
                 output_path: str,
                 tile_size: int,
                 tile_overlap: float,
                 global_aoi: str or Path or GeoDataFrame = None,
                 aois_config: AOIConfig = None,
                 ground_resolution: float = None,
                 scale_factor: float = None,
                 output_name_suffix: str = None,
                 ignore_black_white_alpha_tiles_threshold: float = 0.8,
                 temp_dir: str or Path = './tmp',
                 output_dtype: str = None):

        super().__init__(raster_path=raster_path,
                         tile_size=tile_size,
                         tile_overlap=tile_overlap,
                         global_aoi=global_aoi,
                         aois_config=aois_config,
                         ground_resolution=ground_resolution,
                         scale_factor=scale_factor,
                         output_name_suffix=output_name_suffix,
                         ignore_black_white_alpha_tiles_threshold=ignore_black_white_alpha_tiles_threshold,
                         temp_dir=temp_dir)

        self.output_path = Path(output_path) / self.raster.output_name
        self.tiles_path = self.output_path / 'tiles'
        self.tiles_path.mkdir(parents=True, exist_ok=True)
        self.output_dtype = output_dtype
        self.aois_tiles = None
        self.aois_gdf = None

    def _get_tiles_per_aoi(self, tiles: List[RasterTileMetadata]):
        aois_tiles, aois_gdf = super()._get_tiles_per_aoi(tiles=tiles)

        # Saving the AOIs to disk
        for aoi in aois_gdf['aoi'].unique():
            if aoi in aois_tiles and len(aois_tiles[aoi]) > 0:
                aoi_gdf = aois_gdf[aois_gdf['aoi'] == aoi]
                aoi_gdf = self.raster.revert_polygons_pixels_coordinates_to_crs(aoi_gdf)
                aoi_file_name = AoiGeoPackageConvention.create_name(
                    product_name=self.raster.output_name,
                    aoi=aoi,
                    ground_resolution=self.ground_resolution,
                    scale_factor=self.scale_factor
                )
                aoi_gdf.drop(columns=['id'], inplace=True, errors='ignore')
                aoi_gdf.to_file(self.output_path / aoi_file_name, driver='GPKG')
                print(f"Final AOI '{aoi}' saved to {self.output_path / aoi_file_name}.")

        self.aois_tiles = aois_tiles
        self.aois_gdf = aois_gdf


class RasterTilerizer(BaseDiskRasterTilerizer):
    """
    A standard tilerizer for Raster data without annotations or labels. The generate_tiles method generates and then saves the tiles to the disk.

    Parameters
    ---------------------
    raster_path : str or pathlib.Path
        Path to the raster (.tif, .png...).
    output_path : str or pathlib.Path
        Path to parent folder where to save the image tiles.
    tile_size : int
        The wanted size of the tiles (tile_size, tile_size).
    tile_overlap : float
        The overlap between the tiles (should be 0 <= overlap < 1).
    global_aoi : str or pathlib.Path or geopandas.GeoDataFrame, optional
        Path to the global AOI file, or directly a GeoDataFrame.
        If provided, only the tiles intersecting this AOI will be kept, even if some tiles are inside one of the aois
        in aois_config (if AOIFromPackageConfig).

        This parameter can be really useful to create a kfold dataset in association with an AOIGeneratorConfig config like this:

        aois_config = AOIGeneratorConfig(aois={
                'zone1': {'percentage': 0.2, 'position': 1, 'actual_name': f'train{kfold_id}'},
                'zone2': {'percentage': 0.2, 'position': 2, 'actual_name': f'train{kfold_id}'},
                'zone3': {'percentage': 0.2, 'position': 3, 'actual_name': f'valid{kfold_id}'},
                'zone4': {'percentage': 0.2, 'position': 4, 'actual_name': f'train{kfold_id}'},
                'zone5': {'percentage': 0.2, 'position': 5, 'actual_name': f'train{kfold_id}'}
            },
            aoi_type='band'
        )
    aois_config : :class:`~geodataset.aoi.AOIGeneratorConfig` or :class:`~geodataset.aoi.AOIFromPackageConfig` or None
        An instance of AOIConfig to use, or None if all tiles should be kept in a DEFAULT_AOI_NAME AOI.
    ground_resolution : float
        The ground resolution in meter per pixel desired when loading the raster.
        Only one of ground_resolution and scale_factor can be set at the same time.
    scale_factor : float
        Scale factor for rescaling the data (change pixel resolution).
        Only one of ground_resolution and scale_factor can be set at the same time.
    output_name_suffix : str
        Suffix to add to the output file names.
    ignore_black_white_alpha_tiles_threshold : float
        Threshold ratio of black, white or transparent pixels in a tile to skip it. Default is 0.8.
    temp_dir : str or pathlib.Path
        Temporary directory to store the resampled Raster, if it is too big to fit in memory.
    output_dtype : str
        The data type to use when saving the tile. If None, the original data type will
        be used. Currently supported values are None and 'uint8' (0-255).
    """

    def __init__(self,
                 raster_path: str or Path,
                 output_path: str or Path,
                 tile_size: int,
                 tile_overlap: float,
                 global_aoi: str or Path or GeoDataFrame = None,
                 aois_config: AOIConfig = None,
                 ground_resolution: float = None,
                 scale_factor: float = None,
                 output_name_suffix: str = None,
                 ignore_black_white_alpha_tiles_threshold: float = 0.8,
                 temp_dir: str or Path = './tmp',
                 output_dtype: str = None):

        super().__init__(raster_path=raster_path,
                         output_path=output_path,
                         tile_size=tile_size,
                         tile_overlap=tile_overlap,
                         global_aoi=global_aoi,
                         aois_config=aois_config,
                         ground_resolution=ground_resolution,
                         scale_factor=scale_factor,
                         output_name_suffix=output_name_suffix,
                         ignore_black_white_alpha_tiles_threshold=ignore_black_white_alpha_tiles_threshold,
                         temp_dir=temp_dir,
                         output_dtype=output_dtype)

    def generate_tiles(self):
        """
        Generate the tiles and save them to the disk.
        """

        tiles = self._create_tiles()
        self._get_tiles_per_aoi(tiles=tiles)

        save_aois_tiles_picture(aois_tiles=self.aois_tiles,
                                save_path=self.output_path / AoiTilesImageConvention.create_name(
                                    product_name=self.raster.output_name,
                                    ground_resolution=self.ground_resolution,
                                    scale_factor=self.scale_factor
                                ),
                                tile_coordinate_step=self.tile_coordinate_step)

        [print(f'No tiles found for AOI {aoi}.') for aoi in self.aois_config.actual_names
         if aoi not in self.aois_tiles or len(self.aois_tiles[aoi]) == 0]

        for aoi in self.aois_tiles:
            # Save the tile images
            tiles_path_aoi = self.tiles_path / aoi
            tiles_path_aoi.mkdir(parents=True, exist_ok=True)

            for tile in tqdm(self.aois_tiles[aoi], desc=f"Saving tiles for AOI '{aoi}'"):
                tile.save(output_folder=tiles_path_aoi, output_dtype=self.output_dtype)

        print(f"The tiles have been saved to {self.tiles_path}.")


class RasterTilerizerGDF(BaseRasterTilerizer):
    """
    A standard tilerizer for Raster data without annotations or labels. The generate_tiles_gdf method returns tiles extents as a GeoDataFrame.

    Parameters
    ----------
    raster_path : str or pathlib.Path
        Path to the raster (.tif, .png...).
    tile_size : int
        The size of the tiles in pixels (tile_size, tile_size).
    tile_overlap : float
        The overlap between the tiles (0 <= overlap < 1).
    aois_config : :class:`~geodataset.aoi.AOIGeneratorConfig` or :class:`~geodataset.aoi.AOIFromPackageConfig` or None
        An instance of AOIConfig to use, or None if all tiles should be kept in an DEFAULT_AOI_NAME AOI.
    ground_resolution : float, optional
        The ground resolution in meter per pixel desired when loading the raster.
        Only one of ground_resolution and scale_factor can be set at the same time.
    scale_factor : float, optional
        Scale factor for rescaling the data (change pixel resolution).
        Only one of ground_resolution and scale_factor can be set at the same time.
    output_name_suffix : str, optional
        Suffix to add to the output file names.
    ignore_black_white_alpha_tiles_threshold : float, optional
        Threshold ratio of black, white or transparent pixels in a tile to skip it. Default is 0.8.
    temp_dir : str or pathlib.Path
        Temporary directory to store the resampled Raster, if it is too big to fit in memory.
    """
    def __init__(self,
                 raster_path: str or Path,
                 tile_size: int,
                 tile_overlap: float,
                 aois_config: AOIConfig = None,
                 ground_resolution: float = None,
                 scale_factor: float = None,
                 output_name_suffix: str = None,
                 ignore_black_white_alpha_tiles_threshold: float = 0.8,
                 temp_dir: str or Path = './tmp'):

        super().__init__(raster_path=raster_path,
                         tile_size=tile_size,
                         tile_overlap=tile_overlap,
                         aois_config=aois_config,
                         ground_resolution=ground_resolution,
                         scale_factor=scale_factor,
                         output_name_suffix=output_name_suffix,
                         ignore_black_white_alpha_tiles_threshold=ignore_black_white_alpha_tiles_threshold,
                         temp_dir=temp_dir)

    def generate_tiles_gdf(self):
        """
        Generate the tiles and return them as a GeoDataFrame.
        """

        tiles = self._create_tiles()
        aois_tiles, _ = self._get_tiles_per_aoi(tiles=tiles)

        tiles_gdf = GeoDataFrame(
            {
                'geometry': [tile.get_bbox() for aoi in aois_tiles for tile in aois_tiles[aoi]],
                'id': [tile.tile_id for aoi in aois_tiles for tile in aois_tiles[aoi]],
                'col': [tile.col for aoi in aois_tiles for tile in aois_tiles[aoi]],
                'row': [tile.row for aoi in aois_tiles for tile in aois_tiles[aoi]],
                'aoi': [aoi for aoi in aois_tiles for _ in aois_tiles[aoi]]
            },
            crs=None
        )

        return tiles_gdf



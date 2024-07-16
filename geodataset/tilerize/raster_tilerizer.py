from abc import ABC
from typing import List
import numpy as np
import rasterio
from pathlib import Path
from typing import cast

from geopandas import GeoDataFrame
from tqdm import tqdm

from geodataset.aoi import AOIGeneratorForTiles, AOIFromPackageForTiles
from geodataset.aoi import AOIConfig, AOIGeneratorConfig, AOIFromPackageConfig
from geodataset.geodata import Raster, Tile
from geodataset.utils import save_aois_tiles_picture, AoiTilesImageConvention


class BaseRasterTilerizer(ABC):
    """
    Base class for raster tilerizers.

    Parameters
    ----------
    raster_path : str
        Path to the raster (.tif, .png...).
    tile_size : int
        The size of the tiles in pixels (tile_size, tile_size).
    tile_overlap : float
        The overlap between the tiles (0 <= overlap < 1).
    aois_config : AOIConfig or None, optional
        An instance of AOIConfig to use, or None if all tiles should be kept in an 'all' AOI.
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
    """

    def __init__(self,
                 raster_path: str,
                 tile_size: int,
                 tile_overlap: float,
                 aois_config: AOIConfig = None,
                 ground_resolution: float = None,
                 scale_factor: float = None,
                 output_name_suffix: str = None,
                 ignore_black_white_alpha_tiles_threshold: float = 0.8):

        self.raster_path = Path(raster_path)
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.tile_coordinate_step = int((1 - self.tile_overlap) * self.tile_size)
        self.aois_config = aois_config
        self.output_name_suffix = output_name_suffix
        self.ignore_black_white_alpha_tiles_threshold = ignore_black_white_alpha_tiles_threshold

        self.scale_factor = scale_factor
        self.ground_resolution = ground_resolution

        self._check_parameters()

        self.raster = self._load_raster()

    def _check_parameters(self):
        assert self.raster_path.exists(), \
            f"Raster file not found at {self.raster_path}."
        assert isinstance(self.tile_size, int) and self.tile_size > 0, \
            "The tile size must be and integer greater than 0."
        assert isinstance(self.tile_size, float) and 0 <= self.tile_overlap < 1, \
            "The tile overlap must be between 0 and 1."
        assert not (self.ground_resolution and self.scale_factor), \
            "Both a ground_resolution and a scale_factor were provided. Please only specify one."

    def _load_raster(self):
        raster = Raster(path=self.raster_path,
                        output_name_suffix=self.output_name_suffix,
                        ground_resolution=self.ground_resolution,
                        scale_factor=self.scale_factor)
        return raster

    def _create_tiles(self) -> List[Tile]:
        width = self.raster.metadata['width']
        height = self.raster.metadata['height']

        print('Raster size: ', (height, width))
        print('Desired tile size: ', self.tile_size)
        print('Creating tiles and finding their associated labels...')

        tile_id_counter = 0
        tiles = []
        for row in tqdm(range(0, height, self.tile_coordinate_step), desc="Processing rows"):
            for col in range(0, width, self.tile_coordinate_step):
                window = rasterio.windows.Window(col, row, width=self.tile_size, height=self.tile_size)
                tile = self.raster.get_tile(window=window, tile_id=tile_id_counter)

                if self._check_skip_tile(tile=tile, tile_size=self.tile_size):
                    continue

                tiles.append(tile)
                tile_id_counter += 1

        return tiles

    def _get_tiles_per_aoi(self, tiles: List[Tile]):
        print('Assigning the tiles to the aois...')

        if self.aois_config is not None:
            if type(self.aois_config) is AOIGeneratorConfig:
                aoi_engine = AOIGeneratorForTiles(tiles=tiles,
                                                  tile_coordinate_step=self.tile_coordinate_step,
                                                  aois_config=cast(AOIGeneratorConfig, self.aois_config))
            elif type(self.aois_config) is AOIFromPackageConfig:
                aoi_engine = AOIFromPackageForTiles(tiles=tiles,
                                                    tile_coordinate_step=self.tile_coordinate_step,
                                                    aois_config=cast(AOIFromPackageConfig, self.aois_config),
                                                    associated_raster=self.raster,
                                                    ground_resolution=self.ground_resolution,
                                                    scale_factor=self.scale_factor)
            else:
                raise Exception(f'aois_config type unsupported: {type(self.aois_config)}')

            aois_tiles, aois_gdf = aoi_engine.get_aoi_tiles()

            # When the tiles are assigned to AOIs, sometimes part of them are black-out
            # (pixels within the tile that are outside the assigned AOI),
            # so we need to check again if the tile has too many black pixels, and remove it if yes.
            final_aois_tiles = {aoi: [] for aoi in aois_tiles}
            for aoi in aois_tiles:
                for tile in aois_tiles[aoi]:
                    if self._check_skip_tile(tile=tile, tile_size=self.tile_size):
                        continue
                    else:
                        final_aois_tiles[aoi].append(tile)
        else:
            final_aois_tiles = {}
            aois_gdf = None

        return final_aois_tiles, aois_gdf

    def _check_skip_tile(self, tile, tile_size):
        is_rgb = self.raster.data.shape[0] == 3
        is_rgba = self.raster.data.shape[0] == 4
        skip_ratio = self.ignore_black_white_alpha_tiles_threshold

        # Checking if the tile has more than a certain ratio of white, black, or alpha pixels.
        if is_rgb:
            if np.sum(tile.data == 0) / (tile_size * tile_size * 3) > skip_ratio:
                return True
            if np.sum(tile.data == 255) / (tile_size * tile_size * 3) > skip_ratio:
                return True
        elif is_rgba:
            if np.sum(tile.data[:-1] == 0) / (tile_size * tile_size * 3) > skip_ratio:
                return True
            if np.sum(tile.data[:-1] == 255) / (tile_size * tile_size * 3) > skip_ratio:
                return True
            if np.sum(tile.data[-1] == 0) / (tile_size * tile_size * 3) > skip_ratio:
                return True

        return False


class BaseDiskRasterTilerizer(BaseRasterTilerizer, ABC):
    """
    A base class for tilerizers that save tiles to the disk.

    Parameters
    ---------------------

    """
    def __init__(self,
                 raster_path: str,
                 output_path: str,
                 tile_size: int,
                 tile_overlap: float,
                 aois_config: AOIConfig = None,
                 ground_resolution: float = None,
                 scale_factor: float = None,
                 output_name_suffix: str = None,
                 ignore_black_white_alpha_tiles_threshold: float = 0.8):

        super().__init__(raster_path=raster_path,
                         tile_size=tile_size,
                         tile_overlap=tile_overlap,
                         aois_config=aois_config,
                         ground_resolution=ground_resolution,
                         scale_factor=scale_factor,
                         output_name_suffix=output_name_suffix,
                         ignore_black_white_alpha_tiles_threshold=ignore_black_white_alpha_tiles_threshold)

        self.output_path = Path(output_path) / self.raster.output_name
        self.tiles_path = self.output_path / 'tiles'
        self.tiles_path.mkdir(parents=True, exist_ok=True)


class RasterTilerizer(BaseDiskRasterTilerizer):
    """
    A standard tilerizer for Raster data without annotations or labels.

    Parameters
    ---------------------
    raster_path : Path
        Path to the raster (.tif, .png...).

    """

    def __init__(self,
                 raster_path: str or Path,
                 output_path: str or Path,
                 tile_size: int,
                 tile_overlap: float,
                 aois_config: AOIConfig = None,
                 ground_resolution: float = None,
                 scale_factor: float = None,
                 output_name_suffix: str = None,
                 ignore_black_white_alpha_tiles_threshold: float = 0.8):
        """
        raster_path: Path,
            Path to the raster (.tif, .png...).
        output_path: Path,
            Path to parent folder where to save the image tiles and associated labels.
        tile_size: int,
            The wanted size of the tiles (tile_size, tile_size).
        tile_overlap: float,
            The overlap between the tiles (should be 0 <= overlap < 1).
        aois_config: AOIConfig or None,
            An instance of AOIConfig to use, or None if all tiles should be kept in an 'all' AOI.
        ground_resolution: float,
            The ground resolution in meter per pixel desired when loading the raster.
            Only one of ground_resolution and scale_factor can be set at the same time.
        scale_factor: float,
            Scale factor for rescaling the data (change pixel resolution).
            Only one of ground_resolution and scale_factor can be set at the same time.
        ignore_black_white_alpha_tiles_threshold: bool,
            Whether to ignore (skip) mostly black or white (>ignore_black_white_alpha_tiles_threshold%) tiles.
        """

        super().__init__(raster_path=raster_path,
                         output_path=output_path,
                         tile_size=tile_size,
                         tile_overlap=tile_overlap,
                         aois_config=aois_config,
                         ground_resolution=ground_resolution,
                         scale_factor=scale_factor,
                         output_name_suffix=output_name_suffix,
                         ignore_black_white_alpha_tiles_threshold=ignore_black_white_alpha_tiles_threshold)

    def generate_tiles(self):
        tiles = self._create_tiles()
        aois_tiles, _ = self._get_tiles_per_aoi(tiles=tiles)
        aois_tiles['all'] = [tile for tile_list in aois_tiles.values() for tile in tile_list]

        save_aois_tiles_picture(aois_tiles=aois_tiles,
                                save_path=self.output_path / AoiTilesImageConvention.create_name(
                                    product_name=self.raster.output_name,
                                    ground_resolution=self.ground_resolution,
                                    scale_factor=self.scale_factor
                                ),
                                tile_coordinate_step=self.tile_coordinate_step)

        print("Saving tiles...")
        for aoi in aois_tiles:
            if aoi == 'all' and len(aois_tiles.keys()) > 1:
                # don't save the 'all' tiles if aois were provided.
                continue

            # Save the tile images
            for tile in aois_tiles[aoi]:
                tile.save(output_folder=self.tiles_path)

        print(f"The tiles have been saved to {self.tiles_path}.")


class RasterTilerizerGDF(BaseRasterTilerizer):
    """
    A standard tilerizer for Raster data without annotations or labels. Returns tiles extents as a GeoDataFrame.

    Parameters
    ----------
    raster_path : str or Path
        Path to the raster (.tif, .png...).
    tile_size : int
        The size of the tiles in pixels (tile_size, tile_size).
    tile_overlap : float
        The overlap between the tiles (0 <= overlap < 1).
    aois_config : AOIConfig or None, optional
        An instance of AOIConfig to use, or None if all tiles should be kept in an 'all' AOI.
    ground_resolution : float, optional
        The ground resolution in meter per pixel desired when loading the raster.
        Only one of ground_resolution and scale_factor can be set at the same time.
    scale_factor : float, optional
        Scale factor for rescaling the data (change pixel resolution).
        Only one of ground_resolution and scale_factor can be set at the same time.
    output_name_suffix : str, optional
        Suffix to add to the output file names.
    ignore_black_white_alpha_tiles_threshold : float, optional
        Threshold ratio of black, white or transparent pixels in a tile to skip it.
    """
    def __init__(self,
                 raster_path: str or Path,
                 tile_size: int,
                 tile_overlap: float,
                 aois_config: AOIConfig = None,
                 ground_resolution: float = None,
                 scale_factor: float = None,
                 output_name_suffix: str = None,
                 ignore_black_white_alpha_tiles_threshold: float = 0.8):
        """
        raster_path: Path,
            Path to the raster (.tif, .png...).
        tile_size: int,
            The wanted size of the tiles (tile_size, tile_size).
        tile_overlap: float,
            The overlap between the tiles (should be 0 <= overlap < 1).
        aois_config: AOIConfig or None,
            An instance of AOIConfig to use, or None if all tiles should be kept in an 'all' AOI.
        ground_resolution: float,
            The ground resolution in meter per pixel desired when loading the raster.
            Only one of ground_resolution and scale_factor can be set at the same time.
        scale_factor: float,
            Scale factor for rescaling the data (change pixel resolution).
            Only one of ground_resolution and scale_factor can be set at the same time.
        ignore_black_white_alpha_tiles_threshold: bool,
            Whether to ignore (skip) mostly black or white (>ignore_black_white_alpha_tiles_threshold%) tiles.
        """

        super().__init__(raster_path=raster_path,
                         tile_size=tile_size,
                         tile_overlap=tile_overlap,
                         aois_config=aois_config,
                         ground_resolution=ground_resolution,
                         scale_factor=scale_factor,
                         output_name_suffix=output_name_suffix,
                         ignore_black_white_alpha_tiles_threshold=ignore_black_white_alpha_tiles_threshold)

    def generate_tiles_gdf(self):
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

        if len(aois_tiles.keys()) > 1:
            tiles_gdf = tiles_gdf[tiles_gdf['aoi'] != 'all']

        return tiles_gdf



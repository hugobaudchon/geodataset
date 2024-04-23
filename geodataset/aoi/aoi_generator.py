import random
from typing import List

import numpy as np
import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame
from shapely import box

from .aoi_disambiguator import AOIDisambiguator
from .aoi_config import AOIGeneratorConfig
from .aoi_base import AOIBaseForTiles, AOIBaseGenerator
from geodataset.geodata import Tile
from ..utils import get_tiles_array


class AOIGeneratorForTiles(AOIBaseForTiles, AOIBaseGenerator):
    def __init__(self,
                 tiles: List[Tile],
                 tile_coordinate_step: int,
                 aois_config: AOIGeneratorConfig):
        """
        :param aois_config: An instanced AOIGeneratorConfig.
        """

        self.tiles_array = get_tiles_array(tiles=self.tiles, tile_coordinate_step=self.tile_coordinate_step)
        self.n_tiles = len(tiles)

        assert type(aois_config) is AOIGeneratorConfig

        AOIBaseForTiles.__init__(self, tiles=tiles, tile_coordinate_step=tile_coordinate_step)
        AOIBaseGenerator.__init__(self, aois_config=aois_config)

        self.aois = self.aois_config.aois
        self.aoi_type = self.aois_config.aoi_type

    def get_aoi_tiles(self) -> (dict[str, List[Tile]], dict[str, gpd.GeoDataFrame]):
        sorted_aois = {k: v for k, v in sorted(self.aois.items(), key=lambda item: (float('inf'),) if item[1]['position'] is None else (item[1]['position'],))}
        aois_tiles = {}
        for key in sorted_aois:
            max_n_tiles_in_aoi = np.round(sorted_aois[key]['percentage'] * self.n_tiles)
            n_tiles_left = np.sum(self.tiles_array[self.tiles_array == 1])
            if max_n_tiles_in_aoi > n_tiles_left:
                max_n_tiles_in_aoi = n_tiles_left

            if self.aoi_type == "corner":
                aoi_array = self._get_corner_aoi_array(max_n_tiles_in_aoi=max_n_tiles_in_aoi,
                                                       forced_position=sorted_aois[key]['percentage'])
            elif self.aoi_type == 'band':
                aoi_array = self._get_band_aoi_array(max_n_tiles_in_aoi=max_n_tiles_in_aoi,
                                                     forced_position=sorted_aois[key]['percentage'])
            else:
                raise Exception(f'aoi_type value \'{self.aoi_type}\' is not supported.')

            # Finding associated tile objects and storing them in a dict
            aois_tiles[key] = []
            for tile in self.tiles:
                if aoi_array[int(tile.row/self.tile_coordinate_step), int(tile.col/self.tile_coordinate_step)] == 2:
                    aois_tiles[key].append(tile)

            # Setting tiles already assigned to an AOI as 0 so that they don't get assigned again.
            self.tiles_array[aoi_array == 2] = 0

        # Getting gdfs of the AOIs and the tiles.
        tiles_gdfs = {}
        aois_gdfs = {}
        for aoi, tiles in aois_tiles.items():
            aoi_tiles_gdf = GeoDataFrame({'tile': tiles,
                                          'tile_id': [tile.tile_id for tile in tiles],
                                          'geometry': [box(tile.col,
                                                           tile.row,
                                                           tile.col + tile.metadata['width'],
                                                           tile.row + tile.metadata['height']) for tile in tiles]})

            aoi_tiles_gdf['aoi'] = aoi
            aoi_gdf = GeoDataFrame({'geometry': [aoi_tiles_gdf.geometry.unary_union]})
            aoi_gdf['aoi'] = aoi
            tiles_gdfs[aoi] = aoi_tiles_gdf
            aois_gdfs[aoi] = aoi_gdf

        # Blacking out the parts of the tiles that are not in their assigned AOI.
        tiles_gdf = gpd.GeoDataFrame(pd.concat(tiles_gdfs.values(), ignore_index=True)).reset_index()
        aois_gdf = gpd.GeoDataFrame(pd.concat(aois_gdfs.values(), ignore_index=True)).reset_index()

        aoi_disambiguator = AOIDisambiguator(
            tiles_gdf=tiles_gdf,
            aois_gdf=aois_gdf,
            aois_tiles=aois_tiles,
        )
        aoi_disambiguator.redistribute_generated_aois_intersections(aois_config=self.aois_config)
        aoi_disambiguator.disambiguate_tiles()

        return aois_tiles, aois_gdf

    def _get_corner_aoi_array(self,
                              max_n_tiles_in_aoi: int,
                              forced_position: int or None):
        array = self.tiles_array.copy()
        rows, cols = array.shape

        # Randomly choose a corner
        corner_choices = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]
        if forced_position:
            chosen_corner = corner_choices[forced_position - 1]
        else:
            chosen_corner = random.choice(corner_choices)

        # Rotate the array based on the chosen corner
        if chosen_corner == (0, cols - 1):
            # Rotate 90 degrees clockwise
            rotated_array = np.rot90(array, k=-1)
        elif chosen_corner == (rows - 1, 0):
            # Rotate 90 degrees counterclockwise
            rotated_array = np.rot90(array, k=1)
        elif chosen_corner == (rows - 1, cols - 1):
            # Rotate 180 degrees
            rotated_array = np.rot90(array, k=2)
        else:
            # No rotation needed
            rotated_array = array

        rotated_rows, rotated_cols = rotated_array.shape
        next_row_id, next_col_id = 1, 1
        while next_row_id < rotated_rows or next_col_id < rotated_cols:
            r_row = rotated_array[next_row_id, :next_col_id]
            r_col = rotated_array[:next_col_id + 1, next_row_id]
            row_item_id = 0
            while row_item_id < len(r_row) and np.sum(rotated_array == 2) < max_n_tiles_in_aoi:
                if r_row[row_item_id] == 1:
                    rotated_array[next_row_id, row_item_id] = 2
                row_item_id += 1

            if np.sum(rotated_array == 2) == max_n_tiles_in_aoi:
                break

            col_item_id = 0
            while col_item_id < len(r_col) and np.sum(rotated_array == 2) < max_n_tiles_in_aoi:
                if r_col[col_item_id] == 1:
                    rotated_array[col_item_id, next_col_id] = 2

                col_item_id += 1

            if np.sum(rotated_array == 2) == max_n_tiles_in_aoi:
                break

            if next_row_id + 1 < rotated_rows:
                next_row_id += 1
            if next_col_id + 1 < rotated_cols:
                next_col_id += 1

        # Rotate the array based on the chosen corner
        if chosen_corner == (0, cols - 1):
            # Rotate 90 degrees counterclockwise
            array = np.rot90(rotated_array, k=1)
        elif chosen_corner == (rows - 1, 0):
            # Rotate 90 degrees clockwise
            array = np.rot90(rotated_array, k=-1)
        elif chosen_corner == (rows - 1, cols - 1):
            # Rotate 180 degrees
            array = np.rot90(rotated_array, k=2)
        else:
            # No rotation needed
            array = rotated_array

        return array

    def _get_band_aoi_array(self,
                            max_n_tiles_in_aoi: int,
                            forced_position: int or None):
        array = self.tiles_array.copy()
        rows, cols = array.shape

        # Randomly choose to start from the left (0) or right (1) side
        if forced_position is None:
            start_side = random.choice([0, 1])  # 0 for left, 1 for right
            if start_side == 0:
                # Start from the left side
                current_col_id = 0
                direction = 1  # Move right
            else:
                # Start from the right side
                current_col_id = cols - 1
                direction = -1  # Move left
        else:
            current_col_id = 0
            direction = 1

        # Traverse the array in bands
        while 0 <= current_col_id < cols and np.sum(array == 2) < max_n_tiles_in_aoi:
            for row_id in range(rows):
                if array[row_id, current_col_id] == 1 and np.sum(array == 2) < max_n_tiles_in_aoi:
                    array[row_id, current_col_id] = 2

            current_col_id += direction
            if current_col_id == cols or current_col_id < 0:  # If we've reached an end, change direction
                direction *= -1
                current_col_id += direction  # Adjust to stay within bounds

        return array

import random
from pathlib import Path
from typing import List

import numpy as np
import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame

from .aoi_disambiguator import AOIDisambiguator
from .aoi_config import AOIGeneratorConfig
from .aoi_base import AOIBaseForTiles, AOIBaseGenerator
from geodataset.geodata import RasterTile, Raster
from ..utils import get_tiles_array


class AOIGeneratorForTiles(AOIBaseForTiles, AOIBaseGenerator):
    def __init__(self,
                 tiles: List[RasterTile],
                 tile_coordinate_step: int,
                 associated_raster: Raster,
                 global_aoi: str or Path or gpd.GeoDataFrame or None,
                 aois_config: AOIGeneratorConfig,
                 ignore_black_white_alpha_tiles_threshold: float = False):
        """
        :param aois_config: An instanced AOIGeneratorConfig.
        """

        assert type(aois_config) is AOIGeneratorConfig

        super().__init__(
            tiles=tiles,
            tile_coordinate_step=tile_coordinate_step,
            associated_raster=associated_raster,
            global_aoi=global_aoi,
            aois_config=aois_config,
            ignore_black_white_alpha_tiles_threshold=ignore_black_white_alpha_tiles_threshold
        )

        self.aois = self.aois_config.aois
        self.aoi_type = self.aois_config.aoi_type
        self.ignore_black_white_alpha_tiles_threshold = ignore_black_white_alpha_tiles_threshold

        self.n_tiles = len(tiles)
        self.tiles_array = get_tiles_array(
            tiles=self.tiles,
            tile_coordinate_step=self.tile_coordinate_step
        )

    def get_aoi_tiles(self) -> (dict[str, List[RasterTile]], dict[str, gpd.GeoDataFrame]):
        tiles_gdf = gpd.GeoDataFrame({'tile': self.tiles,
                                      'tile_id': [tile.tile_id for tile in self.tiles],
                                      'geometry': [tile.get_bbox() for tile in self.tiles],
                                      'array_row_coord': [int(tile.row / self.tile_coordinate_step) for tile in self.tiles],
                                      'array_col_coord': [int(tile.col / self.tile_coordinate_step) for tile in self.tiles]})

        # Intersecting the tiles with the global AOI
        if self.loaded_global_aoi is not None:
            global_tiles_gdf = gpd.overlay(tiles_gdf, self.loaded_global_aoi, how='intersection')
            global_tiles_gdf['intersection_area'] = global_tiles_gdf.geometry.area
            # Removing tiles that have less than ignore_black_white_alpha_tiles_threshold of their area
            # intersecting with the global AOI
            if self.ignore_black_white_alpha_tiles_threshold:
                global_tiles_gdf = global_tiles_gdf[global_tiles_gdf.intersection_area >= 0.1 * global_tiles_gdf.area]

            for tile_id in set(tiles_gdf['tile_id']):
                if tile_id not in set(global_tiles_gdf['tile_id']):
                    # If the tile is not intersecting the global AOI (or not enough), set its value in the tiles_array
                    # to 0, which means that it won't be assigned to any aoi and be ignored.
                    array_row_coord = tiles_gdf[tiles_gdf['tile_id'] == tile_id]['array_row_coord'].values[0]
                    array_col_coord = tiles_gdf[tiles_gdf['tile_id'] == tile_id]['array_col_coord'].values[0]
                    self.tiles_array[array_row_coord, array_col_coord] = 0

        aois_n_tiles = self._allocate_tiles(n_tiles=self.n_tiles)
        sorted_aois = {k: v for k, v in sorted(self.aois.items(), key=lambda item: (float('inf'),) if item[1]['position'] is None else (item[1]['position'],))}
        aois_tiles = {}
        for aoi in sorted_aois:
            max_n_tiles_in_aoi = aois_n_tiles[aoi]

            if self.aoi_type == "corner":
                aoi_array = self._get_corner_aoi_array(max_n_tiles_in_aoi=max_n_tiles_in_aoi,
                                                       forced_position=sorted_aois[aoi]['percentage'])
            elif self.aoi_type == 'band':
                aoi_array = self._get_band_aoi_array(max_n_tiles_in_aoi=max_n_tiles_in_aoi,
                                                     forced_position=sorted_aois[aoi]['percentage'])
            else:
                raise Exception(f'aoi_type value \'{self.aoi_type}\' is not supported.')

            # Finding associated tile objects and storing them in a dict
            actual_aoi_name = self.aois[aoi]['actual_name'] if 'actual_name' in self.aois[aoi] else aoi

            if actual_aoi_name not in aois_tiles:
                aois_tiles[actual_aoi_name] = []
            for tile in self.tiles:
                if aoi_array[int(tile.row/self.tile_coordinate_step), int(tile.col/self.tile_coordinate_step)] == 2:
                    tile = tile.copy_with_aoi_and_id(new_aoi=actual_aoi_name, new_id=tile.tile_id)
                    aois_tiles[actual_aoi_name].append(tile)

            # Setting tiles already assigned to an AOI as 0 so that they don't get assigned again.
            self.tiles_array[aoi_array == 2] = 0

        # Getting gdfs of the AOIs and the tiles.
        aois_gdfs = {}
        for actual_aoi_name, tiles in aois_tiles.items():
            aoi_tiles_gdf = GeoDataFrame({'tile': tiles,
                                          'tile_id': [tile.tile_id for tile in tiles],
                                          'geometry': [tile.get_bbox() for tile in tiles]})
            aoi_tiles_gdf['aoi'] = actual_aoi_name
            aoi_gdf = GeoDataFrame({'geometry': [aoi_tiles_gdf.geometry.unary_union]})
            aoi_gdf['aoi'] = actual_aoi_name
            aois_gdfs[actual_aoi_name] = aoi_gdf

        # Blacking out the parts of the tiles that are not in their assigned AOI.
        tiles_gdf = self.duplicate_tiles_at_aoi_intersection(aois_tiles=aois_tiles)
        aois_gdf = gpd.GeoDataFrame(pd.concat(aois_gdfs.values(), ignore_index=True)).reset_index()

        aoi_disambiguator = AOIDisambiguator(
            tiles_gdf=tiles_gdf,
            aois_gdf=aois_gdf,
            aois_tiles=aois_tiles,
        )
        aoi_disambiguator.redistribute_generated_aois_intersections(aois_config=self.aois_config)
        aoi_disambiguator.disambiguate_tiles()

        # aois_tiles, aois_gdf = self.use_actual_aois_names(self.aois_config, aois_tiles, aois_gdf) # this is now handled above in this method

        return aois_tiles, aois_gdf

    def _allocate_tiles(self, n_tiles):
        aois_n_tiles = {}

        # Sort AOIs by percentage in descending order
        sorted_aois = sorted(self.aois, key=lambda x: self.aois[x]['percentage'], reverse=True)

        total_percentage = sum(self.aois[aoi]['percentage'] for aoi in self.aois)

        # Initial allocation based on floor of (percentage * n_tiles)
        remainders = {}
        assigned_tiles = 0
        for aoi in sorted_aois:
            exact_allocation = (self.aois[aoi]['percentage'] / total_percentage) * n_tiles
            allocated = int(exact_allocation)
            aois_n_tiles[aoi] = allocated
            assigned_tiles += allocated
            remainders[aoi] = exact_allocation - allocated

        # Calculate remaining tiles to assign
        remaining_tiles = n_tiles - assigned_tiles

        # Distribute remaining tiles based on largest remainders
        while remaining_tiles > 0:
            # Find the AOI with the largest remainder
            aoi_to_assign = max(remainders, key=lambda x: remainders[x])
            aois_n_tiles[aoi_to_assign] += 1
            remainders[aoi_to_assign] = 0  # Set to zero to prevent re-selection
            remaining_tiles -= 1

        # Handle edge case where n_tiles is less than number of AOIs
        if n_tiles < len(self.aois):
            # Reset allocations
            aois_n_tiles = {aoi: 0 for aoi in self.aois}
            # Assign one tile to the top n_tiles AOIs based on percentage
            for aoi in sorted_aois[:n_tiles]:
                aois_n_tiles[aoi] = 1

        return aois_n_tiles

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

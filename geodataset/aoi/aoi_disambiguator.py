from typing import Dict, List

import geopandas as gpd
from shapely import Polygon

from shapely.affinity import translate

from .aoi_config import AOIGeneratorConfig
from geodataset.geodata import RasterTileMetadata
from ..utils import polygon_to_mask


class AOIDisambiguator:
    def __init__(self,
                 tiles_gdf: gpd.GeoDataFrame,
                 aois_tiles: Dict[str, List[RasterTileMetadata]],
                 aois_gdf: gpd.GeoDataFrame):
        self.tiles_gdf = tiles_gdf
        self.aois_tiles = aois_tiles
        self.aois_gdf = aois_gdf

    def disambiguate_tiles(self):
        # For each tile, black out the part of the tile which is not in its AOI.
        from shapely.ops import unary_union  # minimal change: ensure unary_union is imported
        for aoi, tiles in self.aois_tiles.items():
            # Get the subset of intersections for this AOI
            aoi_intersections = gpd.overlay(self.tiles_gdf[self.tiles_gdf['aoi'] == aoi],
                                            self.aois_gdf[self.aois_gdf['aoi'] == aoi],
                                            how='intersection')

            # For each tile relevant to this AOI
            for tile in tiles:
                # Retrieve the original tile geometry.
                original_tile_row = self.tiles_gdf[self.tiles_gdf['tile_id'] == tile.tile_id]
                original_tile_geom = original_tile_row.geometry.iloc[0]

                # Retrieve the intersection geometry by combining all parts.
                intersection_tile_row = aoi_intersections.loc[aoi_intersections['tile_id'] == tile.tile_id]
                intersection_geom = unary_union(intersection_tile_row.geometry)

                # Check if the tile is completely within the AOI.
                # Using 'equals' (or 'almost_equals' if desired) to allow for minor floating point differences.
                if original_tile_geom.equals(intersection_geom):
                    # The tile is completely inside the AOI, so skip updating the mask.
                    continue

                # Otherwise, translate the intersection geometry into the tile coordinate system.
                translated_geom = translate(intersection_geom, -tile.col, -tile.row)

                # Create a mask from the translated geometry and update the tile.
                mask = polygon_to_mask(translated_geom, tile.metadata['height'], tile.metadata['width'])
                tile.update_mask(mask)

    def redistribute_generated_aois_intersections(self, aois_config: AOIGeneratorConfig):
        self.aois_gdf['total_area'] = self.aois_gdf.geometry.area
        sorted_aois = self.aois_gdf.groupby('aoi').total_area.sum().sort_values().index.tolist()

        percentages = {}
        for aoi in aois_config.aois:
            actual_aoi_name = aois_config.aois[aoi]['actual_name'] if 'actual_name' in aois_config.aois[aoi] else aoi
            if actual_aoi_name not in percentages:
                percentages[actual_aoi_name] = 0
            percentages[actual_aoi_name] += aois_config.aois[aoi]['percentage']

        for idx_1, actual_aoi_name_1 in enumerate(sorted_aois):
            for idx_2, actual_aoi_name_2 in enumerate(sorted_aois):
                if idx_1 <= idx_2:
                    continue

                # Calculating percentage for area redistribution
                percent_1 = percentages[actual_aoi_name_1] / (percentages[actual_aoi_name_1] + percentages[actual_aoi_name_2])

                # Get the geometries for the two AOIs
                aoi_1_polygon = self.aois_gdf[self.aois_gdf['aoi'] == actual_aoi_name_1].geometry.values[0]
                aoi_2_polygon = self.aois_gdf[self.aois_gdf['aoi'] == actual_aoi_name_2].geometry.values[0]

                # Redistribute intersection
                aoi_1_polygon, aoi_2_polygon = self.redistribute_intersection(
                    aois_config=aois_config,
                    aoi_name_1=actual_aoi_name_1,
                    aoi_name_2=actual_aoi_name_2,
                    aoi_1_polygon=aoi_1_polygon,
                    aoi_2_polygon=aoi_2_polygon,
                    percent_1=percent_1
                )

                # Update the geometries in the original dataframe
                self.aois_gdf.loc[self.aois_gdf['aoi'] == actual_aoi_name_1, 'geometry'] = aoi_1_polygon
                self.aois_gdf.loc[self.aois_gdf['aoi'] == actual_aoi_name_2, 'geometry'] = aoi_2_polygon

    def redistribute_intersection(self,
                                  aois_config: AOIGeneratorConfig,
                                  aoi_name_1: str,
                                  aoi_name_2: str,
                                  aoi_1_polygon: Polygon,
                                  aoi_2_polygon: Polygon,
                                  percent_1: float,
                                  max_iterations=500):
        """
        Redistributes the intersection of two geometries based on the given percentage,
        aiming for a more complex and accurate handling of non-rectangular intersections.

        Parameters:
        - geom1, geom2: The input geometries (Shapely geometries).
        - x_percent: The percentage of the intersection area to add back to geom1.
        - max_iterations: The maximum number of iterations to refine the area adjustment.

        Returns:
        - Modified geom1 and geom2 with redistributed intersections.
        """
        assert 0 <= percent_1 <= 1, "The percentage should be between 0 and 1."

        actual_aoi_priority = {}
        for aoi in aois_config.aois:
            actual_aoi_name = aois_config.aois[aoi]['actual_name'] if 'actual_name' in aois_config.aois[aoi] else aoi
            if actual_aoi_name not in actual_aoi_priority:
                actual_aoi_priority[actual_aoi_name] = aois_config.aois[aoi].get('priority_aoi', False)
            else:
                actual_aoi_priority[actual_aoi_name] = actual_aoi_priority[actual_aoi_name] or aois_config.aois[aoi].get('priority_aoi', False)

        # Calculate the intersection
        intersection = aoi_1_polygon.intersection(aoi_2_polygon)
        if intersection.is_empty:
            pass
        elif actual_aoi_priority[aoi_name_1]:
            # keep aoi_1 unchanged and remove intersection from aoi_2
            aoi_2_polygon = aoi_2_polygon.difference(intersection)
        elif actual_aoi_priority[aoi_name_2]:
            # keep aoi_2 unchanged and remove intersection from aoi_1
            aoi_1_polygon = aoi_1_polygon.difference(intersection)
        else:
            # Remove the intersection from both geometries
            aoi_1_polygon = aoi_1_polygon.difference(intersection)
            aoi_2_polygon = aoi_2_polygon.difference(intersection)

            # Target areas to add back to each geometry based on the percentage
            target_area_1 = intersection.area * percent_1
            target_area_2 = intersection.area * (1 - percent_1)

            # Function to progressively adjust area
            def adjust_area(aoi_polygon, target_area, other_aoi_polygon=None):
                adjusted_polygon = aoi_polygon
                intersection_width = intersection.bounds[2] - intersection.bounds[0]
                intersection_height = intersection.bounds[3] - intersection.bounds[1]
                step = max(intersection_width, intersection_height) / max_iterations

                for _ in range(max_iterations):
                    added_area = adjusted_polygon.intersection(intersection).area
                    if added_area < target_area:
                        buffered_polygon = adjusted_polygon.buffer(step)
                        adjusted_polygon = adjusted_polygon.union(buffered_polygon.intersection(intersection))
                        if other_aoi_polygon:
                            adjusted_polygon = adjusted_polygon.difference(other_aoi_polygon)

                        new_added_area = adjusted_polygon.intersection(intersection).area
                        if new_added_area == added_area:
                            break
                    else:
                        break

                return adjusted_polygon

            # Adjust the areas, do the smaller AOI first.
            if target_area_1 < target_area_2:
                adjusted_part_1 = adjust_area(aoi_1_polygon, target_area_1)
                adjusted_part_2 = adjust_area(aoi_2_polygon, target_area_2, other_aoi_polygon=adjusted_part_1)
            else:
                adjusted_part_2 = adjust_area(aoi_2_polygon, target_area_2)
                adjusted_part_1 = adjust_area(aoi_1_polygon, target_area_1, other_aoi_polygon=adjusted_part_2)

            # Add back the adjusted parts to the original geometries
            aoi_1_polygon = aoi_1_polygon.union(adjusted_part_1)
            aoi_2_polygon = aoi_2_polygon.union(adjusted_part_2)

        return aoi_1_polygon, aoi_2_polygon

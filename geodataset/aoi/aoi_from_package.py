from typing import List

import geopandas as gpd
import pandas as pd
from shapely import MultiPolygon, Polygon, box

from .aoi_config import AOIFromPackageConfig
from .aoi_base import AOIBase
from geodataset.geodata import Tile, Raster


class AOIFromPackage(AOIBase):
    def __init__(self,
                 tiles: List[Tile],
                 tile_size: int,
                 tile_overlap: float,
                 aois_config: AOIFromPackageConfig,
                 associated_raster: Raster,
                 scale_factor: float):
        """
        :param aois_config: An instanced AOIFromPackageConfig.
        """
        super().__init__(tiles=tiles,
                         tile_size=tile_size,
                         tile_overlap=tile_overlap,
                         aois_config=aois_config)

        self.associated_raster = associated_raster
        self.scale_factor = scale_factor

        assert type(aois_config) is AOIFromPackageConfig

        self.loaded_aois = self._load_aois()

    def _load_aois(self):
        """
        Load the AOI from the provided path, converting it to a MultiPolygon if necessary.
        """

        loaded_aois = {}
        for aoi_name in self.aois_config.aois:
            # Load the AOI using geopandas
            aoi_gdf = gpd.read_file(self.aois_config.aois[aoi_name])

            # Ensure the geometry is a MultiPolygon
            aoi_gdf['geometry'] = aoi_gdf['geometry'].astype(object).apply(
                lambda geom: MultiPolygon([geom]) if isinstance(geom, Polygon) else geom
            )

            # Making sure the geometries have the same CRS as the raster
            aoi_gdf = self.associated_raster.adjust_geometries_to_raster_crs_if_necessary(gdf=aoi_gdf)

            # Scaling the geometries to pixel coordinates aligned with the Raster
            aoi_gdf = self.associated_raster.adjust_geometries_to_raster_pixel_coordinates(gdf=aoi_gdf)

            # Store the loaded data
            loaded_aois[aoi_name] = aoi_gdf

        return loaded_aois

    def get_aoi_tiles(self):

        aois_frames = []
        for aoi, gdf in self.loaded_aois.items():
            gdf = gdf.copy()
            gdf['aoi'] = aoi
            aois_frames.append(gdf)
        aois_gdf = gpd.GeoDataFrame(pd.concat(aois_frames, ignore_index=True)).reset_index()

        tiles_gdf = gpd.GeoDataFrame({'tile': self.tiles,
                                      'tile_id': [tile.tile_id for tile in self.tiles],
                                      'geometry': [box(tile.col,
                                                       tile.row,
                                                       tile.col + tile.metadata['width'],
                                                       tile.row + tile.metadata['height']) for tile in self.tiles]})

        intersections = gpd.overlay(tiles_gdf, aois_gdf, how='intersection')
        intersections['intersection_area'] = intersections.geometry.area
        max_intersection_per_tile = intersections.loc[intersections.groupby('tile_id')['intersection_area'].idxmax()]
        aois_tiles = max_intersection_per_tile.groupby('aoi')['tile'].apply(list).to_dict()

        return aois_tiles

from typing import List

import geopandas as gpd
import pandas as pd
from shapely import box

from .aoi_disambiguator import AOIDisambiguator
from .aoi_config import AOIFromPackageConfig
from .aoi_base import AOIBaseForTiles, AOIBaseForPolygons, AOIBaseFromPackage
from geodataset.geodata import Tile, Raster
from ..labels import RasterPolygonLabels


class AOIFromPackageForTiles(AOIBaseForTiles, AOIBaseFromPackage):
    def __init__(self,
                 tiles: List[Tile],
                 tile_coordinate_step: int,
                 aois_config: AOIFromPackageConfig,
                 associated_raster: Raster,
                 ground_resolution: float,
                 scale_factor: float):
        """
        :param aois_config: An instanced AOIFromPackageConfig.
        """

        self.associated_raster = associated_raster
        self.ground_resolution = ground_resolution
        self.scale_factor = scale_factor

        assert self.ground_resolution == self.associated_raster.ground_resolution, \
            "The specified ground_resolution for the labels and Raster are different."
        assert self.scale_factor == self.associated_raster.scale_factor, \
            "The specified scale_factor for the labels and Raster are different."
        assert type(aois_config) is AOIFromPackageConfig

        AOIBaseForTiles.__init__(self, tiles=tiles, tile_coordinate_step=tile_coordinate_step)
        AOIBaseFromPackage.__init__(self, associated_raster=associated_raster, aois_config=aois_config)

    def get_aoi_tiles(self) -> (dict[str, List[Tile]], dict[str, gpd.GeoDataFrame]):

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

        tiles_gdf = gpd.GeoDataFrame(tiles_gdf.merge(max_intersection_per_tile[['tile_id', 'aoi']], on='tile_id', how='left'))
        tiles_gdf = tiles_gdf.dropna(subset=['aoi'])

        aoi_disambiguator = AOIDisambiguator(
            tiles_gdf=tiles_gdf,
            aois_tiles=aois_tiles,
            aois_gdf=aois_gdf
        )
        aoi_disambiguator.disambiguate_tiles()

        return aois_tiles, aois_gdf


class AOIFromPackageForPolygons(AOIBaseForPolygons, AOIBaseFromPackage):
    def __init__(self,
                 labels: RasterPolygonLabels,
                 associated_raster: Raster,
                 aois_config: AOIFromPackageConfig):
        """
        :param aois_config: An instanced AOIFromPackageConfig.
        """

        assert type(aois_config) is AOIFromPackageConfig

        AOIBaseForPolygons.__init__(self, labels=labels)
        AOIBaseFromPackage.__init__(self, associated_raster=associated_raster, aois_config=aois_config)

    def get_aoi_polygons(self) -> (dict[str, gpd.GeoDataFrame], dict[str, gpd.GeoDataFrame]):

        aois_frames = []
        for aoi, gdf in self.loaded_aois.items():
            gdf = gdf.copy()
            gdf['aoi'] = aoi
            aois_frames.append(gdf)
        aois_gdf = gpd.GeoDataFrame(pd.concat(aois_frames, ignore_index=True)).reset_index()

        polygons = self.labels.geometries_gdf.copy()
        polygons['geometry_id'] = polygons.index

        intersections = gpd.overlay(polygons, aois_gdf, how='intersection')
        intersections['intersection_area'] = intersections.geometry.area

        max_intersection_per_polygon = intersections.groupby('geometry_id')['intersection_area'].idxmax()

        aois_polygons_ids = intersections.loc[max_intersection_per_polygon].groupby('aoi')['geometry_id'].apply(list).to_dict()

        aois_polygons = {aoi: polygons.loc[aois_polygons_ids[aoi]] for aoi in aois_polygons_ids}

        return aois_polygons, aois_gdf

from pathlib import Path
from typing import List

import geopandas as gpd

from .aoi_disambiguator import AOIDisambiguator
from .aoi_config import AOIFromPackageConfig
from .aoi_base import AOIBaseForTiles, AOIBaseForPolygons, AOIBaseFromPackage
from geodataset.geodata import RasterTileMetadata, Raster
from ..labels import RasterPolygonLabels


class AOIFromPackageForTiles(AOIBaseForTiles, AOIBaseFromPackage):
    def __init__(self,
                 tiles: List[RasterTileMetadata],
                 tile_coordinate_step: int,
                 associated_raster: Raster,
                 global_aoi: str or Path or gpd.GeoDataFrame or None,
                 aois_config: AOIFromPackageConfig,
                 ground_resolution: float,
                 scale_factor: float):
        """
        :param aois_config: An instanced AOIFromPackageConfig.
        """

        super().__init__(
            tiles=tiles,
            tile_coordinate_step=tile_coordinate_step,
            associated_raster=associated_raster,
            global_aoi=global_aoi,
            aois_config=aois_config
        )

        self.ground_resolution = ground_resolution
        self.scale_factor = scale_factor

        assert self.ground_resolution == self.associated_raster.ground_resolution, \
            "The specified ground_resolution for the labels and Raster are different."
        assert self.scale_factor == self.associated_raster.scale_factor, \
            "The specified scale_factor for the labels and Raster are different."
        assert type(aois_config) is AOIFromPackageConfig

    def get_aoi_tiles(self) -> (dict[str, List[RasterTileMetadata]], dict[str, gpd.GeoDataFrame]):
        aois_gdf = self._get_aois_gdf()

        tiles_gdf = gpd.GeoDataFrame({'tile': self.tiles,
                                      'tile_id': [tile.tile_id for tile in self.tiles],
                                      'geometry': [tile.get_bbox() for tile in self.tiles]})

        intersections = gpd.overlay(tiles_gdf, aois_gdf, how='intersection')
        intersections['intersection_area'] = intersections.geometry.area
        aois_tiles = intersections.groupby('aoi')['tile'].apply(list).to_dict()

        final_aoi_tiles_gdf = self.duplicate_tiles_at_aoi_intersection(aois_tiles=aois_tiles)

        aoi_disambiguator = AOIDisambiguator(
            tiles_gdf=final_aoi_tiles_gdf,
            aois_tiles=aois_tiles,
            aois_gdf=aois_gdf
        )
        aoi_disambiguator.disambiguate_tiles()
        
        aois_tiles, aois_gdf = self.use_actual_aois_names(self.aois_config, aois_tiles, aois_gdf)

        return aois_tiles, aois_gdf


class AOIFromPackageForPolygons(AOIBaseForPolygons, AOIBaseFromPackage):
    def __init__(self,
                 labels: RasterPolygonLabels,
                 associated_raster: Raster,
                 global_aoi: str or Path or gpd.GeoDataFrame or None,
                 aois_config: AOIFromPackageConfig):
        """
        :param aois_config: An instanced AOIFromPackageConfig.
        """

        assert type(aois_config) is AOIFromPackageConfig

        super().__init__(
            labels=labels,
            associated_raster=associated_raster,
            global_aoi=global_aoi,
            aois_config=aois_config
        )

    def get_aoi_polygons(self) -> (dict[str, gpd.GeoDataFrame], dict[str, gpd.GeoDataFrame]):
        aois_gdf = self._get_aois_gdf()

        polygons = self.labels.geometries_gdf.copy()
        polygons['geometry_id'] = polygons.index

        if self.loaded_global_aoi is not None:
            polygons = gpd.overlay(polygons, self.loaded_global_aoi, how='intersection')

        intersections = gpd.overlay(polygons, aois_gdf, how='intersection')
        intersections['intersection_area'] = intersections.geometry.area

        max_intersection_per_polygon = intersections.groupby('geometry_id')['intersection_area'].idxmax()

        aois_polygons_ids = intersections.loc[max_intersection_per_polygon].groupby('aoi')['geometry_id'].apply(list).to_dict()

        aois_polygons = {aoi: polygons.loc[aois_polygons_ids[aoi]] for aoi in aois_polygons_ids}
        
        aois_polygons, aois_gdf = self.use_actual_aois_names(self.aois_config, aois_polygons, aois_gdf)

        return aois_polygons, aois_gdf

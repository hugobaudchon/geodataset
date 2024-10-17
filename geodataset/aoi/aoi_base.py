from abc import ABC, abstractmethod
from typing import List

from shapely import MultiPolygon, Polygon
import geopandas as gpd
import pandas as pd

from .aoi_config import AOIFromPackageConfig, AOIGeneratorConfig, AOIConfig
from geodataset.geodata import RasterTile, Raster
from ..labels import RasterPolygonLabels

class AOIForRasterBase(ABC):
    @staticmethod
    def use_actual_aois_names(aois_config: AOIConfig, aois_tiles: dict[str, List[RasterTile]], aois_gdf: gpd.GeoDataFrame):
        new_aois_tiles = {}
        for aoi in aois_tiles:
            tiles = aois_tiles[aoi]

            if 'actual_name' in aois_config.aois[aoi]:
                actual_name = aois_config.aois[aoi]['actual_name']
                aois_gdf.loc[aois_gdf['aoi'] == aoi, 'aoi'] = actual_name
                for tile in tiles:
                    tile.aoi = actual_name

                if actual_name in new_aois_tiles:
                    new_aois_tiles[actual_name] += tiles
                else:
                    new_aois_tiles[actual_name] = tiles
            else:
                if aoi in new_aois_tiles:
                    new_aois_tiles[aoi] += tiles
                else:
                    new_aois_tiles[aoi] = tiles

        return new_aois_tiles, aois_gdf


class AOIBaseForTiles(AOIForRasterBase, ABC):
    def __init__(self,
                 tiles: List[RasterTile],
                 tile_coordinate_step: int):
        """
        :param tiles: A list of instanced Tile.
        :param tile_coordinate_step: The step in pixels between each tile.
        """
        self.tiles = tiles
        self.tile_coordinate_step = tile_coordinate_step

    @abstractmethod
    def get_aoi_tiles(self, *args, **kwargs) -> dict[str, dict]:
        pass

    def duplicate_tiles_at_aoi_intersection(self, aois_tiles: dict[str, List[RasterTile]]) -> gpd.GeoDataFrame:
        """
        Duplicate each Tile object that is in multiple AOI into multiple Tile objects (one per AOI).
        """
        # adding aoi data info to each Tile, duplicating each Tile object that is in multiple aoi into multiple Tile objects (one per aoi)
        current_max_tile_id = max([tile.tile_id for tile in self.tiles])
        final_aoi_tiles_gdfs = []
        for aoi in aois_tiles:
            for i in range(len(aois_tiles[aoi])):
                aois_tiles[aoi][i] = aois_tiles[aoi][i].copy_with_aoi_and_id(new_aoi=aoi,
                                                                             new_id=current_max_tile_id + 1)  # doing it like this to avoid duplicating all tiles in memory, which could lead to OOM issues
                final_aoi_tiles_gdfs.append(gpd.GeoDataFrame({'tile': [aois_tiles[aoi][i]],
                                                              'tile_id': [aois_tiles[aoi][i].tile_id],
                                                              'geometry': [aois_tiles[aoi][i].get_bbox()],
                                                              'aoi': [aoi]}))
                current_max_tile_id += 1

        return gpd.GeoDataFrame(pd.concat(final_aoi_tiles_gdfs, ignore_index=True)).reset_index()


class AOIBaseForPolygons(AOIForRasterBase, ABC):
    def __init__(self,
                 labels: RasterPolygonLabels):
        """
        :param labels: An instanced RasterPolygonLabels.
        """
        self.labels = labels

    @abstractmethod
    def get_aoi_polygons(self, *args, **kwargs) -> dict[str, dict]:
        pass


class AOIBaseGenerator(AOIForRasterBase, ABC):
    def __init__(self,
                 aois_config: AOIGeneratorConfig):
        """
        :param aois_config: An instanced AOIGeneratorConfig.
        """
        self.aois_config = aois_config


class AOIBaseFromPackage(AOIForRasterBase, ABC):
    def __init__(self,
                 associated_raster: Raster,
                 aois_config: AOIFromPackageConfig):
        """
        :param associated_raster: An instanced Raster.
        :param aois_config: An instanced AOIFromPackageConfig.
        """
        self.aois_config = aois_config
        self.associated_raster = associated_raster

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

class AOIBaseFromGeoFileInCRS:
    def __init__(self, aois_config: AOIFromPackageConfig):
        
        self.aois_config = aois_config
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

            loaded_aois[aoi_name] = aoi_gdf

        return loaded_aois
    
    def get_aoi_gdf(self):
        aois_frames = []

        for aoi, gdf in self.loaded_aois.items():
            gdf = gdf.copy()
            gdf["aoi"] = aoi
            aois_frames.append(gdf)

        aois_gdf = gpd.GeoDataFrame(pd.concat(aois_frames, ignore_index=True)).reset_index()

        return aois_gdf
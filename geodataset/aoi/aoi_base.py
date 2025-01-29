from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from shapely import MultiPolygon, Polygon
import geopandas as gpd
import pandas as pd

from .aoi_config import AOIFromPackageConfig, AOIGeneratorConfig, AOIConfig
from geodataset.geodata import RasterTile, Raster
from ..labels import RasterPolygonLabels

class AOIForRasterBase(ABC):
    def __init__(self,
                 associated_raster: Raster,
                 global_aoi: str or Path or gpd.GeoDataFrame or None,
                 **kwargs):

        self.associated_raster = associated_raster
        self.global_aoi = global_aoi

    @staticmethod
    def _get_aoi_gdf(aoi_name: str, aoi_item: str or Path or gpd.GeoDataFrame):
        if isinstance(aoi_item, (str, Path)):
            aoi_gdf = gpd.read_file(aoi_item)
        elif isinstance(aoi_item, gpd.GeoDataFrame):
            aoi_gdf = aoi_item
        else:
            raise ValueError(f"AOI {aoi_name} is not a string, pathlib.Path or GeoDataFrame.")

        return aoi_gdf

    def _align_aoi_gdf(self, gdf):
        # Ensure the geometry is a MultiPolygon
        gdf['geometry'] = gdf['geometry'].astype(object).apply(
            lambda geom: MultiPolygon([geom]) if isinstance(geom, Polygon) else geom
        )
        # Making sure the geometries have the same CRS as the raster
        gdf = self.associated_raster.adjust_geometries_to_raster_crs_if_necessary(gdf=gdf)
        # Scaling the geometries to pixel coordinates aligned with the Raster
        gdf = self.associated_raster.adjust_geometries_to_raster_pixel_coordinates(gdf=gdf)
        return gdf

    @staticmethod
    def use_actual_aois_names(aois_config: AOIConfig, aois_tiles: dict[str, List[RasterTile]], aois_gdf: gpd.GeoDataFrame):
        if isinstance(aois_config, AOIGeneratorConfig):
            # If the AOI has an 'actual_name' field, use it to rename the AOI in the tiles and gdf
            # This is useful when the AOI is named 'train1', 'train2', 'valid'...
            # and we want to rename them to 'train', 'valid'...
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
        else:
            return aois_tiles, aois_gdf


class AOIBaseForTiles(AOIForRasterBase, ABC):
    def __init__(self,
                 tiles: List[RasterTile],
                 tile_coordinate_step: int,
                 associated_raster: Raster,
                 global_aoi: str or Path or gpd.GeoDataFrame or None,
                 **kwargs):
        """
        :param tiles: A list of instanced Tile.
        :param tile_coordinate_step: The step in pixels between each tile.
        """
        super().__init__(associated_raster=associated_raster, global_aoi=global_aoi, **kwargs)
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
                 labels: RasterPolygonLabels,
                 associated_raster: Raster,
                 global_aoi: str or Path or gpd.GeoDataFrame or None,
                 **kwargs):
        """
        :param labels: An instanced RasterPolygonLabels.
        """
        super().__init__(associated_raster=associated_raster, global_aoi=global_aoi, **kwargs)
        self.labels = labels

    @abstractmethod
    def get_aoi_polygons(self, *args, **kwargs) -> dict[str, dict]:
        pass


class AOIBaseGenerator(AOIForRasterBase, ABC):
    def __init__(self,
                 associated_raster: Raster,
                 global_aoi: str or Path or gpd.GeoDataFrame or None,
                 aois_config: AOIGeneratorConfig,
                 **kwargs):
        """
        :param aois_config: An instanced AOIGeneratorConfig.
        """
        super().__init__(associated_raster=associated_raster, global_aoi=global_aoi, **kwargs)

        self.aois_config = aois_config

        if self.global_aoi is not None:
            self.loaded_global_aoi = self._get_aoi_gdf('global', self.global_aoi)
            self.loaded_global_aoi = self._align_aoi_gdf(self.loaded_global_aoi)
        else:
            self.loaded_global_aoi = None


class AOIBaseFromPackage(AOIForRasterBase, ABC):
    def __init__(self,
                 associated_raster: Raster,
                 global_aoi: str or Path or gpd.GeoDataFrame or None,
                 aois_config: AOIFromPackageConfig,
                 **kwargs):
        """
        :param associated_raster: An instanced Raster.
        :param aois_config: An instanced AOIFromPackageConfig.
        """

        super().__init__(associated_raster=associated_raster, global_aoi=global_aoi, **kwargs)

        self.aois_config = aois_config
        self.loaded_aois, self.loaded_global_aoi = self._load_aois()

    def _load_aois(self):
        """
        Load the AOI from the provided path, converting it to a MultiPolygon if necessary.
        """

        loaded_aois = {}
        for aoi_name in self.aois_config.aois:
            # Load the AOI using geopandas
            aoi_gdf = self._get_aoi_gdf(aoi_name, self.aois_config.aois[aoi_name])
            # Store the loaded data
            loaded_aois[aoi_name] = self._align_aoi_gdf(aoi_gdf)

        # Loading the global AOI if it exists
        if self.global_aoi is not None:
            global_aoi_gdf = self._get_aoi_gdf('global', self.global_aoi)
            loaded_global_aoi_gdf = self._align_aoi_gdf(global_aoi_gdf)
        else:
            loaded_global_aoi_gdf = None

        return loaded_aois, loaded_global_aoi_gdf

    def _get_aois_gdf(self):
        aois_frames = []
        for aoi, gdf in self.loaded_aois.items():
            gdf = gdf.copy()
            gdf['aoi'] = aoi
            aois_frames.append(gdf)

        aois_gdf = gpd.GeoDataFrame(pd.concat(aois_frames, ignore_index=True)).reset_index()

        # Intersecting with the global AOI if it was provided
        if self.loaded_global_aoi is not None:
            aois_gdf = gpd.overlay(aois_gdf, self.loaded_global_aoi, how='intersection')

        return aois_gdf

class AOIBaseFromGeoFileInCRS:
    def __init__(self, aois_config: AOIFromPackageConfig, **kwargs):
        
        self.aois_config = aois_config
        self.loaded_aois = self._load_aois()

    def _get_aoi_gdf(self, aoi_name):
        if isinstance(self.aois_config.aois[aoi_name], (str, Path)):
            aoi_gdf = gpd.read_file(self.aois_config.aois[aoi_name])
        elif isinstance(self.aois_config.aois[aoi_name], gpd.GeoDataFrame):
            aoi_gdf = self.aois_config.aois[aoi_name]
        else:
            raise ValueError(f"AOI {aoi_name} is not a string, pathlib.Path or GeoDataFrame.")

        return aoi_gdf
    
    def _load_aois(self):
        """
        Load the AOI from the provided path, converting it to a MultiPolygon if necessary.
        """

        loaded_aois = {}
        for aoi_name in self.aois_config.aois:
            # Load the AOI using geopandas
            aoi_gdf = self._get_aoi_gdf(aoi_name)

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
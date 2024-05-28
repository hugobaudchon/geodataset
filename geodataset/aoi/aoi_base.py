from abc import ABC, abstractmethod
from typing import List

from shapely import MultiPolygon, Polygon
import geopandas as gpd

from .aoi_config import AOIFromPackageConfig, AOIGeneratorConfig
from geodataset.geodata import Tile, Raster
from ..labels import RasterPolygonLabels


class AOIBaseForTiles(ABC):
    def __init__(self,
                 tiles: List[Tile],
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


class AOIBaseForPolygons(ABC):
    def __init__(self,
                 labels: RasterPolygonLabels):
        """
        :param labels: An instanced RasterPolygonLabels.
        """
        self.labels = labels

    @abstractmethod
    def get_aoi_polygons(self, *args, **kwargs) -> dict[str, dict]:
        pass


class AOIBaseGenerator(ABC):
    def __init__(self,
                 aois_config: AOIGeneratorConfig):
        """
        :param aois_config: An instanced AOIGeneratorConfig.
        """
        self.aois_config = aois_config


class AOIBaseFromPackage(ABC):
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

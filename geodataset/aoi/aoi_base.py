from abc import ABC, abstractmethod
from typing import List

from .aoi_config import AOIConfig
from geodataset.geodata import Tile


class AOIBase(ABC):
    def __init__(self,
                 tiles: List[Tile],
                 tile_coordinate_step: int,
                 aois_config: AOIConfig):
        """
        :param aois_config: An instanced AOIConfig.
        """
        self.tiles = tiles
        self.tile_coordinate_step = tile_coordinate_step
        self.aois_config = aois_config

    @abstractmethod
    def get_aoi_tiles(self, *args, **kwargs) -> dict[str, dict]:
        pass


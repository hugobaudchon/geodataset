from typing import Dict, List

from geodataset.aoi import AOIGeneratorConfig, AOIFromPackageConfig
from geodataset.geodata import Tile


class AOIDisambiguator:
    def __init__(self, aois_tiles: Dict[str, List[Tile]], aois_config: AOIGeneratorConfig or AOIFromPackageConfig):
        self.aois_tiles = aois_tiles
        self.aois_config = aois_config

    def disambiguate(self):
        if isinstance(self.aois_config, AOIGeneratorConfig):
            self.disambiguate_generated_aois()
        elif isinstance(self.aois_config, AOIFromPackageConfig):
            self.disambiguate_geopackage_aois()

    def disambiguate_generated_aois(self):
        pass

    def disambiguate_geopackage_aois(self):
        # For each tile, black out the part of the tile which is not in its AOI.
        for aoi, tiles in aois_tiles.items():
            for tile in tiles:
                # Get the intersection of the tile and the AOI
                intersection = intersections[(intersections['tile_id'] == tile.tile_id) & (intersections['aoi'] == aoi)]
                if not intersection.empty:
                    # Convert the intersection geometry to a format that rasterio can understand
                    shapes = [mapping(intersection.geometry.values[0])]
                    # Create a mask where the pixels inside the AOI are set to 1 and the pixels outside the AOI are set to 0
                    mask, _, _ = raster_geometry_mask(tile.data, shapes, invert=True)
                    # Black out the part of the tile which is not in its AOI by multiplying the tile by the mask
                    tile.data = tile.data * (~mask)

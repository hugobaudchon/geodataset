import re
from abc import ABC, abstractmethod


def validate_and_convert_product_name(product_name_stem: str):
    standardized_product_name_stem = product_name_stem.replace(" ", "_").replace("-", "_")
    standardized_product_name_stem = standardized_product_name_stem.lower()

    pattern = r"^([a-z0-9]+_)+[a-z0-9]+$"
    if not re.match(pattern, standardized_product_name_stem):
        raise ValueError(f"The product name stem (without extension) {product_name_stem},"
                         f" which has been standardized to {standardized_product_name_stem}"
                         f" does not match the expected format {pattern}.")

    return standardized_product_name_stem


class FileNameConvention(ABC):
    @staticmethod
    def create_specifier(scale_factor=None, ground_resolution=None):
        if scale_factor is not None:
            return f"sf{str(scale_factor).replace('.', 'p')}"
        elif ground_resolution is not None:
            return f"gr{str(ground_resolution).replace('.', 'p')}"
        else:
            return FileNameConvention.create_specifier(scale_factor=1.0)

    @staticmethod
    def parse_specifier(specifier):
        if 'sf' in specifier:
            return float(specifier.replace('sf', '').replace('p', '.')), None
        elif 'gr' in specifier:
            return None, float(specifier.replace('gr', '').replace('p', '.'))
        else:
            raise ValueError("Specifier must contain either 'sf' for scale factor or 'gr' for ground resolution.")

    @staticmethod
    @abstractmethod
    def _validate_name(**kwargs):
        pass

    @staticmethod
    @abstractmethod
    def create_name(**kwargs):
        pass

    @staticmethod
    @abstractmethod
    def parse_name(**kwargs):
        pass


class TileNameConvention(FileNameConvention):
    @staticmethod
    def _validate_name(name):
        pattern = r"^([a-zA-Z0-9]+_)*[a-zA-Z0-9]+_tile_((sf[0-9]+p[0-9]+)|(gr[0-9]+p[0-9]+))_[0-9]+_[0-9]+\.tif$"
        if not re.match(pattern, name):
            raise ValueError(f"tile_name {name} does not match the expected format {pattern}.")

    @staticmethod
    def create_name(product_name: str, col: int, row: int, scale_factor=None, ground_resolution=None):
        specifier = FileNameConvention.create_specifier(scale_factor=scale_factor, ground_resolution=ground_resolution)
        tile_name = f"{product_name}_tile_{specifier}_{col}_{row}.tif"
        TileNameConvention._validate_name(tile_name)
        return tile_name

    @staticmethod
    def parse_name(tile_name: str):
        TileNameConvention._validate_name(tile_name)

        parts = tile_name.split("_")
        product_name = "_".join(parts[:-4])
        specifier = parts[-3]
        col = parts[-2]
        row = parts[-1].replace('.tif', '')

        scale_factor, ground_resolution = FileNameConvention.parse_specifier(specifier)

        return product_name, scale_factor, ground_resolution, col, row


class PolygonTileNameConvention(FileNameConvention):
    @staticmethod
    def _validate_name(name):
        pattern = r"^([a-zA-Z0-9]+_)*[a-zA-Z0-9]+_polygontile_((sf[0-9]+p[0-9]+)|(gr[0-9]+p[0-9]+))_[0-9]+\.tif$"
        if not re.match(pattern, name):
            raise ValueError(f"tile_name {name} does not match the expected format {pattern}.")

    @staticmethod
    def create_name(product_name: str, polygon_id: int, scale_factor=None, ground_resolution=None):
        specifier = FileNameConvention.create_specifier(scale_factor=scale_factor, ground_resolution=ground_resolution)
        tile_name = f"{product_name}_polygontile_{specifier}_{polygon_id}.tif"
        PolygonTileNameConvention._validate_name(tile_name)
        return tile_name

    @staticmethod
    def parse_name(tile_name: str):
        PolygonTileNameConvention._validate_name(tile_name)

        parts = tile_name.split("_")
        product_name = "_".join(parts[:-3])
        specifier = parts[-2]
        polygon_id = parts[-1].replace('.tif', '')

        scale_factor, ground_resolution = FileNameConvention.parse_specifier(specifier)

        return product_name, scale_factor, ground_resolution, polygon_id


class CocoNameConvention(FileNameConvention):
    @staticmethod
    def _validate_name(name):
        pattern = r"^([a-zA-Z0-9]+_)*[a-zA-Z0-9]+_coco_((sf[0-9]+p[0-9]+)|(gr[0-9]+p[0-9]+))_[a-zA-Z0-9]+\.json$"
        if not re.match(pattern, name):
            raise ValueError(f"coco_name {name} does not match the expected format {pattern}.")

    @staticmethod
    def create_name(product_name: str, fold: str, scale_factor=None, ground_resolution=None):
        specifier = FileNameConvention.create_specifier(scale_factor=scale_factor, ground_resolution=ground_resolution)
        coco_name = f"{product_name}_coco_{specifier}_{fold}.json"
        CocoNameConvention._validate_name(coco_name)
        return coco_name

    @staticmethod
    def parse_name(coco_name: str):
        CocoNameConvention._validate_name(coco_name)

        parts = coco_name.split("_")
        product_name = "_".join(parts[:-3])
        specifier = parts[-2]
        fold = parts[-1].replace(".json", "")

        scale_factor, ground_resolution = FileNameConvention.parse_specifier(specifier)

        return product_name, scale_factor, ground_resolution, fold


class GeoJsonNameConvention(FileNameConvention):
    @staticmethod
    def _validate_name(name):
        pattern = r"^([a-zA-Z0-9]+_)*[a-zA-Z0-9]+_((sf[0-9]+p[0-9]+)|(gr[0-9]+p[0-9]+))_[a-zA-Z0-9]+\.geojson$"
        if not re.match(pattern, name):
            raise ValueError(f"geojson_name {name} does not match the expected format {pattern}.")

    @staticmethod
    def create_name(product_name: str, fold: str, scale_factor=None, ground_resolution=None):
        specifier = FileNameConvention.create_specifier(scale_factor=scale_factor, ground_resolution=ground_resolution)
        geojson_name = f"{product_name}_{specifier}_{fold}.geojson"
        GeoJsonNameConvention._validate_name(geojson_name)
        return geojson_name

    @staticmethod
    def parse_name(geojson_name: str):
        GeoJsonNameConvention._validate_name(geojson_name)

        parts = geojson_name.split("_")
        product_name = "_".join(parts[:-2])
        specifier = parts[-2]
        fold = parts[-1].replace(".geojson", "")

        scale_factor, ground_resolution = FileNameConvention.parse_specifier(specifier)

        return product_name, scale_factor, ground_resolution, fold


class AoiTilesImageConvention(FileNameConvention):
    @staticmethod
    def _validate_name(name):
        pattern = r"^([a-zA-Z0-9]+_)*[a-zA-Z0-9]+_aoistiles_((sf[0-9]+p[0-9]+)|(gr[0-9]+p[0-9]+))\.png$"
        if not re.match(pattern, name):
            raise ValueError(f"file_name {name} does not match the expected format {pattern}.")

    @staticmethod
    def create_name(product_name: str, scale_factor=None, ground_resolution=None):
        specifier = FileNameConvention.create_specifier(scale_factor=scale_factor, ground_resolution=ground_resolution)
        file_name = f"{product_name}_aoistiles_{specifier}.png"
        AoiTilesImageConvention._validate_name(file_name)
        return file_name

    @staticmethod
    def parse_name(file_name: str):
        AoiTilesImageConvention._validate_name(file_name)

        parts = file_name.split("_")
        product_name = "_".join(parts[:-2])
        specifier = parts[-1].replace(".png", "")

        scale_factor, ground_resolution = FileNameConvention.parse_specifier(specifier)

        return product_name, scale_factor, ground_resolution



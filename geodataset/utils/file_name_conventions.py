import re
from abc import ABC, abstractmethod


def validate_and_convert_product_name(product_name_stem):
    standardized_product_name_stem = product_name_stem.replace(" ", "_").replace("-", "_")

    pattern = r"^([a-zA-Z0-9]+_)+[a-zA-Z0-9]+$"
    if not re.match(pattern, standardized_product_name_stem):
        raise ValueError(f"The product name stem (without extension) {product_name_stem},"
                         f" which has been standardized to {standardized_product_name_stem}"
                         f" does not match the expected format {pattern}.")

    return standardized_product_name_stem


class FileNameConvention(ABC):
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
        pattern = r"^([a-zA-Z0-9]+_)+[a-zA-Z0-9]+_tile_[0-9]+_[0-9]+_[0-9]+\.tif$"
        if not re.match(pattern, name):
            raise ValueError(f"tile_name {name} does not match the expected format {pattern}.")

    @staticmethod
    def create_name(product_name: str, col: int, row: int, scale_factor: float):
        tile_name = f"{product_name}_tile_{int(scale_factor*100)}_{col}_{row}.tif"
        TileNameConvention._validate_name(tile_name)
        return tile_name

    @staticmethod
    def parse_name(tile_name: str):
        TileNameConvention._validate_name(tile_name)

        parts = tile_name.split("_")

        product_name = "_".join(parts[:-4])
        scale_factor = int(parts[-3])/100
        col = parts[-2]
        row = parts[-1]

        return product_name, scale_factor, col, row


class CocoNameConvention(FileNameConvention):
    @staticmethod
    def _validate_name(name):
        # Adjusted pattern for COCO naming convention
        pattern = r"^([a-zA-Z0-9]+_)+[a-zA-Z0-9]+_coco_[0-9]+_[a-zA-Z0-9]+\.json$"
        if not re.match(pattern, name):
            raise ValueError(f"coco_name {name} does not match the expected format {pattern}.")

    @staticmethod
    def create_name(product_name: str, fold: str, scale_factor: float):
        coco_name = f"{product_name}_coco_{int(scale_factor*100)}_{fold}.json"
        CocoNameConvention._validate_name(coco_name)
        return coco_name

    @staticmethod
    def parse_name(coco_name: str):
        CocoNameConvention._validate_name(coco_name)

        parts = coco_name.split("_")

        product_name = "_".join(parts[:-3])
        scale_factor = int(parts[-2]) / 100
        fold = parts[-1].replace(".json", "")

        return product_name, scale_factor, fold


class AoiTilesImageConvention(FileNameConvention):
    @staticmethod
    def _validate_name(name):
        # Pattern for AOI Tiles image naming convention
        pattern = r"^([a-zA-Z0-9]+_)+[a-zA-Z0-9]+_aoistiles_[0-9]+\.png$"
        if not re.match(pattern, name):
            raise ValueError(f"file_name {name} does not match the expected format {pattern}.")

    @staticmethod
    def create_name(product_name: str, scale_factor: float):
        file_name = f"{product_name}_aoistiles_{int(scale_factor*100)}.png"
        AoiTilesImageConvention._validate_name(file_name)
        return file_name

    @staticmethod
    def parse_name(file_name: str):
        AoiTilesImageConvention._validate_name(file_name)

        parts = file_name.split("_")

        product_name = "_".join(parts[:-2])
        scale_factor = int(parts[-1].replace(".png", "")) / 100

        return product_name, scale_factor



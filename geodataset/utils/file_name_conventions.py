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
    def create_specifier(scale_factor=None, ground_resolution=None, voxel_size=None):
        if scale_factor is not None:
            specifier = f"sf{str(scale_factor).replace('.', 'p')}"
        elif ground_resolution is not None:
            specifier =  f"gr{str(ground_resolution).replace('.', 'p')}"
        else:
            return FileNameConvention.create_specifier(scale_factor=1.0, voxel_size=voxel_size)
        if voxel_size is not None:
            specifier = specifier + f"_vs{str(voxel_size).replace('.', 'p')}"

        return specifier

    @staticmethod
    def parse_specifier(specifier):
        scale_factor = None
        ground_resolution = None
        voxel_size = None

        if 'sf' in specifier:
            # Extract the part after 'sf' up to the next specifier or end of string
            sf_part = specifier.split('sf')[1].split('_')[0]
            scale_factor = float(sf_part.replace('p', '.'))
        if 'gr' in specifier:
            gr_part = specifier.split('gr')[1].split('_')[0]
            ground_resolution = float(gr_part.replace('p', '.'))
        if 'vs' in specifier:
            vs_part = specifier.split('vs')[1].split('_')[0]
            voxel_size = float(vs_part.replace('p', '.'))

        if scale_factor is None and ground_resolution is None:
            raise ValueError("Specifier must contain either 'sf' for scale factor or 'gr' for ground resolution.")

        return scale_factor, ground_resolution, voxel_size
    
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
    pattern_aoi = r"^([a-zA-Z0-9]+_)*[a-zA-Z0-9]+_tile_[a-zA-Z]+_((sf[0-9]+p[0-9]+)|(gr[0-9]+p[0-9]+))_[0-9]+_[0-9]+\.tif$"
    pattern_no_aoi = r"^([a-zA-Z0-9]+_)*[a-zA-Z0-9]+_tile_((sf[0-9]+p[0-9]+)|(gr[0-9]+p[0-9]+))_[0-9]+_[0-9]+\.tif$"

    @staticmethod
    def _validate_name(name):
        if re.match(TileNameConvention.pattern_aoi, name):
            return 'pattern_aoi'
        elif re.match(TileNameConvention.pattern_no_aoi, name):
            return 'pattern_no_aoi'
        else:
            raise ValueError(f"tile_name {name} does not match any of the supported patterns.")


    @staticmethod
    def create_name(product_name: str, col: int, row: int, scale_factor=None, ground_resolution=None, aoi=None):
        specifier = FileNameConvention.create_specifier(scale_factor=scale_factor, ground_resolution=ground_resolution)
        tile_name = f"{product_name}_tile_{aoi if aoi else 'noaoi'}_{specifier}_{col}_{row}.tif"
        TileNameConvention._validate_name(tile_name)
        return tile_name

    @staticmethod
    def parse_name(tile_name: str):
        pattern_name = TileNameConvention._validate_name(tile_name)

        parts = tile_name.split("_")
        row = int(parts[-1].replace('.tif', ''))
        col = int(parts[-2])
        specifier = parts[-3]
        scale_factor, ground_resolution, _ = FileNameConvention.parse_specifier(specifier)

        if pattern_name == 'pattern_aoi':
            aoi = parts[-4]
            if aoi == 'noaoi':
                aoi = None
            product_name = "_".join(parts[:-5])
        elif pattern_name == 'pattern_no_aoi':
            aoi = None
            product_name = "_".join(parts[:-4])
        else:
            raise ValueError(f"pattern_name {pattern_name} is not supported.")

        return product_name, scale_factor, ground_resolution, col, row, aoi



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

        scale_factor, ground_resolution, _ = FileNameConvention.parse_specifier(specifier)

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

        scale_factor, ground_resolution, _ = FileNameConvention.parse_specifier(specifier)

        return product_name, scale_factor, ground_resolution, fold

class PointCloudCocoNameConvention(FileNameConvention):
    @staticmethod
    def _validate_name(name):
        pattern = r'^.*pc_coco_((sf|gr)[0-9]+p[0-9]+_)?(vs[0-9]+p[0-9]+_)?[a-zA-Z0-9]+\.json$'
        if not re.match(pattern, name):
            raise ValueError(f"coco_name {name} does not match the expected format {pattern}.")

    @staticmethod
    def create_name(product_name: str, fold: str, scale_factor=None, ground_resolution=None, voxel_size=None):
        specifier = FileNameConvention.create_specifier(scale_factor=scale_factor, ground_resolution=ground_resolution, voxel_size=voxel_size)
        coco_name = f"{product_name}_pc_coco_{specifier}_{fold}.json"
        PointCloudCocoNameConvention._validate_name(coco_name)
        return coco_name

    @staticmethod
    def parse_name(coco_name: str):
        PointCloudCocoNameConvention._validate_name(coco_name)

        parts = coco_name.split("pc_coco_")
        product_name = parts[0]
        specifier, fold_extension = parts[1].rsplit("_", 1)
        fold, extension = fold_extension.split(".")

        scale_factor, ground_resolution, voxel_size = FileNameConvention.parse_specifier(specifier)

        return product_name, scale_factor, ground_resolution, voxel_size, fold 
    
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

        scale_factor, ground_resolution, _ = FileNameConvention.parse_specifier(specifier)

        return product_name, scale_factor, ground_resolution, fold


class GeoPackageNameConvention(FileNameConvention):
    @staticmethod
    def _validate_name(name):
        pattern = r"^([a-zA-Z0-9]+_)*[a-zA-Z0-9]+_((sf[0-9]+p[0-9]+)|(gr[0-9]+p[0-9]+))_[a-zA-Z0-9]+\.gpkg$"
        if not re.match(pattern, name):
            raise ValueError(f"geopackage_name {name} does not match the expected format {pattern}.")

    @staticmethod
    def create_name(product_name: str, fold: str, scale_factor=None, ground_resolution=None):
        specifier = FileNameConvention.create_specifier(scale_factor=scale_factor, ground_resolution=ground_resolution,)
        geojson_name = f"{product_name}_{specifier}_{fold}.gpkg"
        GeoPackageNameConvention._validate_name(geojson_name)
        return geojson_name

    @staticmethod
    def parse_name(geopackage_name: str):
        GeoPackageNameConvention._validate_name(geopackage_name)

        parts = geopackage_name.split("_")
        product_name = "_".join(parts[:-2])
        specifier = parts[-2]
        fold = parts[-1].replace(".gpkg", "")

        scale_factor, ground_resolution, _ = FileNameConvention.parse_specifier(specifier)

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

        scale_factor, ground_resolution, _ = FileNameConvention.parse_specifier(specifier)

        return product_name, scale_factor, ground_resolution


class PointCloudTileNameConvention(FileNameConvention):
    @staticmethod
    def _validate_name(name):
        pattern = r"^.*pc_tile_((sf|gr)[0-9]+p[0-9]+_)?(vs[0-9]+p[0-9]+_)?[0-9]+_[0-9]+_id_[0-9]+\.(ply|las)$"
        if not re.match(pattern, name):
            raise ValueError(f"tile_name {name} does not match the expected format {pattern}.")
        else:
            return True

    @staticmethod
    def create_name(product_name: str, tile_id:str, scale_factor=None, ground_resolution=None, voxel_size=None, extension='pcd', row=None, col=None):
        assert extension in ["pcd", "ply", "las"], f"Extension must be either 'ply' or 'las', not {extension}."
        specifier = FileNameConvention.create_specifier(scale_factor=scale_factor, ground_resolution=ground_resolution, voxel_size=voxel_size)
        tile_name = f"{product_name}_pc_tile_{specifier}_{row}_{col}_id_{tile_id}.{extension}"
        PointCloudTileNameConvention._validate_name(tile_name)
        return tile_name

    @staticmethod
    def parse_name(tile_name: str):
        PointCloudTileNameConvention._validate_name(tile_name)

        parts = tile_name.split("_pc_tile_")
        product_name = parts[0]
        specifier, col, row, _, tile_id_extension = parts[1].rsplit("_", 4)
        tile_id, extension = tile_id_extension.split(".")

        scale_factor, ground_resolution, voxel_size = FileNameConvention.parse_specifier(specifier)

        return product_name, scale_factor, ground_resolution, voxel_size, int(col), int(row), int(tile_id)
    
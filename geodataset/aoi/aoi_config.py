import re
import geopandas as gpd
from abc import ABC, abstractmethod
from pathlib import Path


class AOIConfig(ABC):
    def __init__(self, aois: dict):
        self.aois = aois
        self.actual_names = self._check_aois_names()

    def _check_aois_names(self):
        actual_names = set()
        for aoi_name in self.aois.keys():
            assert re.match(r'^[a-zA-Z0-9]+$', aoi_name), (
                "The AOI name should only contain alphanumeric characters like 'train', 'test1', 'test2'... "
            )
            if isinstance(self.aois[aoi_name], dict) and 'actual_name' in self.aois[aoi_name]:
                actual_name = self.aois[aoi_name]['actual_name']
                assert re.match(r'^[a-zA-Z0-9]+$', actual_name), (
                    f"The AOI {aoi_name}'s 'actual_name={actual_name}' should only contain alphanumeric characters like 'train', 'test1', 'test2'... "
                )
                actual_names.add(actual_name)
            else:
                actual_names.add(aoi_name)

        return actual_names

    @abstractmethod
    def _check_config(self):
        pass


class AOIFromPackageConfig(AOIConfig):
    """
    Configuration class for AOIFromPackage.

    Parameters
    ----------
    aois: dict
        A dictionary containing the name of the AOI as key and the path to the AOI (.gpkg, .geojson...) as value.
    """

    def __init__(self, aois: dict):
        super().__init__(aois=aois)
        self._check_config()

    def _check_config(self):
        assert type(self.aois) is dict, f"The provided aois value is not a dict, it is a {type(self.aois)}"

        for aoi_name in self.aois:
            aoi_path = self.aois[aoi_name]
            assert isinstance(aoi_name, str), (
                f"The keys in aois should be string, for the name of the aoi (train, valid, test...)."
                f" Found {aoi_name} which is not a string.")
            assert isinstance(aoi_path, (str, Path, gpd.GeoDataFrame)), \
                f"The value associated to aoi {aoi_name} is not a string or pathlib.Path. Got value {type(aoi_path)}."

            if isinstance(aoi_path, str):
                self.aois[aoi_name] = Path(aoi_path)


class AOIGeneratorConfig(AOIConfig):
    """
    Configuration class for AOIGenerator.

    Parameters
    ----------
    aoi_type: str
        The type of AOI to generate. Supported values are 'band' and 'corner'.
    aois: dict
        A dictionary containing the name(s) of the AOI(s) as key and the position and percentage of the AOI as value:
        {key: {'position': int (unique integer going from 1 to n_aois), 'percentage': float}}
    """

    SUPPORTED_AOI_TYPES = ['band', 'corner']

    def __init__(self, aoi_type: str, aois: dict):
        super().__init__(aois=aois)
        self.aoi_type = aoi_type
        self._check_config()

    def _check_config(self):
        assert type(self.aois) is dict, f"The provided aois value is not a dict, it is a {type(self.aois)}"
        assert self.aoi_type in self.SUPPORTED_AOI_TYPES, (f"The specified aoi_type {self.aoi_type} is not supported."
                                                           f" Valid values are {self.SUPPORTED_AOI_TYPES}")

        priority_aoi = None
        positions = []
        percentages = []
        for aoi_name in self.aois:
            aoi = self.aois[aoi_name]
            assert 'position' in aoi, f"The aoi '{aoi_name}' doesn't have a 'position' value."
            assert type(aoi['position']) is int or aoi['position'] is None, \
                f"The 'position' value of aoi {aoi_name} is not an int or None. Got value {aoi['position']}"
            assert 'percentage' in aoi, f"The aoi '{aoi_name}' doesn't have a 'percentage' value."
            assert type(aoi['percentage']) is float or aoi['percentage'] == 1, \
                f"The 'percentage' value of aoi {aoi_name} is not a float. Got value {aoi['percentage']}"

            if 'priority_aoi' in aoi and aoi['priority_aoi']:
                assert priority_aoi is None, "Only one AOI can be set as 'priority_aoi=True'."
                priority_aoi = aoi_name

            positions.append(aoi['position'])
            percentages.append(aoi['percentage'])

        assert (sum(percentages)) <= 1, "The sum of the aois 'percentage' should be less or equal to 1."

        if self.aoi_type == 'band':
            if None in positions:
                assert len(set(positions)) != 1, \
                    ("If one of the positions is set to None, then all positions should be set to None."
                     " (None will randomize the aois position)")
            else:
                assert len(set(positions)) == len(positions),\
                    f"The 'position' values should be different for each aoi. Got values {positions}."
                assert min(positions) == 1 and max(positions) == len(positions),\
                    f"The 'position' values should be from 1 to n_aois. Got values {positions}"
        elif self.aoi_type == 'corner':
            if None in positions:
                assert len(set(positions)) != 1, \
                    ("If one of the positions is set to None, then all positions should be set to None."
                     " (None will randomize the aois position)")
            else:
                assert len(self.aois) <= 4, \
                        (f"If using aoi_type='corner', a maximum of 4 aois can be defined, one for each corner."
                         f" Got {len(self.aois)} aois.")
                assert len(set(positions)) == len(positions), \
                    f"The 'position' values should be different for each aoi. Got values {positions}."
                assert min(positions) == 1 and max(positions) == len(positions), \
                    f"The 'position' values should be from 1 to n_aois. Got values {positions}"

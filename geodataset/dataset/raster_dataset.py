from pathlib import Path
from typing import List

from geodataset.dataset import BaseDataset
from geodataset.geodata.base_geodata import BaseGeoData


class RasterDataset(BaseDataset):
    def __init__(self, tiles_paths: List[Path]):
        self.data = 0

    def __len__(self):
        a = 0

    def __getitem__(self, index):
        a = 0


class LabeledRasterDataset:
    def __init__(self, data: BaseGeoData):
        self.data = data

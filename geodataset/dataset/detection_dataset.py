from geodataset.geodata.base_geodata import BaseGeoData
from geodataset.labels.raster_labels import RasterDetectionLabels


class DetectionRasterDataset:
    def __init__(self, data: BaseGeoData):
        self.data = data


class LabeledDetectionRasterDataset(DetectionRasterDataset):
    def __init__(self, data: BaseGeoData, labels: RasterDetectionLabels):
        super().__init__(data=data)
        self.labels = labels

from geodataset.geodata.base_geodata import BaseGeoData
from geodataset.labels.detection_labels import DetectionLabels


class DetectionRasterDataset:
    def __init__(self, data: BaseGeoData):
        self.data = data


class LabeledDetectionRasterDataset(DetectionRasterDataset):
    def __init__(self, data: BaseGeoData, labels: DetectionLabels):
        super().__init__(data=data)
        self.labels = labels
